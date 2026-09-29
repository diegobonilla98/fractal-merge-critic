from __future__ import annotations

import inspect
import math

import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from ..candidates import vision_adjacent_pairs
from ..geometry import vision_grid_geometry
from ..losses import cosine_damage
from ..types import TokenState
from .base import FrozenBackboneAdapter


class CLIPVisionAdapter(FrozenBackboneAdapter):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", device: str = "cuda", dtype: torch.dtype = torch.bfloat16, proportional_attention: bool = True):
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name, dtype=dtype).to(device)
        self.device = torch.device(device)
        self.proportional_attention = proportional_attention
        super().__init__()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def n_layers(self) -> int:
        return len(self.model.vision_model.encoder.layers)

    def prepare(self, item) -> dict[str, torch.Tensor]:
        values = self.processor(images=item, return_tensors="pt")["pixel_values"].to(self.device)
        return {"pixel_values": values.to(dtype=next(self.model.parameters()).dtype)}

    def _attention_bias(self, mass: torch.Tensor, query_count: int) -> torch.Tensor | None:
        if not self.proportional_attention:
            return None
        if mass.ndim == 1:
            mass = mass.unsqueeze(0)
        key_bias = torch.log(mass.clamp_min(1.0)).to(dtype=mass.dtype)
        return key_bias[:, None, None, :].expand(-1, 1, query_count, -1)

    def _layer(self, layer, hidden, attention_mask=None):
        params = inspect.signature(layer.forward).parameters
        kwargs = {}
        if "attention_mask" in params:
            kwargs["attention_mask"] = attention_mask
        if "causal_attention_mask" in params:
            kwargs["causal_attention_mask"] = None
        if "output_attentions" in params:
            kwargs["output_attentions"] = False
        out = layer(hidden, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def to_layer(self, batch: dict[str, torch.Tensor], layer_idx: int) -> TokenState:
        pixel_values = batch["pixel_values"]
        hidden = self.model.vision_model.embeddings(pixel_values)
        hidden = self.model.vision_model.pre_layrnorm(hidden)
        for layer in self.model.vision_model.encoder.layers[:layer_idx]:
            hidden = self._layer(layer, hidden)
        hidden = hidden[0]
        patch_count = hidden.shape[0] - 1
        side = int(round(math.sqrt(patch_count)))
        if side * side != patch_count:
            raise ValueError("CLIP adapter currently expects a square patch grid")
        geom = vision_grid_geometry(side, side, hidden.device, hidden.dtype)
        special_geom = torch.zeros((1, geom.shape[-1]), device=geom.device, dtype=geom.dtype)
        geometry = torch.cat([special_geom, geom], dim=0)
        mass = torch.ones(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        special = torch.zeros(hidden.shape[0], device=hidden.device, dtype=torch.bool)
        special[0] = True
        groups = [[-1]] + [[i] for i in range(patch_count)]
        return TokenState(hidden, mass, geometry, special, groups, {"original_grid_hw": (side, side), "prefix_tokens": 1, "modality": "vision"})

    def advance(self, state: TokenState, start_layer: int, end_layer: int) -> TokenState:
        out = state.clone()
        hidden = out.hidden.unsqueeze(0)
        attention_bias = self._attention_bias(out.mass, out.n_tokens)
        for layer in self.model.vision_model.encoder.layers[start_layer:end_layer]:
            hidden = self._layer(layer, hidden, attention_bias)
        out.hidden = hidden[0]
        return out

    def from_layer(self, state: TokenState, layer_idx: int) -> torch.Tensor:
        final_state = self.advance(state, layer_idx, self.n_layers)
        hidden = final_state.hidden.unsqueeze(0)
        pooled = self.model.vision_model.post_layernorm(hidden[:, 0])
        projected = self.model.visual_projection(pooled)
        return torch.nn.functional.normalize(projected, dim=-1)[0]

    def from_layer_batch(self, states: list[TokenState], layer_idx: int) -> torch.Tensor:
        hidden = torch.stack([state.hidden for state in states])
        mass = torch.stack([state.mass for state in states])
        attention_bias = self._attention_bias(mass, hidden.shape[1])
        for layer in self.model.vision_model.encoder.layers[layer_idx:]:
            hidden = self._layer(layer, hidden, attention_bias)
        pooled = self.model.vision_model.post_layernorm(hidden[:, 0])
        projected = self.model.visual_projection(pooled)
        return torch.nn.functional.normalize(projected, dim=-1)

    def candidates(self, state: TokenState) -> torch.Tensor:
        return vision_adjacent_pairs(state)

    def damage(self, reference: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
        return cosine_damage(reference.unsqueeze(0), counterfactual.unsqueeze(0))[0]

    def damage_batch(self, reference: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
        return cosine_damage(reference.unsqueeze(0).expand_as(counterfactual), counterfactual)
