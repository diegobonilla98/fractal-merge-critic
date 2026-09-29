from __future__ import annotations

import math

import torch
from transformers import AutoImageProcessor, AutoModel

from ..candidates import vision_adjacent_pairs
from ..geometry import vision_grid_geometry
from ..losses import cosine_damage
from ..types import TokenState
from .base import FrozenBackboneAdapter


class DINOv3VisionAdapter(FrozenBackboneAdapter):
    def __init__(self, model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m", device: str = "cuda", dtype: torch.dtype = torch.bfloat16, proportional_attention: bool = True):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device)
        self.device = torch.device(device)
        self.proportional_attention = proportional_attention
        super().__init__()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def n_layers(self) -> int:
        return len(self.model.layer)

    def prepare(self, item) -> dict[str, torch.Tensor]:
        values = self.processor(images=item, return_tensors="pt")["pixel_values"].to(self.device)
        return {"pixel_values": values.to(dtype=next(self.model.parameters()).dtype)}

    def to_layer(self, batch: dict[str, torch.Tensor], layer_idx: int) -> TokenState:
        pixel_values = batch["pixel_values"]
        hidden = self.model.embeddings(pixel_values)
        original_position_embeddings = self.model.rope_embeddings(pixel_values)
        for layer in self.model.layer[:layer_idx]:
            hidden = layer(hidden, position_embeddings=original_position_embeddings)
        hidden = hidden[0]
        prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
        patch_count = hidden.shape[0] - prefix
        side = int(round(math.sqrt(patch_count)))
        if side * side != patch_count:
            raise ValueError("DINOv3 adapter currently expects a square patch grid")
        geom = vision_grid_geometry(side, side, hidden.device, hidden.dtype)
        special_geom = torch.zeros((prefix, geom.shape[-1]), device=geom.device, dtype=geom.dtype)
        geometry = torch.cat([special_geom, geom], dim=0)
        mass = torch.ones(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        special = torch.zeros(hidden.shape[0], device=hidden.device, dtype=torch.bool)
        special[:prefix] = True
        groups = [[-i - 1] for i in range(prefix)] + [[i] for i in range(patch_count)]
        aux = {
            "original_grid_hw": (side, side),
            "prefix_tokens": prefix,
            "position_embeddings": tuple(x.detach() for x in original_position_embeddings),
            "modality": "vision",
        }
        return TokenState(hidden, mass, geometry, special, groups, aux)

    def _reduced_rope(self, state: TokenState):
        cos, sin = state.aux["position_embeddings"]
        cos = cos[0] if cos.ndim == 3 else cos
        sin = sin[0] if sin.ndim == 3 else sin
        patch_groups = [g for idx, g in enumerate(state.groups) if not state.special_mask[idx]]
        if all(len(group) == 1 and group[0] == index for index, group in enumerate(patch_groups)):
            return cos, sin
        out_cos = []
        out_sin = []
        for group in patch_groups:
            index = torch.tensor(group, device=cos.device, dtype=torch.long)
            c = cos[index].mean(dim=0)
            s = sin[index].mean(dim=0)
            norm = torch.sqrt(c.pow(2) + s.pow(2)).clamp_min(1e-8)
            out_cos.append(c / norm)
            out_sin.append(s / norm)
        return torch.stack(out_cos), torch.stack(out_sin)

    def _attention_bias(self, mass: torch.Tensor, query_count: int) -> torch.Tensor | None:
        if not self.proportional_attention:
            return None
        if mass.ndim == 1:
            mass = mass.unsqueeze(0)
        key_bias = torch.log(mass.clamp_min(1.0)).to(dtype=mass.dtype)
        return key_bias[:, None, None, :].expand(-1, 1, query_count, -1)

    def advance(self, state: TokenState, start_layer: int, end_layer: int) -> TokenState:
        out = state.clone()
        hidden = out.hidden.unsqueeze(0)
        position_embeddings = self._reduced_rope(out)
        attention_bias = self._attention_bias(out.mass, out.n_tokens)
        for layer in self.model.layer[start_layer:end_layer]:
            hidden = layer(hidden, attention_mask=attention_bias, position_embeddings=position_embeddings)
        out.hidden = hidden[0]
        return out

    def from_layer(self, state: TokenState, layer_idx: int) -> torch.Tensor:
        final_state = self.advance(state, layer_idx, self.n_layers)
        hidden = self.model.norm(final_state.hidden.unsqueeze(0))
        cls = hidden[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1)[0]

    def candidates(self, state: TokenState) -> torch.Tensor:
        return vision_adjacent_pairs(state)

    def damage(self, reference: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
        return cosine_damage(reference.unsqueeze(0), counterfactual.unsqueeze(0))[0]

    def damage_batch(self, reference: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
        return cosine_damage(reference.unsqueeze(0).expand_as(counterfactual), counterfactual)
