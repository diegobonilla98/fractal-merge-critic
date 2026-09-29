from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..candidates import text_adjacent_pairs
from ..geometry import text_geometry
from ..losses import symmetric_kl
from ..types import TokenState
from .base import FrozenBackboneAdapter


class Qwen3Adapter(FrozenBackboneAdapter):
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str = "cuda", dtype: torch.dtype = torch.bfloat16, proportional_attention: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)
        self.device = next(self.model.parameters()).device
        self.proportional_attention = proportional_attention
        if any(x != "full_attention" for x in getattr(self.model.config, "layer_types", ["full_attention"] * self.model.config.num_hidden_layers)):
            raise ValueError("Qwen3Adapter currently supports dense full-attention Qwen3 checkpoints only")
        super().__init__()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def n_layers(self) -> int:
        return len(self.model.model.layers)

    def prepare(self, item: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(item, return_tensors="pt", add_special_tokens=True)
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _position_ids(self, state: TokenState) -> torch.Tensor:
        positions = [max(g) if min(g) >= 0 else 0 for g in state.groups]
        return torch.tensor(positions, device=state.hidden.device, dtype=torch.long).unsqueeze(0)

    def _causal_mask(self, state: TokenState) -> torch.Tensor:
        n = state.n_tokens
        dtype = state.hidden.dtype
        mask = torch.full((n, n), torch.finfo(dtype).min, device=state.hidden.device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        if self.proportional_attention:
            key_bias = torch.log(state.mass.clamp_min(1.0)).to(dtype)
            mask = mask + key_bias.unsqueeze(0)
        return mask.unsqueeze(0).unsqueeze(0)

    def to_layer(self, batch: dict[str, torch.Tensor], layer_idx: int) -> TokenState:
        input_ids = batch["input_ids"]
        hidden = self.model.model.embed_tokens(input_ids)
        length = hidden.shape[1]
        position_ids = torch.arange(length, device=hidden.device).unsqueeze(0)
        state_mass = torch.ones(length, device=hidden.device, dtype=hidden.dtype)
        geometry = text_geometry(length, hidden.device, hidden.dtype)
        ids = input_ids[0]
        specials = torch.tensor([int(x) in set(self.tokenizer.all_special_ids) for x in ids.tolist()], device=hidden.device, dtype=torch.bool)
        groups = [[i] for i in range(length)]
        state = TokenState(hidden[0], state_mass, geometry, specials, groups, {"original_length": length, "input_ids": ids.detach(), "modality": "text", "layer_idx": layer_idx})
        mask = self._causal_mask(state)
        pos = self._position_ids(state)
        rope = self.model.model.rotary_emb(state.hidden.unsqueeze(0), pos)
        for layer in self.model.model.layers[:layer_idx]:
            out = layer(state.hidden.unsqueeze(0), attention_mask=mask, position_ids=pos, position_embeddings=rope, use_cache=False)
            state.hidden = (out[0] if isinstance(out, tuple) else out)[0]
        return state

    def advance(self, state: TokenState, start_layer: int, end_layer: int) -> TokenState:
        out_state = state.clone()
        hidden = out_state.hidden.unsqueeze(0)
        position_ids = self._position_ids(out_state)
        mask = self._causal_mask(out_state)
        rope = self.model.model.rotary_emb(hidden, position_ids)
        for layer in self.model.model.layers[start_layer:end_layer]:
            out = layer(hidden, attention_mask=mask, position_ids=position_ids, position_embeddings=rope, use_cache=False)
            hidden = out[0] if isinstance(out, tuple) else out
        out_state.hidden = hidden[0]
        out_state.aux["layer_idx"] = end_layer
        return out_state

    def from_layer(self, state: TokenState, layer_idx: int) -> torch.Tensor:
        final_state = self.advance(state, layer_idx, self.n_layers)
        hidden = self.model.model.norm(final_state.hidden.unsqueeze(0))
        logits = self.model.lm_head(hidden[:, -1])
        return logits[0]

    def from_layer_batch(self, states: list[TokenState], layer_idx: int) -> torch.Tensor:
        if not states:
            return torch.empty(0, self.model.config.vocab_size, device=self.device)
        lengths = {state.n_tokens for state in states}
        if len(lengths) != 1:
            return super().from_layer_batch(states, layer_idx)
        hidden = torch.stack([state.hidden for state in states])
        position_ids = torch.cat([self._position_ids(state) for state in states])
        attention_mask = torch.cat([self._causal_mask(state) for state in states])
        position_embeddings = self.model.model.rotary_emb(hidden, position_ids)
        for layer in self.model.model.layers[layer_idx:]:
            out = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            hidden = out[0] if isinstance(out, tuple) else out
        hidden = self.model.model.norm(hidden)
        return self.model.lm_head(hidden[:, -1])


    @torch.no_grad()
    def compressed_prefill(self, state: TokenState):
        if int(state.aux.get("layer_idx", 0)) != 0:
            raise ValueError("compressed prefill requires a layer-0 state")
        position_ids = self._position_ids(state)
        outputs = self.model.model(
            inputs_embeds=state.hidden.unsqueeze(0),
            attention_mask={"full_attention": self._causal_mask(state)},
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        logits = self.model.lm_head(outputs.last_hidden_state[:, -1])
        return logits[0], outputs.past_key_values

    @torch.no_grad()
    def generate_token_ids_from_layer0_state(self, state: TokenState, max_new_tokens: int = 64, temperature: float = 0.0) -> list[int]:
        logits, cache = self.compressed_prefill(state)
        generated = []
        original_length = int(state.aux["original_length"])
        for step in range(max_new_tokens):
            if temperature > 0:
                probs = torch.softmax(logits.float() / temperature, dim=-1)
                token = torch.multinomial(probs, 1)
            else:
                token = logits.argmax(dim=-1, keepdim=True)
            generated.append(int(token.item()))
            position_ids = torch.tensor([[original_length + step]], device=self.device, dtype=torch.long)
            cache_position = torch.tensor([state.n_tokens + step], device=self.device, dtype=torch.long)
            historical_mass = torch.cat([
                state.mass,
                torch.ones(step + 1, device=state.mass.device, dtype=state.mass.dtype),
            ])
            attention_mask = torch.log(historical_mass.clamp_min(1.0)).reshape(1, 1, 1, -1) if self.proportional_attention else torch.zeros(1, 1, 1, historical_mass.numel(), device=self.device, dtype=state.hidden.dtype)
            outputs = self.model(
                input_ids=token.reshape(1, 1),
                past_key_values=cache,
                attention_mask={"full_attention": attention_mask},
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            logits = outputs.logits[0, -1]
            if generated[-1] == self.tokenizer.eos_token_id:
                break
        return generated

    @torch.no_grad()
    def generate_from_layer0_state(self, state: TokenState, max_new_tokens: int = 64, temperature: float = 0.0):
        generated = self.generate_token_ids_from_layer0_state(state, max_new_tokens, temperature)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def candidates(self, state: TokenState) -> torch.Tensor:
        return text_adjacent_pairs(state)

    def damage(self, reference: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
        return symmetric_kl(reference.unsqueeze(0), counterfactual.unsqueeze(0))[0]

    def damage_batch(self, reference: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
        return symmetric_kl(reference.unsqueeze(0).expand_as(counterfactual), counterfactual)
