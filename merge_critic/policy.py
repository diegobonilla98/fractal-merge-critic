from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .model import POLICY_CONTEXT_DIM


@dataclass(frozen=True)
class PolicyContext:
    layer_idx: int
    round_idx: int
    initial_tokens: int
    current_tokens: int
    target_tokens: int
    distortion_budget: float
    n_layers: int

    def tensor(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        initial = max(self.initial_tokens, 1)
        maximum_merges = max(self.initial_tokens - self.target_tokens, 1)
        values = [
            self.layer_idx / max(self.n_layers - 1, 1),
            (self.n_layers - self.layer_idx) / max(self.n_layers, 1),
            self.round_idx / maximum_merges,
            self.current_tokens / initial,
            self.target_tokens / initial,
            max(self.current_tokens - self.target_tokens, 0) / initial,
            max(min(math.log10(max(self.distortion_budget, 1e-12)) / 12.0, 0.0), -1.0),
        ]
        output = torch.tensor(values, device=device, dtype=dtype)
        if output.shape != (POLICY_CONTEXT_DIM,):
            raise RuntimeError("invalid policy context")
        return output


@dataclass(frozen=True)
class TransformerCostModel:
    hidden_size: int
    intermediate_size: int
    n_layers: int

    def layer_cost(self, tokens: int) -> float:
        n = float(tokens)
        d = float(self.hidden_size)
        ff = float(self.intermediate_size)
        return 4.0 * n * d * d + 2.0 * n * n * d + 2.0 * n * d * ff

    def full_cost(self, initial_tokens: int) -> float:
        return self.n_layers * self.layer_cost(initial_tokens)

    def suffix_saving_fraction(self, current_tokens: int, removed_tokens: int, layer_idx: int, initial_tokens: int) -> float:
        remaining_layers = max(self.n_layers - layer_idx, 0)
        next_tokens = max(current_tokens - removed_tokens, 1)
        saved = remaining_layers * (self.layer_cost(current_tokens) - self.layer_cost(next_tokens))
        return max(saved / max(self.full_cost(initial_tokens), 1.0), 0.0)

    def target_saving_fraction(self, initial_tokens: int, target_tokens: int) -> float:
        saved = self.full_cost(initial_tokens) - self.n_layers * self.layer_cost(target_tokens)
        return max(saved / max(self.full_cost(initial_tokens), 1.0), 0.0)

    @classmethod
    def from_adapter(cls, adapter) -> "TransformerCostModel":
        config = adapter.model.config
        vision_config = getattr(config, "vision_config", config)
        hidden_size = int(getattr(vision_config, "hidden_size"))
        intermediate_size = int(getattr(vision_config, "intermediate_size", hidden_size * 4))
        return cls(hidden_size, intermediate_size, int(adapter.n_layers))


@dataclass(frozen=True)
class PolicyConfig:
    target_tokens: int
    distortion_budget: float
    confidence_z: float = 1.645
    compute_reward: float | None = None
    maximum_rounds_per_site: int = 4
    maximum_merges_per_round: int | None = None
    use_merge_operator: bool = True

    def resolved_compute_reward(self, cost_model: TransformerCostModel, initial_tokens: int) -> float:
        if self.compute_reward is not None:
            return self.compute_reward
        target_saving = cost_model.target_saving_fraction(initial_tokens, self.target_tokens)
        return self.distortion_budget / max(target_saving, 1e-6)
