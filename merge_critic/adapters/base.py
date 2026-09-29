from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..types import TokenState


class FrozenBackboneAdapter(ABC):
    def __init__(self):
        self._freeze()

    def _freeze(self):
        if hasattr(self, "model"):
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_layers(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, item: Any) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def to_layer(self, batch: dict[str, torch.Tensor], layer_idx: int) -> TokenState:
        raise NotImplementedError

    @abstractmethod
    def advance(self, state: TokenState, start_layer: int, end_layer: int) -> TokenState:
        raise NotImplementedError

    @abstractmethod
    def from_layer(self, state: TokenState, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def from_layer_batch(self, states: list[TokenState], layer_idx: int) -> torch.Tensor:
        return torch.stack([self.from_layer(state, layer_idx) for state in states])

    @abstractmethod
    def candidates(self, state: TokenState) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def damage(self, reference: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def damage_batch(self, reference: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.damage(reference, value) for value in counterfactual])

    def reference(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = self.to_layer(batch, 0)
        return self.from_layer(state, 0)
