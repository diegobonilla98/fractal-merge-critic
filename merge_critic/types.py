from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TokenState:
    hidden: torch.Tensor
    mass: torch.Tensor
    geometry: torch.Tensor
    special_mask: torch.Tensor
    groups: list[list[int]]
    aux: dict[str, Any]

    @property
    def n_tokens(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def dim(self) -> int:
        return int(self.hidden.shape[-1])

    def clone(self) -> "TokenState":
        return TokenState(
            hidden=self.hidden.clone(),
            mass=self.mass.clone(),
            geometry=self.geometry.clone(),
            special_mask=self.special_mask.clone(),
            groups=[list(g) for g in self.groups],
            aux=dict(self.aux),
        )


@dataclass
class MergePrediction:
    mu_log_damage: torch.Tensor
    log_sigma: torch.Tensor
    safe_probability: torch.Tensor | None = None


@dataclass
class MergeRoundResult:
    state: TokenState
    selected_pairs: torch.Tensor
    predicted_damage: torch.Tensor
    safe_probability: torch.Tensor
    utility: torch.Tensor
    compute_saving_fraction: float
    stop_reason: str
