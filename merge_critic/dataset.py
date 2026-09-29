from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .types import TokenState


class PairCalibrationDataset(Dataset):
    def __init__(self, path: str):
        self.records = torch.load(path, weights_only=False)
        required = {"policy_context", "round_idx", "damage_terminal", "schema_version"}
        if not self.records or any(not required.issubset(record) for record in self.records):
            raise ValueError("calibration corpus is legacy or incomplete; regenerate it with policy rollout calibration")
        if any(int(record["schema_version"]) != 2 for record in self.records):
            raise ValueError("unsupported calibration schema")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def record_to_state(batch: dict[str, torch.Tensor], item: int, device: torch.device) -> tuple[TokenState, torch.Tensor, torch.Tensor]:
    a = batch["a"][item].to(device)
    b = batch["b"][item].to(device)
    ga = batch["ga"][item].to(device)
    gb = batch["gb"][item].to(device)
    ma = batch["ma"][item].to(device)
    mb = batch["mb"][item].to(device)
    context = batch["context"][item].to(device)
    state = TokenState(
        hidden=torch.stack([a, b]),
        mass=torch.stack([ma, mb]),
        geometry=torch.stack([ga, gb]),
        special_mask=torch.zeros(2, dtype=torch.bool, device=device),
        groups=[[0], [1]],
        aux={},
    )
    pair = torch.tensor([[0, 1]], dtype=torch.long, device=device)
    return state, pair, context
