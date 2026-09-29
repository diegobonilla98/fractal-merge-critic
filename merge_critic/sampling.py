from __future__ import annotations

import torch


def stratified_pairs(hidden: torch.Tensor, pairs: torch.Tensor, count: int, seed: int = 0) -> torch.Tensor:
    if pairs.shape[0] <= count:
        return pairs
    a = hidden[pairs[:, 0]]
    b = hidden[pairs[:, 1]]
    sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    order = torch.argsort(sim)
    q = max(count // 4, 1)
    low = order[:q]
    high = order[-q:]
    mid_center = order.shape[0] // 2
    mid_start = min(max(0, mid_center - q // 2), order.shape[0] - q)
    mid = order[mid_start:mid_start + q]
    chosen_values = []
    chosen_set = set()
    for value in torch.cat([low, mid, high]).tolist():
        if value not in chosen_set and len(chosen_values) < count:
            chosen_values.append(value)
            chosen_set.add(value)
    remaining_values = [i for i in range(pairs.shape[0]) if i not in chosen_set]
    chosen = torch.tensor(chosen_values, dtype=torch.long, device=pairs.device)
    remaining = torch.tensor(remaining_values, dtype=torch.long, device=pairs.device)
    need = count - len(chosen_values)
    if need > 0 and remaining.numel() > 0:
        generator = torch.Generator(device=pairs.device)
        generator.manual_seed(seed)
        perm = remaining[torch.randperm(remaining.numel(), generator=generator, device=pairs.device)[:need]]
        chosen = torch.cat([chosen, perm])
    return pairs[chosen]


def active_pairs(mu: torch.Tensor, log_sigma: torch.Tensor, count: int, log_threshold: float) -> torch.Tensor:
    uncertainty = log_sigma.exp()
    boundary = (mu - log_threshold).abs()
    score = uncertainty / (boundary + 1e-4)
    return torch.argsort(score, descending=True)[:count]
