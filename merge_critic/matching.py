from __future__ import annotations

import torch


def greedy_maximal_matching(pairs: torch.Tensor, scores: torch.Tensor, maximize: bool = True, max_merges: int | None = None) -> torch.Tensor:
    if pairs.numel() == 0:
        return pairs
    order = torch.argsort(scores, descending=maximize)
    used = set()
    chosen = []
    for idx in order.tolist():
        a, b = [int(x) for x in pairs[idx].tolist()]
        if a in used or b in used:
            continue
        chosen.append(idx)
        used.add(a)
        used.add(b)
        if max_merges is not None and len(chosen) >= max_merges:
            break
    if not chosen:
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device)
    return pairs[torch.tensor(chosen, device=pairs.device)]
