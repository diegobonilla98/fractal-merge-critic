from __future__ import annotations

import torch

from .types import TokenState


def vision_adjacent_pairs(state: TokenState) -> torch.Tensor:
    pairs = []
    groups = [set(g) for g in state.groups]
    original_hw = state.aux.get("original_grid_hw")
    prefix = int(state.aux.get("prefix_tokens", 0))
    if original_hw is None:
        raise ValueError("vision state requires original_grid_hw")
    h, w = original_hw
    pixel_to_node = {}
    for node_idx, group in enumerate(groups):
        if state.special_mask[node_idx]:
            continue
        for p in group:
            pixel_to_node[p] = node_idx
    edge_set = set()
    for p, node in pixel_to_node.items():
        y, x = divmod(p, w)
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w:
                q = yy * w + xx
                other = pixel_to_node.get(q)
                if other is not None and other != node:
                    edge_set.add(tuple(sorted((node, other))))
    pairs.extend(sorted(edge_set))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=state.hidden.device)
    return torch.tensor(pairs, dtype=torch.long, device=state.hidden.device)


def text_adjacent_pairs(state: TokenState) -> torch.Tensor:
    pairs = []
    for i in range(state.n_tokens - 1):
        if state.special_mask[i] or state.special_mask[i + 1]:
            continue
        ga = state.groups[i]
        gb = state.groups[i + 1]
        if max(ga) + 1 == min(gb):
            pairs.append((i, i + 1))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=state.hidden.device)
    return torch.tensor(pairs, dtype=torch.long, device=state.hidden.device)


def all_nonspecial_pairs(state: TokenState) -> torch.Tensor:
    valid = torch.nonzero(~state.special_mask, as_tuple=False).flatten()
    if valid.numel() < 2:
        return torch.empty((0, 2), dtype=torch.long, device=state.hidden.device)
    return torch.combinations(valid, r=2)
