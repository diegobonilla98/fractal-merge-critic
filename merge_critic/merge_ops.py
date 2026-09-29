from __future__ import annotations

import torch

from .geometry import merge_geometry, merge_text_geometry
from .types import TokenState


def weighted_mean_token(a: torch.Tensor, b: torch.Tensor, ma: torch.Tensor, mb: torch.Tensor) -> torch.Tensor:
    return (ma * a + mb * b) / (ma + mb).clamp_min(1e-8)


def merge_pair(state: TokenState, i: int, j: int, delta: torch.Tensor | None = None) -> TokenState:
    if i == j:
        raise ValueError("cannot merge token with itself")
    if i > j:
        i, j = j, i
    if bool(state.special_mask[i]) or bool(state.special_mask[j]):
        raise ValueError("special tokens cannot be merged")
    ma = state.mass[i]
    mb = state.mass[j]
    merged_hidden = weighted_mean_token(state.hidden[i], state.hidden[j], ma, mb)
    if delta is not None:
        merged_hidden = merged_hidden + delta
    merged_mass = ma + mb
    if state.aux.get("modality") == "text":
        merged_geometry = merge_text_geometry(state.geometry[i], state.geometry[j])
    else:
        merged_geometry = merge_geometry(state.geometry[i], state.geometry[j], ma, mb)
    keep = [k for k in range(state.n_tokens) if k not in (i, j)]
    insertion = min(i, j)
    hidden_items = [state.hidden[k] for k in keep]
    mass_items = [state.mass[k] for k in keep]
    geom_items = [state.geometry[k] for k in keep]
    special_items = [state.special_mask[k] for k in keep]
    group_items = [state.groups[k] for k in keep]
    insert_at = sum(1 for k in keep if k < insertion)
    hidden_items.insert(insert_at, merged_hidden)
    mass_items.insert(insert_at, merged_mass)
    geom_items.insert(insert_at, merged_geometry)
    special_items.insert(insert_at, torch.tensor(False, device=state.hidden.device))
    group_items.insert(insert_at, sorted(state.groups[i] + state.groups[j]))
    return TokenState(
        hidden=torch.stack(hidden_items),
        mass=torch.stack(mass_items),
        geometry=torch.stack(geom_items),
        special_mask=torch.stack(special_items).bool(),
        groups=group_items,
        aux=dict(state.aux),
    )


def merge_matching(state: TokenState, pairs: torch.Tensor, deltas: torch.Tensor | None = None) -> TokenState:
    if pairs.numel() == 0:
        return state.clone()
    pair_map = {}
    for pidx, (a, b) in enumerate(pairs.tolist()):
        if a in pair_map or b in pair_map:
            raise ValueError("pairs must form a matching")
        pair_map[a] = (a, b, pidx)
        pair_map[b] = (a, b, pidx)
    hidden_items = []
    mass_items = []
    geom_items = []
    special_items = []
    group_items = []
    visited = set()
    for idx in range(state.n_tokens):
        if idx in visited:
            continue
        if idx not in pair_map:
            hidden_items.append(state.hidden[idx])
            mass_items.append(state.mass[idx])
            geom_items.append(state.geometry[idx])
            special_items.append(state.special_mask[idx])
            group_items.append(list(state.groups[idx]))
            continue
        a, b, pidx = pair_map[idx]
        if a in visited or b in visited:
            continue
        ma = state.mass[a]
        mb = state.mass[b]
        hidden = weighted_mean_token(state.hidden[a], state.hidden[b], ma, mb)
        if deltas is not None:
            hidden = hidden + deltas[pidx]
        if state.aux.get("modality") == "text":
            geom = merge_text_geometry(state.geometry[a], state.geometry[b])
        else:
            geom = merge_geometry(state.geometry[a], state.geometry[b], ma, mb)
        hidden_items.append(hidden)
        mass_items.append(ma + mb)
        geom_items.append(geom)
        special_items.append(torch.tensor(False, device=state.hidden.device))
        group_items.append(sorted(state.groups[a] + state.groups[b]))
        visited.add(a)
        visited.add(b)
    return TokenState(
        hidden=torch.stack(hidden_items),
        mass=torch.stack(mass_items),
        geometry=torch.stack(geom_items),
        special_mask=torch.stack(special_items).bool(),
        groups=group_items,
        aux=dict(state.aux),
    )


def counterfactual_batch(state: TokenState, pairs: torch.Tensor, deltas: torch.Tensor | None = None) -> list[TokenState]:
    outputs = []
    for idx, pair in enumerate(pairs.tolist()):
        delta = None if deltas is None else deltas[idx]
        outputs.append(merge_pair(state, int(pair[0]), int(pair[1]), delta))
    return outputs
