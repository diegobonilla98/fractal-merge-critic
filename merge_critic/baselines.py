from __future__ import annotations

import torch

from .matching import greedy_maximal_matching
from .merge_ops import counterfactual_batch, merge_matching


def cosine_scores(state, pairs: torch.Tensor) -> torch.Tensor:
    a = state.hidden[pairs[:, 0]]
    b = state.hidden[pairs[:, 1]]
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def cosine_merge_round(state, pairs: torch.Tensor, max_merges: int | None = None):
    if pairs.numel() == 0:
        return state.clone(), pairs
    scores = cosine_scores(state, pairs)
    chosen = greedy_maximal_matching(pairs, scores, maximize=True, max_merges=max_merges)
    return merge_matching(state, chosen), chosen


def random_merge_round(state, pairs: torch.Tensor, max_merges: int | None = None, seed: int = 0):
    if pairs.numel() == 0:
        return state.clone(), pairs
    generator = torch.Generator(device=pairs.device)
    generator.manual_seed(seed)
    scores = torch.rand(pairs.shape[0], generator=generator, device=pairs.device)
    chosen = greedy_maximal_matching(pairs, scores, maximize=True, max_merges=max_merges)
    return merge_matching(state, chosen), chosen


@torch.no_grad()
def exact_oracle_scores(adapter, state, layer_idx: int, reference: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    if pairs.numel() == 0:
        return torch.empty(0, device=state.hidden.device)
    states = counterfactual_batch(state, pairs)
    return adapter.damage_batch(reference, adapter.from_layer_batch(states, layer_idx))


@torch.no_grad()
def oracle_merge_round(adapter, state, layer_idx: int, reference: torch.Tensor, pairs: torch.Tensor, max_merges: int | None = None):
    damages = exact_oracle_scores(adapter, state, layer_idx, reference, pairs)
    chosen = greedy_maximal_matching(pairs, -damages, maximize=True, max_merges=max_merges)
    return merge_matching(state, chosen), chosen, damages


@torch.no_grad()
def replay_cosine_trace(adapter, batch, trace: list[dict[str, float | int | str]]):
    state = adapter.to_layer(batch, 0)
    current_layer = 0
    replayed = []
    for step in trace:
        layer = int(step["layer"])
        merge_count = int(step["selected_merges"])
        if layer > current_layer:
            state = adapter.advance(state, current_layer, layer)
            current_layer = layer
        if merge_count <= 0:
            continue
        pairs = adapter.candidates(state)
        state, chosen = cosine_merge_round(state, pairs, merge_count)
        replayed.append({"layer": layer, "selected_merges": int(chosen.shape[0])})
    return adapter.from_layer(state, current_layer), state.n_tokens, replayed
