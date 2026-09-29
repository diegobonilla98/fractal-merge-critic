from __future__ import annotations

from dataclasses import dataclass

import torch

from .policy import PolicyConfig, TransformerCostModel
from .runtime import merge_round


@dataclass
class ScheduleStats:
    initial_tokens: int
    final_tokens: int
    merges_per_site: dict[int, int]
    rounds: int
    estimated_compute_saving_fraction: float
    trace: list[dict[str, float | int | str]]


@torch.no_grad()
def run_merge_schedule(
    adapter,
    batch,
    module,
    policy: PolicyConfig,
    merge_sites: list[int] | None = None,
    cost_model: TransformerCostModel | None = None,
):
    sites = list(range(adapter.n_layers)) if merge_sites is None else sorted(set(int(x) for x in merge_sites))
    if any(x < 0 or x >= adapter.n_layers for x in sites):
        raise ValueError("merge site outside transformer depth")
    if cost_model is None:
        cost_model = TransformerCostModel.from_adapter(adapter)
    state = adapter.to_layer(batch, 0)
    initial_tokens = state.n_tokens
    if policy.target_tokens >= initial_tokens:
        output = adapter.from_layer(state, 0)
        return output, ScheduleStats(initial_tokens, initial_tokens, {}, 0, 0.0, [])
    current_layer = 0
    global_round = 0
    merges_per_site = {}
    trace = []
    estimated_saving = 0.0
    for site in sites:
        if state.n_tokens <= policy.target_tokens:
            break
        if site > current_layer:
            state = adapter.advance(state, current_layer, site)
            current_layer = site
        count = 0
        for site_round in range(policy.maximum_rounds_per_site):
            pairs = adapter.candidates(state)
            result = merge_round(state, pairs, module, site, global_round, initial_tokens, policy, cost_model)
            trace.append({
                "layer": site,
                "site_round": site_round,
                "global_round": global_round,
                "tokens_before": state.n_tokens,
                "selected_merges": int(result.selected_pairs.shape[0]),
                "compute_saving_fraction": result.compute_saving_fraction,
                "stop_reason": result.stop_reason,
            })
            if result.stop_reason != "merged":
                break
            count += int(result.selected_pairs.shape[0])
            estimated_saving += result.compute_saving_fraction
            state = result.state
            global_round += 1
        merges_per_site[site] = count
    output = adapter.from_layer(state, current_layer)
    stats = ScheduleStats(initial_tokens, state.n_tokens, merges_per_site, global_round, estimated_saving, trace)
    return output, stats
