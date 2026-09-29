from __future__ import annotations

import torch

from .matching import greedy_maximal_matching
from .merge_ops import merge_matching
from .policy import PolicyConfig, PolicyContext, TransformerCostModel
from .types import MergeRoundResult, TokenState


def selected_pair_indices(pairs: torch.Tensor, selected: torch.Tensor, token_count: int) -> torch.Tensor:
    if selected.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=pairs.device)
    pair_keys = pairs[:, 0] * token_count + pairs[:, 1]
    selected_keys = selected[:, 0] * token_count + selected[:, 1]
    matches = selected_keys.unsqueeze(1) == pair_keys.unsqueeze(0)
    if not bool(matches.any(dim=1).all()):
        raise ValueError("selected pair is not present in the candidate list")
    return matches.to(torch.int8).argmax(dim=1)


def empty_round(state: TokenState, pairs: torch.Tensor, reason: str) -> MergeRoundResult:
    empty = torch.empty(0, device=state.hidden.device)
    empty_pairs = torch.empty((0, 2), dtype=torch.long, device=state.hidden.device)
    return MergeRoundResult(state.clone(), empty_pairs, empty, empty, empty, 0.0, reason)


@torch.no_grad()
def merge_round(
    state: TokenState,
    pairs: torch.Tensor,
    module,
    layer_idx: int,
    round_idx: int,
    initial_tokens: int,
    policy: PolicyConfig,
    cost_model: TransformerCostModel,
) -> MergeRoundResult:
    if state.n_tokens <= policy.target_tokens:
        return empty_round(state, pairs, "target_reached")
    if pairs.numel() == 0:
        return empty_round(state, pairs, "no_candidates")
    context = PolicyContext(layer_idx, round_idx, initial_tokens, state.n_tokens, policy.target_tokens, policy.distortion_budget, cost_model.n_layers)
    context_tensor = context.tensor(state.hidden.device, state.hidden.dtype)
    prediction = module.value_model(state, pairs, layer_idx, policy_context=context_tensor, epsilon=policy.distortion_budget)
    sigma = prediction.log_sigma.exp()
    risk = torch.exp(prediction.mu_log_damage + policy.confidence_z * sigma)
    saving_per_edge = cost_model.suffix_saving_fraction(state.n_tokens, 1, layer_idx, initial_tokens)
    compute_reward = policy.resolved_compute_reward(cost_model, initial_tokens)
    utility = compute_reward * saving_per_edge - risk
    allowed = utility > 0
    candidate_pairs = pairs[allowed]
    candidate_utility = utility[allowed]
    remaining_merges = state.n_tokens - policy.target_tokens
    maximum_merges = remaining_merges
    if policy.maximum_merges_per_round is not None:
        maximum_merges = min(maximum_merges, policy.maximum_merges_per_round)
    chosen = greedy_maximal_matching(candidate_pairs, candidate_utility, maximize=True, max_merges=maximum_merges)
    if chosen.shape[0] == 0:
        return empty_round(state, pairs, "no_positive_utility")
    selected_indices = selected_pair_indices(pairs, chosen, state.n_tokens)
    deltas = module.merge_operator(state, chosen, layer_idx, policy_context=context_tensor) if policy.use_merge_operator else None
    new_state = merge_matching(state, chosen, deltas)
    actual_saving = cost_model.suffix_saving_fraction(state.n_tokens, chosen.shape[0], layer_idx, initial_tokens)
    return MergeRoundResult(
        new_state,
        chosen,
        risk[selected_indices],
        prediction.safe_probability[selected_indices],
        utility[selected_indices],
        actual_saving,
        "merged",
    )
