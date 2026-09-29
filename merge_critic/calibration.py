from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .baselines import cosine_scores
from .matching import greedy_maximal_matching
from .merge_ops import counterfactual_batch, merge_matching
from .policy import PolicyContext, TransformerCostModel
from .runtime import merge_round
from .sampling import stratified_pairs
from .types import TokenState


CALIBRATION_SCHEMA_VERSION = 2


@dataclass
class CalibrationRecord:
    a: torch.Tensor
    b: torch.Tensor
    ga: torch.Tensor
    gb: torch.Tensor
    ma: torch.Tensor
    mb: torch.Tensor
    total_mass: torch.Tensor
    context: torch.Tensor
    policy_context: torch.Tensor
    layer_idx: int
    round_idx: int
    initial_tokens: int
    current_tokens: int
    target_tokens: int
    distortion_budget: float
    compute_saving: torch.Tensor
    damage_raw: torch.Tensor
    damage_terminal: torch.Tensor
    damage_oracle: torch.Tensor
    oracle_delta: torch.Tensor
    rollout_rounds: int
    sample_id: int = -1
    schema_version: int = CALIBRATION_SCHEMA_VERSION

    def cpu_dict(self):
        output = asdict(self)
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                output[key] = value.detach().cpu()
        return output


def state_context(state: TokenState) -> torch.Tensor:
    nonspecial = state.hidden[~state.special_mask]
    if nonspecial.numel() == 0:
        return state.hidden.mean(dim=0)
    return nonspecial.mean(dim=0)


def optimize_oracle_delta(adapter, state: TokenState, pair: torch.Tensor, layer_idx: int, reference: torch.Tensor, steps: int = 8, lr: float = 0.05, max_ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    deltas, damages = optimize_oracle_deltas(adapter, state, pair.reshape(1, 2), layer_idx, reference, steps, lr, max_ratio)
    return deltas[0], damages[0]


def optimize_oracle_deltas(adapter, state: TokenState, pairs: torch.Tensor, layer_idx: int, reference: torch.Tensor, steps: int = 8, lr: float = 0.05, max_ratio: float = 0.1, initial_damage: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    i = pairs[:, 0]
    j = pairs[:, 1]
    ma = state.mass[i].unsqueeze(-1)
    mb = state.mass[j].unsqueeze(-1)
    base = ((ma * state.hidden[i] + mb * state.hidden[j]) / (ma + mb).clamp_min(1e-8)).float()
    delta = torch.zeros_like(base, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    if initial_damage is None:
        with torch.no_grad():
            initial_states = counterfactual_batch(state, pairs)
            initial_damage = adapter.damage_batch(reference, adapter.from_layer_batch(initial_states, layer_idx))
    best_damage = initial_damage.detach().float().clone()
    best_delta = torch.zeros_like(delta)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        max_norm = max_ratio * base.detach().norm(dim=-1, keepdim=True).clamp_min(1e-8)
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        projected = delta * torch.minimum(torch.ones_like(norm), max_norm / norm)
        merged = counterfactual_batch(state, pairs, projected.to(state.hidden.dtype))
        output = adapter.from_layer_batch(merged, layer_idx)
        loss = adapter.damage_batch(reference.detach(), output)
        with torch.no_grad():
            improved = loss.float() < best_damage
            best_damage = torch.where(improved, loss.float(), best_damage)
            best_delta = torch.where(improved.unsqueeze(-1), projected.detach(), best_delta)
        loss.sum().backward()
        optimizer.step()
    with torch.no_grad():
        max_norm = max_ratio * base.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        projected = delta * torch.minimum(torch.ones_like(norm), max_norm / norm)
        merged = counterfactual_batch(state, pairs, projected.to(state.hidden.dtype))
        damage = adapter.damage_batch(reference, adapter.from_layer_batch(merged, layer_idx)).float()
        improved = damage < best_damage
        best_damage = torch.where(improved, damage, best_damage)
        best_delta = torch.where(improved.unsqueeze(-1), projected, best_delta)
    return best_delta.detach(), best_damage.detach()


def cosine_rollout_state(adapter, state: TokenState, rounds: int, merges_per_round: int) -> TokenState:
    output = state
    for _ in range(rounds):
        pairs = adapter.candidates(output)
        if pairs.numel() == 0:
            break
        chosen = greedy_maximal_matching(pairs, cosine_scores(output, pairs), maximize=True, max_merges=merges_per_round)
        if chosen.numel() == 0:
            break
        output = merge_matching(output, chosen)
    return output


def policy_rollout_state(adapter, state: TokenState, module, layer_idx: int, starting_round: int, initial_tokens: int, policy, cost_model: TransformerCostModel, rounds: int) -> TokenState:
    output = state
    for offset in range(rounds):
        result = merge_round(output, adapter.candidates(output), module, layer_idx, starting_round + offset, initial_tokens, policy, cost_model)
        if result.stop_reason != "merged":
            break
        output = result.state
    return output


def terminal_damage_batch(
    adapter,
    states: list[TokenState],
    layer_idx: int,
    reference: torch.Tensor,
    rollout_rounds: int,
    rollout_merges_per_round: int,
    rollout_module=None,
    rollout_policy=None,
    starting_round: int = 0,
    initial_tokens: int | None = None,
    cost_model: TransformerCostModel | None = None,
) -> torch.Tensor:
    if rollout_module is None:
        terminal_states = [cosine_rollout_state(adapter, state, rollout_rounds, rollout_merges_per_round) for state in states]
    else:
        if rollout_policy is None or cost_model is None or initial_tokens is None:
            raise ValueError("learned-policy rollout requires policy, cost model, and initial token count")
        terminal_states = [policy_rollout_state(adapter, state, rollout_module, layer_idx, starting_round, initial_tokens, rollout_policy, cost_model, rollout_rounds) for state in states]
    lengths = {state.n_tokens for state in terminal_states}
    if len(lengths) == 1:
        outputs = adapter.from_layer_batch(terminal_states, layer_idx)
    else:
        outputs = torch.stack([adapter.from_layer(state, layer_idx) for state in terminal_states])
    return adapter.damage_batch(reference, outputs)


def collect_records_from_state(
    adapter,
    state: TokenState,
    layer_idx: int,
    reference: torch.Tensor,
    pairs: torch.Tensor,
    oracle_steps: int = 8,
    oracle_lr: float = 0.05,
    max_refine_ratio: float = 0.1,
    sample_id: int = -1,
    round_idx: int = 0,
    initial_tokens: int | None = None,
    target_tokens: int | None = None,
    distortion_budget: float = 1e-3,
    rollout_rounds: int = 2,
    rollout_merges_per_round: int = 1,
    cost_model: TransformerCostModel | None = None,
    rollout_module=None,
    rollout_policy=None,
) -> list[CalibrationRecord]:
    initial_tokens = state.n_tokens if initial_tokens is None else initial_tokens
    target_tokens = max(1, initial_tokens // 2) if target_tokens is None else target_tokens
    cost_model = TransformerCostModel.from_adapter(adapter) if cost_model is None else cost_model
    context = state_context(state)
    policy_context = PolicyContext(layer_idx, round_idx, initial_tokens, state.n_tokens, target_tokens, distortion_budget, cost_model.n_layers).tensor(state.hidden.device, state.hidden.dtype)
    with torch.no_grad():
        raw_states = counterfactual_batch(state, pairs)
        raw_outputs = adapter.from_layer_batch(raw_states, layer_idx)
        damage_raw = adapter.damage_batch(reference, raw_outputs)
        damage_terminal = terminal_damage_batch(adapter, raw_states, layer_idx, reference, rollout_rounds, rollout_merges_per_round, rollout_module, rollout_policy, round_idx + 1, initial_tokens, cost_model)
    oracle_delta, damage_oracle = optimize_oracle_deltas(adapter, state, pairs, layer_idx, reference, oracle_steps, oracle_lr, max_refine_ratio, damage_raw)
    compute_saving = cost_model.suffix_saving_fraction(state.n_tokens, 1, layer_idx, initial_tokens)
    records = []
    for pair_index, pair in enumerate(pairs):
        i, j = [int(value) for value in pair.tolist()]
        records.append(CalibrationRecord(
            a=state.hidden[i].detach(),
            b=state.hidden[j].detach(),
            ga=state.geometry[i].detach(),
            gb=state.geometry[j].detach(),
            ma=state.mass[i].detach(),
            mb=state.mass[j].detach(),
            total_mass=state.mass.sum().detach(),
            context=context.detach(),
            policy_context=policy_context.detach(),
            layer_idx=layer_idx,
            round_idx=round_idx,
            initial_tokens=initial_tokens,
            current_tokens=state.n_tokens,
            target_tokens=target_tokens,
            distortion_budget=distortion_budget,
            compute_saving=torch.tensor(compute_saving),
            damage_raw=damage_raw[pair_index].detach(),
            damage_terminal=damage_terminal[pair_index].detach(),
            damage_oracle=damage_oracle[pair_index].detach(),
            oracle_delta=oracle_delta[pair_index].detach(),
            rollout_rounds=rollout_rounds,
            sample_id=sample_id,
        ))
    return records


def collect_records(
    adapter,
    batch: dict[str, torch.Tensor],
    layer_idx: int,
    pair_count: int = 24,
    oracle_steps: int = 8,
    oracle_lr: float = 0.05,
    max_refine_ratio: float = 0.1,
    seed: int = 0,
    sample_id: int = -1,
    state_rounds: int = 3,
    rollout_rounds: int = 2,
    rollout_merges_per_round: int = 1,
    target_fraction: float = 0.5,
    distortion_budget: float = 1e-3,
) -> list[CalibrationRecord]:
    with torch.no_grad():
        reference = adapter.reference(batch)
        state = adapter.to_layer(batch, layer_idx)
    initial_tokens = state.n_tokens
    target_tokens = max(1, round(initial_tokens * target_fraction))
    cost_model = TransformerCostModel.from_adapter(adapter)
    records = []
    for round_idx in range(state_rounds):
        pairs = adapter.candidates(state)
        pairs = stratified_pairs(state.hidden, pairs, pair_count, seed + round_idx)
        if pairs.numel() == 0:
            break
        records.extend(collect_records_from_state(
            adapter,
            state,
            layer_idx,
            reference,
            pairs,
            oracle_steps,
            oracle_lr,
            max_refine_ratio,
            sample_id,
            round_idx,
            initial_tokens,
            target_tokens,
            distortion_budget,
            rollout_rounds,
            rollout_merges_per_round,
            cost_model,
        ))
        all_pairs = adapter.candidates(state)
        chosen = greedy_maximal_matching(all_pairs, cosine_scores(state, all_pairs), maximize=True, max_merges=rollout_merges_per_round)
        if chosen.numel() == 0:
            break
        state = merge_matching(state, chosen)
    return records


def save_record_dicts(values: list[dict], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(values, temporary)
    os.replace(temporary, target)


def save_records(records: list[CalibrationRecord], path: str) -> None:
    save_record_dicts([record.cpu_dict() for record in records], path)


def append_records(records: list[CalibrationRecord], path: str) -> None:
    existing = torch.load(path, weights_only=False) if os.path.exists(path) else []
    existing.extend(record.cpu_dict() for record in records)
    save_record_dicts(existing, path)
