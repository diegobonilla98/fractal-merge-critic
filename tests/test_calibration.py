import torch
from types import SimpleNamespace

from merge_critic.calibration import collect_records_from_state, optimize_oracle_delta, optimize_oracle_deltas
from merge_critic.candidates import text_adjacent_pairs
from merge_critic.geometry import text_geometry
from merge_critic.merge_ops import merge_pair
from merge_critic.model import MergeModule
from merge_critic.policy import PolicyConfig, TransformerCostModel
from merge_critic.types import TokenState


class FakeAdapter:
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=12, intermediate_size=48))
    n_layers = 2

    def candidates(self, state):
        return text_adjacent_pairs(state)

    def from_layer(self, state, layer_idx):
        return state.hidden.sum(dim=0)

    def from_layer_batch(self, states, layer_idx):
        return torch.stack([self.from_layer(state, layer_idx) for state in states])

    def damage(self, reference, counterfactual):
        return (reference - counterfactual).pow(2).mean()

    def damage_batch(self, reference, counterfactual):
        return torch.stack([self.damage(reference, value) for value in counterfactual])


def test_oracle_delta_reduces_counterfactual_damage():
    torch.manual_seed(4)
    hidden = torch.randn(4, 12)
    state = TokenState(
        hidden=hidden,
        mass=torch.ones(4),
        geometry=text_geometry(4),
        special_mask=torch.zeros(4, dtype=torch.bool),
        groups=[[0], [1], [2], [3]],
        aux={"modality": "text"},
    )
    adapter = FakeAdapter()
    reference = hidden.sum(dim=0)
    pair = torch.tensor([1, 2])
    raw = merge_pair(state, 1, 2)
    raw_damage = adapter.damage(reference, adapter.from_layer(raw, 0))
    _, refined_damage = optimize_oracle_delta(adapter, state, pair, 0, reference, steps=50, lr=0.1, max_ratio=10.0)
    assert refined_damage < raw_damage


def test_batched_oracle_matches_individual_optimization():
    torch.manual_seed(7)
    hidden = torch.randn(6, 12)
    state = TokenState(
        hidden=hidden,
        mass=torch.ones(6),
        geometry=text_geometry(6),
        special_mask=torch.zeros(6, dtype=torch.bool),
        groups=[[0], [1], [2], [3], [4], [5]],
        aux={"modality": "text"},
    )
    adapter = FakeAdapter()
    reference = hidden.sum(dim=0)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    batched_delta, batched_damage = optimize_oracle_deltas(adapter, state, pairs, 0, reference, steps=12, lr=0.1, max_ratio=1.0)
    individual = [optimize_oracle_delta(adapter, state, pair, 0, reference, steps=12, lr=0.1, max_ratio=1.0) for pair in pairs]
    individual_delta = torch.stack([value[0] for value in individual])
    individual_damage = torch.stack([value[1] for value in individual])
    assert torch.allclose(batched_delta, individual_delta, atol=1e-6)
    assert torch.allclose(batched_damage, individual_damage, atol=1e-6)
    raw_damage = torch.stack([
        adapter.damage(reference, adapter.from_layer(merge_pair(state, int(pair[0]), int(pair[1])), 0))
        for pair in pairs
    ])
    assert torch.all(batched_damage <= raw_damage)


def test_rollout_record_contains_sequential_policy_targets():
    torch.manual_seed(9)
    hidden = torch.randn(6, 12)
    state = TokenState(
        hidden=hidden,
        mass=torch.ones(6),
        geometry=text_geometry(6),
        special_mask=torch.zeros(6, dtype=torch.bool),
        groups=[[index] for index in range(6)],
        aux={"modality": "text"},
    )
    adapter = FakeAdapter()
    records = collect_records_from_state(
        adapter,
        state,
        0,
        hidden.sum(dim=0),
        text_adjacent_pairs(state)[:2],
        oracle_steps=2,
        round_idx=1,
        initial_tokens=8,
        target_tokens=4,
        rollout_rounds=2,
    )
    assert len(records) == 2
    assert records[0].policy_context.shape == (7,)
    assert records[0].round_idx == 1
    assert records[0].damage_terminal.ndim == 0
    assert records[0].compute_saving > 0
    assert records[0].schema_version == 2


def test_terminal_target_can_follow_learned_policy_rollout():
    torch.manual_seed(12)
    hidden = torch.randn(6, 12)
    state = TokenState(
        hidden=hidden,
        mass=torch.ones(6),
        geometry=text_geometry(6),
        special_mask=torch.zeros(6, dtype=torch.bool),
        groups=[[index] for index in range(6)],
        aux={"modality": "text"},
    )
    adapter = FakeAdapter()
    module = MergeModule(12, 2, critic_dim=8)
    cost_model = TransformerCostModel(12, 48, 2)
    policy = PolicyConfig(3, 1.0, confidence_z=0.0, compute_reward=10.0, maximum_merges_per_round=1)
    records = collect_records_from_state(
        adapter,
        state,
        0,
        hidden.sum(dim=0),
        text_adjacent_pairs(state)[:2],
        oracle_steps=1,
        initial_tokens=6,
        target_tokens=3,
        rollout_rounds=2,
        cost_model=cost_model,
        rollout_module=module,
        rollout_policy=policy,
    )
    assert len(records) == 2
    assert all(torch.isfinite(record.damage_terminal) for record in records)
