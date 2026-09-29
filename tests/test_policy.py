import torch

from merge_critic.candidates import text_adjacent_pairs
from merge_critic.baselines import replay_cosine_trace
from merge_critic.geometry import text_geometry
from merge_critic.model import MergeModule
from merge_critic.policy import PolicyConfig, PolicyContext, TransformerCostModel
from merge_critic.runtime import merge_round
from merge_critic.types import TokenState


def make_state(length=8, dim=16):
    return TokenState(
        hidden=torch.randn(length, dim),
        mass=torch.ones(length),
        geometry=text_geometry(length),
        special_mask=torch.zeros(length, dtype=torch.bool),
        groups=[[index] for index in range(length)],
        aux={"modality": "text", "original_length": length},
    )


def test_policy_context_changes_with_layer_round_and_budget():
    first = PolicyContext(0, 0, 16, 16, 8, 1e-3, 8).tensor("cpu")
    second = PolicyContext(4, 2, 16, 12, 6, 1e-4, 8).tensor("cpu")
    assert first.shape == (7,)
    assert not torch.equal(first, second)


def test_earlier_merge_saves_more_suffix_compute():
    cost = TransformerCostModel(64, 256, 12)
    early = cost.suffix_saving_fraction(64, 1, 0, 64)
    late = cost.suffix_saving_fraction(64, 1, 10, 64)
    assert early > late > 0
    assert 0 < cost.target_saving_fraction(64, 32) < 1


def test_policy_merges_positive_utility_and_stops_negative_utility():
    torch.manual_seed(3)
    state = make_state()
    pairs = text_adjacent_pairs(state)
    module = MergeModule(16, 4, critic_dim=8)
    cost = TransformerCostModel(16, 64, 4)
    merge_policy = PolicyConfig(target_tokens=4, distortion_budget=1.0, confidence_z=0.0, compute_reward=10.0, maximum_merges_per_round=1)
    merged = merge_round(state, pairs, module, 0, 0, state.n_tokens, merge_policy, cost)
    assert merged.stop_reason == "merged"
    assert merged.state.n_tokens == state.n_tokens - 1
    stop_policy = PolicyConfig(target_tokens=4, distortion_budget=1.0, confidence_z=0.0, compute_reward=0.0)
    stopped = merge_round(state, pairs, module, 0, 0, state.n_tokens, stop_policy, cost)
    assert stopped.stop_reason == "no_positive_utility"
    assert stopped.state.n_tokens == state.n_tokens


def test_compute_aware_policy_can_merge_early_and_wait_late_for_same_risk():
    torch.manual_seed(11)
    state = make_state()
    pairs = text_adjacent_pairs(state)
    module = MergeModule(16, 12, critic_dim=8)
    cost = TransformerCostModel(16, 64, 12)
    with torch.no_grad():
        prediction = module.value_model(state, pairs, 0)
        minimum_risk = float(prediction.mu_log_damage.exp().min())
    early_saving = cost.suffix_saving_fraction(state.n_tokens, 1, 0, state.n_tokens)
    late_saving = cost.suffix_saving_fraction(state.n_tokens, 1, 10, state.n_tokens)
    reward = 0.5 * (minimum_risk / early_saving + minimum_risk / late_saving)
    policy = PolicyConfig(target_tokens=4, distortion_budget=1.0, confidence_z=0.0, compute_reward=reward, maximum_merges_per_round=1)
    early = merge_round(state, pairs, module, 0, 0, state.n_tokens, policy, cost)
    late = merge_round(state, pairs, module, 10, 0, state.n_tokens, policy, cost)
    assert early.stop_reason == "merged"
    assert late.stop_reason == "no_positive_utility"


def test_cosine_replay_matches_policy_trace_counts():
    class Adapter:
        def to_layer(self, batch, layer_idx):
            return batch.clone()

        def advance(self, state, start_layer, end_layer):
            return state

        def candidates(self, state):
            return text_adjacent_pairs(state)

        def from_layer(self, state, layer_idx):
            return state.hidden.sum(dim=0)

    state = make_state()
    trace = [
        {"layer": 0, "selected_merges": 2},
        {"layer": 2, "selected_merges": 1},
    ]
    output, final_tokens, replayed = replay_cosine_trace(Adapter(), state, trace)
    assert output.shape == (state.dim,)
    assert final_tokens == state.n_tokens - 3
    assert sum(step["selected_merges"] for step in replayed) == 3
