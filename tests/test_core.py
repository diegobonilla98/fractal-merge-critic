import torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from merge_critic.candidates import text_adjacent_pairs, vision_adjacent_pairs
from merge_critic.geometry import text_geometry, vision_grid_geometry
from merge_critic.losses import cosine_damage, log_damage_nll
from merge_critic.matching import greedy_maximal_matching
from merge_critic.merge_ops import merge_matching, merge_pair, weighted_mean_token
from merge_critic.model import MergeModule, PairEncoder
from merge_critic.types import TokenState
from merge_critic.adapters.clip import CLIPVisionAdapter
from merge_critic.adapters.dinov3 import DINOv3VisionAdapter


def make_vision_state(dim=16):
    hidden = torch.randn(5, dim)
    mass = torch.ones(5)
    geometry = torch.cat([torch.zeros(1, 6), vision_grid_geometry(2, 2)], dim=0)
    special = torch.tensor([True, False, False, False, False])
    groups = [[-1], [0], [1], [2], [3]]
    return TokenState(hidden, mass, geometry, special, groups, {"modality": "vision", "original_grid_hw": (2, 2), "prefix_tokens": 1})


def make_text_state(length=6, dim=16):
    hidden = torch.randn(length, dim)
    mass = torch.ones(length)
    geometry = text_geometry(length)
    special = torch.zeros(length, dtype=torch.bool)
    groups = [[i] for i in range(length)]
    return TokenState(hidden, mass, geometry, special, groups, {"modality": "text", "original_length": length})


def test_weighted_mean_associative():
    a = torch.randn(32)
    b = torch.randn(32)
    c = torch.randn(32)
    ma = torch.tensor(2.0)
    mb = torch.tensor(3.0)
    mc = torch.tensor(5.0)
    ab = weighted_mean_token(a, b, ma, mb)
    left = weighted_mean_token(ab, c, ma + mb, mc)
    bc = weighted_mean_token(b, c, mb, mc)
    right = weighted_mean_token(a, bc, ma, mb + mc)
    direct = (ma * a + mb * b + mc * c) / (ma + mb + mc)
    assert torch.allclose(left, right, atol=1e-6)
    assert torch.allclose(left, direct, atol=1e-6)


def test_pair_encoder_is_symmetric():
    state = make_text_state()
    encoder = PairEncoder(hidden_size=16, n_layers=4, critic_dim=8, geometry_dim=6)
    p1 = torch.tensor([[1, 2]])
    p2 = torch.tensor([[2, 1]])
    context = state.hidden.mean(dim=0)
    f1 = encoder(state, p1, 2, context)
    f2 = encoder(state, p2, 2, context)
    assert torch.allclose(f1, f2, atol=1e-6)


def test_pair_encoder_tensor_batch_matches_state_path():
    state = make_text_state(length=6, dim=16)
    encoder = PairEncoder(hidden_size=16, n_layers=4, critic_dim=8, geometry_dim=6)
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])
    context = state.hidden.mean(dim=0)
    state_features = encoder(state, pairs, 2, context)
    tensor_features = encoder.forward_tensors(
        state.hidden[pairs[:, 0]],
        state.hidden[pairs[:, 1]],
        state.geometry[pairs[:, 0]],
        state.geometry[pairs[:, 1]],
        state.mass[pairs[:, 0]],
        state.mass[pairs[:, 1]],
        state.mass.sum().expand(pairs.shape[0]),
        context.expand(pairs.shape[0], -1),
        torch.full((pairs.shape[0],), 2),
    )
    assert torch.allclose(state_features, tensor_features, atol=1e-6)


def test_refiner_respects_norm_cap():
    state = make_text_state(dim=32)
    module = MergeModule(hidden_size=32, n_layers=4, critic_dim=16, max_refine_ratio=0.05)
    pair = torch.tensor([[1, 2]])
    delta = module.refiner(state, pair, 1)[0]
    a = state.hidden[1]
    b = state.hidden[2]
    base = 0.5 * (a + b)
    assert delta.norm() <= 0.05 * base.norm() + 1e-6


def test_greedy_matching_has_no_overlap():
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [4, 5]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    chosen = greedy_maximal_matching(pairs, scores)
    nodes = chosen.flatten().tolist()
    assert len(nodes) == len(set(nodes))


def test_vision_adjacency_and_mass_preservation():
    state = make_vision_state()
    pairs = vision_adjacent_pairs(state)
    assert pairs.shape[0] == 4
    merged = merge_pair(state, 1, 2)
    assert float(merged.mass.sum()) == float(state.mass.sum())
    represented = sorted(x for g, special in zip(merged.groups, merged.special_mask.tolist()) if not special for x in g)
    assert represented == [0, 1, 2, 3]
    pairs2 = vision_adjacent_pairs(merged)
    assert pairs2.shape[0] >= 2


def test_text_merges_preserve_contiguous_spans():
    state = make_text_state(length=8)
    pairs = text_adjacent_pairs(state)
    chosen = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    merged = merge_matching(state, chosen)
    assert merged.n_tokens == 4
    assert merged.groups == [[0, 1], [2, 3], [4, 5], [6, 7]]
    pairs2 = text_adjacent_pairs(merged)
    merged2 = merge_matching(merged, torch.tensor([[0, 1], [2, 3]]))
    assert merged2.groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert float(merged2.mass.sum()) == 8.0
    assert pairs2.shape[0] == 3


def test_critic_probability_shape_and_range():
    state = make_text_state(dim=24)
    module = MergeModule(hidden_size=24, n_layers=6, critic_dim=12)
    pairs = text_adjacent_pairs(state)
    pred = module.critic(state, pairs, 3, epsilon=1e-3)
    assert pred.safe_probability.shape == (pairs.shape[0],)
    assert torch.all(pred.safe_probability >= 0)
    assert torch.all(pred.safe_probability <= 1)


def test_bfloat16_cosine_damage_is_nonnegative_and_zero_for_identity():
    values = torch.randn(16, 64, dtype=torch.bfloat16)
    normalized = torch.nn.functional.normalize(values.float(), dim=-1).to(torch.bfloat16)
    damage = cosine_damage(normalized, normalized)
    assert torch.all(damage >= 0)
    assert torch.all(damage < 1e-5)


def test_clip_proportional_attention_bias_is_log_mass_and_zero_at_unit_mass():
    adapter = CLIPVisionAdapter.__new__(CLIPVisionAdapter)
    adapter.proportional_attention = True
    unit = adapter._attention_bias(torch.ones(4), 4)
    mass = adapter._attention_bias(torch.tensor([1.0, 2.0, 4.0, 1.0]), 4)
    assert unit.shape == (1, 1, 4, 4)
    assert torch.equal(unit, torch.zeros_like(unit))
    assert torch.allclose(mass[0, 0, 0], torch.log(torch.tensor([1.0, 2.0, 4.0, 1.0])))


def test_tiny_clip_zero_merge_and_proportional_merged_forward():
    config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=8,
        projection_dim=16,
    )
    adapter = CLIPVisionAdapter.__new__(CLIPVisionAdapter)
    adapter.model = CLIPVisionModelWithProjection(config).eval()
    adapter.device = torch.device("cpu")
    adapter.proportional_attention = True
    batch = {"pixel_values": torch.randn(1, 3, 32, 32)}
    with torch.no_grad():
        native = torch.nn.functional.normalize(adapter.model(**batch).image_embeds[0], dim=-1)
        reference = adapter.reference(batch)
        state = adapter.to_layer(batch, 0)
        pair = adapter.candidates(state)[:1]
        merged = merge_pair(state, int(pair[0, 0]), int(pair[0, 1]))
        output = adapter.from_layer(merged, 0)
    assert torch.allclose(reference, native, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(output).all()


def test_dinov3_proportional_attention_bias_is_log_mass():
    adapter = DINOv3VisionAdapter.__new__(DINOv3VisionAdapter)
    adapter.proportional_attention = True
    bias = adapter._attention_bias(torch.tensor([1.0, 3.0, 1.0]), 3)
    assert bias.shape == (1, 1, 3, 3)
    assert torch.allclose(bias[0, 0, 1], torch.log(torch.tensor([1.0, 3.0, 1.0])))


def test_log_damage_loss_keeps_mu_gradient_when_sigma_is_large():
    mu = torch.tensor([0.0], requires_grad=True)
    log_sigma = torch.tensor([2.0], requires_grad=True)
    loss = log_damage_nll(mu, log_sigma, torch.tensor([1e-4]))
    loss.backward()
    assert abs(float(mu.grad)) > 0.1
