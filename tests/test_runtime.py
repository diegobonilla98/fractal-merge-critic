import torch

from merge_critic.candidates import text_adjacent_pairs
from merge_critic.geometry import text_geometry
from merge_critic.merge_ops import merge_pair
from merge_critic.runtime import selected_pair_indices
from merge_critic.types import TokenState


def make_state(length=10, dim=8):
    return TokenState(
        hidden=torch.randn(length, dim),
        mass=torch.ones(length),
        geometry=text_geometry(length),
        special_mask=torch.zeros(length, dtype=torch.bool),
        groups=[[i] for i in range(length)],
        aux={"modality": "text", "original_length": length},
    )


def test_repeated_single_merges_keep_coverage():
    state = make_state()
    for _ in range(4):
        pairs = text_adjacent_pairs(state)
        pair = pairs[0]
        state = merge_pair(state, int(pair[0]), int(pair[1]))
    represented = [x for group in state.groups for x in group]
    assert sorted(represented) == list(range(10))
    assert len(represented) == len(set(represented))
    assert torch.isclose(state.mass.sum(), torch.tensor(10.0))


def test_text_geometry_tracks_span_endpoints():
    state = make_state(length=4)
    merged = merge_pair(state, 1, 2)
    group_index = merged.groups.index([1, 2])
    geom = merged.geometry[group_index]
    assert torch.isclose(geom[2], torch.tensor(0.25))
    assert torch.isclose(geom[3], torch.tensor(0.75))


def test_selected_pair_indices_preserve_selected_order():
    pairs = torch.tensor([[0, 1], [1, 2], [2, 3], [4, 5]])
    selected = torch.tensor([[4, 5], [1, 2]])
    assert selected_pair_indices(pairs, selected, 6).tolist() == [3, 1]
