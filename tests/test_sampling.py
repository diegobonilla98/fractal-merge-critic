import torch

from merge_critic.sampling import stratified_pairs


def test_small_stratified_sample_keeps_low_middle_and_high_similarity():
    hidden = torch.tensor([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.9, 0.1],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    pairs = torch.tensor([[0, index] for index in range(1, 8)])
    selected = stratified_pairs(hidden, pairs, 3, 11)
    similarities = torch.nn.functional.cosine_similarity(hidden[selected[:, 0]], hidden[selected[:, 1]], dim=-1)
    all_similarities = torch.nn.functional.cosine_similarity(hidden[pairs[:, 0]], hidden[pairs[:, 1]], dim=-1)
    assert selected.shape[0] == 3
    assert similarities.min() == all_similarities.min()
    assert similarities.max() == all_similarities.max()
