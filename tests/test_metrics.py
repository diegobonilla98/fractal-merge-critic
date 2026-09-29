import math

import torch

from merge_critic.losses import grouped_pairwise_ranking_loss, grouped_topk_selection_loss
from merge_critic.metrics import binary_auroc, expected_calibration_error, false_safe_rate, selection_regret, spearman_correlation


def test_tied_spearman_is_permutation_invariant_within_ties():
    first = torch.tensor([1.0, 1.0, 2.0, 3.0])
    second = torch.tensor([4.0, 4.0, 6.0, 8.0])
    assert math.isclose(spearman_correlation(first, second), 1.0, abs_tol=1e-6)


def test_safety_metrics_and_regret():
    probabilities = torch.tensor([0.05, 0.10, 0.90, 0.95])
    labels = torch.tensor([False, False, True, True])
    assert math.isclose(binary_auroc(probabilities, labels), 1.0, abs_tol=1e-6)
    assert expected_calibration_error(probabilities, labels) < 0.1
    false_safe, count = false_safe_rate(probabilities, labels, 0.9)
    assert false_safe == 0.0
    assert count == 2
    damages = torch.tensor([0.4, 0.1, 0.2, 0.3])
    assert selection_regret(damages, damages, 0.5) == 0.0


def test_pairwise_ranking_loss_rewards_correct_within_state_order():
    targets = torch.tensor([-8.0, -5.0, -2.0, -7.0])
    groups = torch.tensor([1, 1, 1, 2])
    correct = grouped_pairwise_ranking_loss(targets, targets, groups)
    reversed_values = grouped_pairwise_ranking_loss(-targets, targets, groups)
    assert correct < reversed_values


def test_topk_selection_loss_rewards_the_oracle_candidate_set():
    targets = torch.tensor([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.7, 0.6])
    groups = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    correct = grouped_topk_selection_loss(targets, targets, groups, 0.5)
    reversed_values = grouped_topk_selection_loss(-targets, targets, groups, 0.5)
    assert correct < reversed_values


@torch.no_grad()
def test_binary_auroc_accepts_cuda_inputs():
    if not torch.cuda.is_available():
        return
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9], device="cuda")
    labels = torch.tensor([False, False, True, True], device="cuda")
    assert math.isclose(binary_auroc(scores, labels), 1.0, abs_tol=1e-6)
