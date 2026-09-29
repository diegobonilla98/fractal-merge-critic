from __future__ import annotations

import torch


def log_damage_nll(mu: torch.Tensor, log_sigma: torch.Tensor, damage: torch.Tensor, floor: float = 1e-8, regression_weight: float = 0.25) -> torch.Tensor:
    target = torch.log(damage.clamp_min(floor))
    inv_var = torch.exp(-2.0 * log_sigma)
    nll = (0.5 * (target - mu).pow(2) * inv_var + log_sigma).mean()
    regression = torch.nn.functional.smooth_l1_loss(mu, target)
    return nll + regression_weight * regression


def delta_distillation(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(predicted, target)


def grouped_pairwise_ranking_loss(predicted: torch.Tensor, target: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    predicted_difference = predicted.unsqueeze(1) - predicted.unsqueeze(0)
    target_difference = target.unsqueeze(1) - target.unsqueeze(0)
    same_group = group.unsqueeze(1) == group.unsqueeze(0)
    valid = torch.triu(same_group & (target_difference.abs() > 1e-8), diagonal=1)
    if not bool(valid.any()):
        return predicted.sum() * 0.0
    direction = target_difference.sign()
    gap = target_difference.abs()[valid]
    weight = gap / gap.mean().clamp_min(1e-8)
    return (weight * torch.nn.functional.softplus(-direction[valid] * predicted_difference[valid])).mean()


def grouped_topk_selection_loss(predicted: torch.Tensor, target: torch.Tensor, group: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    losses = []
    for group_value in torch.unique(group):
        selected = group == group_value
        group_predicted = predicted[selected]
        group_target = target[selected]
        if group_predicted.numel() < 2:
            continue
        count = min(max(1, round(group_predicted.numel() * fraction)), group_predicted.numel())
        oracle = torch.argsort(group_target)[:count]
        log_probability = torch.log_softmax(-group_predicted, dim=0)
        losses.append(-log_probability[oracle].mean())
    if not losses:
        return predicted.sum() * 0.0
    return torch.stack(losses).mean()


def cosine_damage(reference: torch.Tensor, counterfactual: torch.Tensor) -> torch.Tensor:
    similarity = torch.nn.functional.cosine_similarity(reference.float(), counterfactual.float(), dim=-1)
    return (1.0 - similarity).clamp_min(0.0)


def symmetric_kl(reference_logits: torch.Tensor, counterfactual_logits: torch.Tensor) -> torch.Tensor:
    p_log = torch.log_softmax(reference_logits.float(), dim=-1)
    q_log = torch.log_softmax(counterfactual_logits.float(), dim=-1)
    p = p_log.exp()
    q = q_log.exp()
    kl_pq = (p * (p_log - q_log)).sum(dim=-1)
    kl_qp = (q * (q_log - p_log)).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)
