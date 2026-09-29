from __future__ import annotations

import time

import torch


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().float().flatten().cpu()
    order = torch.argsort(flat, stable=True)
    sorted_values = flat[order]
    ranks = torch.empty_like(flat)
    start = 0
    while start < flat.numel():
        end = start + 1
        while end < flat.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.numel() < 2 or second.numel() != first.numel():
        return float("nan")
    first_rank = average_ranks(first)
    second_rank = average_ranks(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if denominator == 0:
        return float("nan")
    return float((first_centered * second_centered).sum() / denominator)


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.detach().bool().flatten().cpu()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(scores)
    positive_rank_sum = ranks[labels].sum()
    return float((positive_rank_sum - positives * (positives - 1) / 2) / (positives * negatives))


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    probabilities = probabilities.detach().float().clamp(0.0, 1.0)
    labels = labels.detach().float()
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    error = torch.zeros((), device=probabilities.device)
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if selected.any():
            error += selected.float().mean() * (probabilities[selected].mean() - labels[selected].mean()).abs()
    return float(error)


def false_safe_rate(probabilities: torch.Tensor, labels: torch.Tensor, probability_threshold: float = 0.95) -> tuple[float, int]:
    predicted_safe = probabilities >= probability_threshold
    count = int(predicted_safe.sum())
    if count == 0:
        return float("nan"), 0
    return float((~labels.bool()[predicted_safe]).float().mean()), count


def selection_regret(scores: torch.Tensor, damages: torch.Tensor, fraction: float) -> float:
    count = min(max(1, round(scores.numel() * fraction)), scores.numel())
    selected = torch.argsort(scores)[:count]
    oracle = torch.argsort(damages)[:count]
    return float(damages[selected].float().mean() - damages[oracle].float().mean())


def timed_cuda(fn, warmup: int = 5, repeats: int = 20):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    values = torch.tensor(times)
    return {
        "median_ms": float(values.median() * 1000.0),
        "mean_ms": float(values.mean() * 1000.0),
        "p95_ms": float(torch.quantile(values, 0.95) * 1000.0),
    }
