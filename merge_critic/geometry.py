from __future__ import annotations

import torch


def vision_grid_geometry(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    cx = (xs.reshape(-1) + 0.5) / width
    cy = (ys.reshape(-1) + 0.5) / height
    w = torch.full_like(cx, 1.0 / width)
    h = torch.full_like(cy, 1.0 / height)
    area = w * h
    depth = torch.zeros_like(cx)
    return torch.stack([cx, cy, w, h, area, depth], dim=-1)


def text_geometry(length: int, device=None, dtype=torch.float32) -> torch.Tensor:
    idx = torch.arange(length, device=device, dtype=dtype)
    denom = max(length - 1, 1)
    center = idx / denom
    span = torch.full_like(center, 1.0 / max(length, 1))
    start = idx / max(length, 1)
    end = (idx + 1) / max(length, 1)
    depth = torch.zeros_like(center)
    one = torch.ones_like(center)
    return torch.stack([center, span, start, end, one, depth], dim=-1)


def merge_geometry(ga: torch.Tensor, gb: torch.Tensor, ma: torch.Tensor, mb: torch.Tensor) -> torch.Tensor:
    total = ma + mb
    center = (ma * ga[..., :2] + mb * gb[..., :2]) / total.clamp_min(1e-8)
    if ga.shape[-1] >= 6:
        if ga[..., 2].max() <= 1.0 and ga[..., 3].max() <= 1.0:
            ax0 = ga[..., 0] - ga[..., 2] * 0.5
            ax1 = ga[..., 0] + ga[..., 2] * 0.5
            ay0 = ga[..., 1] - ga[..., 3] * 0.5
            ay1 = ga[..., 1] + ga[..., 3] * 0.5
            bx0 = gb[..., 0] - gb[..., 2] * 0.5
            bx1 = gb[..., 0] + gb[..., 2] * 0.5
            by0 = gb[..., 1] - gb[..., 3] * 0.5
            by1 = gb[..., 1] + gb[..., 3] * 0.5
            x0 = torch.minimum(ax0, bx0)
            x1 = torch.maximum(ax1, bx1)
            y0 = torch.minimum(ay0, by0)
            y1 = torch.maximum(ay1, by1)
            w = x1 - x0
            h = y1 - y0
            area = w * h
            depth = torch.maximum(ga[..., 5], gb[..., 5]) + 1.0
            return torch.stack([center[..., 0], center[..., 1], w, h, area, depth], dim=-1)
    return (ma * ga + mb * gb) / total.clamp_min(1e-8)


def merge_text_geometry(ga: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
    start = torch.minimum(ga[..., 2], gb[..., 2])
    end = torch.maximum(ga[..., 3], gb[..., 3])
    center = 0.5 * (start + end)
    span = end - start
    one = torch.ones_like(center)
    depth = torch.maximum(ga[..., 5], gb[..., 5]) + 1.0
    return torch.stack([center, span, start, end, one, depth], dim=-1)
