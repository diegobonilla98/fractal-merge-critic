from __future__ import annotations

import math

import torch
from torch import nn

from .types import MergePrediction, TokenState


POLICY_CONTEXT_DIM = 7


class DepthFiLM(nn.Module):
    def __init__(self, n_layers: int, dim: int):
        super().__init__()
        self.n_layers = n_layers
        self.layer_gamma = nn.Embedding(n_layers, dim)
        self.layer_beta = nn.Embedding(n_layers, dim)
        self.stage_gamma = nn.Embedding(3, dim)
        self.stage_beta = nn.Embedding(3, dim)
        nn.init.zeros_(self.layer_gamma.weight)
        nn.init.zeros_(self.layer_beta.weight)
        nn.init.zeros_(self.stage_gamma.weight)
        nn.init.zeros_(self.stage_beta.weight)

    def forward(self, x: torch.Tensor, layer_idx: int | torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(layer_idx, device=x.device, dtype=torch.long)
        if bool(((idx < 0) | (idx >= self.n_layers)).any()):
            raise ValueError("layer index outside value-model depth")
        stage = torch.div(idx * 3, self.n_layers, rounding_mode="floor").clamp_max(2)
        gamma = self.layer_gamma(idx) + self.stage_gamma(stage)
        beta = self.layer_beta(idx) + self.stage_beta(stage)
        return x * (1.0 + gamma) + beta


class PairEncoder(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int, critic_dim: int = 64, geometry_dim: int = 6, context_dim: int = 32, policy_dim: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.critic_dim = critic_dim
        self.token_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, critic_dim),
            nn.GELU(),
            nn.Linear(critic_dim, critic_dim),
        )
        self.depth_film = DepthFiLM(n_layers, critic_dim)
        self.context_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, context_dim), nn.GELU())
        self.geometry_proj = nn.Sequential(nn.Linear(geometry_dim * 2 + 4, 32), nn.GELU(), nn.Linear(32, 32))
        self.policy_proj = nn.Sequential(nn.Linear(POLICY_CONTEXT_DIM, policy_dim), nn.GELU(), nn.Linear(policy_dim, policy_dim))
        self.out_dim = critic_dim * 3 + 2 + context_dim + 32 + policy_dim

    def forward(self, state: TokenState, pairs: torch.Tensor, layer_idx: int, context: torch.Tensor | None = None, policy_context: torch.Tensor | None = None) -> torch.Tensor:
        a = state.hidden[pairs[:, 0]]
        b = state.hidden[pairs[:, 1]]
        ga = state.geometry[pairs[:, 0]]
        gb = state.geometry[pairs[:, 1]]
        ma = state.mass[pairs[:, 0]]
        mb = state.mass[pairs[:, 1]]
        if context is None:
            nonspecial = state.hidden[~state.special_mask]
            context = nonspecial.mean(dim=0) if nonspecial.numel() else state.hidden.mean(dim=0)
        return self.forward_tensors(a, b, ga, gb, ma, mb, state.mass.sum(), context, layer_idx, policy_context)

    def forward_tensors(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        ga: torch.Tensor,
        gb: torch.Tensor,
        ma: torch.Tensor,
        mb: torch.Tensor,
        total_mass: torch.Tensor,
        context: torch.Tensor,
        layer_idx: int | torch.Tensor,
        policy_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ua = self.depth_film(self.token_proj(a), layer_idx)
        ub = self.depth_film(self.token_proj(b), layer_idx)
        raw_cosine = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8).to(ua.dtype).unsqueeze(-1)
        cosine = torch.nn.functional.cosine_similarity(ua, ub, dim=-1, eps=1e-8).unsqueeze(-1)
        pair_repr = torch.cat([ua + ub, (ua - ub).abs(), ua * ub, raw_cosine, cosine], dim=-1)
        ma = ma.clamp_min(1e-8)
        mb = mb.clamp_min(1e-8)
        log_ma = torch.log1p(ma)
        log_mb = torch.log1p(mb)
        extra = torch.stack([
            log_ma + log_mb,
            torch.abs(log_ma - log_mb),
            torch.log1p(ma + mb),
            (ma + mb) / total_mass.clamp_min(1e-8),
        ], dim=-1)
        geom = self.geometry_proj(torch.cat([ga + gb, (ga - gb).abs(), extra], dim=-1))
        if context.ndim == 1:
            context = context.unsqueeze(0).expand(a.shape[0], -1)
        context_features = self.context_proj(context)
        if policy_context is None:
            policy_context = torch.zeros(a.shape[0], POLICY_CONTEXT_DIM, device=a.device, dtype=a.dtype)
        elif policy_context.ndim == 1:
            policy_context = policy_context.unsqueeze(0).expand(a.shape[0], -1)
        policy_features = self.policy_proj(policy_context.to(device=a.device, dtype=a.dtype))
        return torch.cat([pair_repr, geom, context_features, policy_features], dim=-1)


class MergeOperator(nn.Module):
    def __init__(self, pair_encoder: PairEncoder, max_ratio: float = 0.1, hidden: int = 128):
        super().__init__()
        self.pair_encoder = pair_encoder
        self.max_ratio = float(max_ratio)
        self.head = nn.Sequential(nn.Linear(pair_encoder.out_dim, hidden), nn.GELU(), nn.Linear(hidden, pair_encoder.hidden_size))

    def _bounded(self, raw: torch.Tensor, a: torch.Tensor, b: torch.Tensor, ma: torch.Tensor, mb: torch.Tensor) -> torch.Tensor:
        ma_column = ma.unsqueeze(-1)
        mb_column = mb.unsqueeze(-1)
        base = (ma_column * a + mb_column * b) / (ma_column + mb_column).clamp_min(1e-8)
        raw_norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        max_norm = self.max_ratio * base.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return raw * torch.minimum(torch.ones_like(max_norm), max_norm / raw_norm)

    def forward(self, state: TokenState, pairs: torch.Tensor, layer_idx: int, context: torch.Tensor | None = None, policy_context: torch.Tensor | None = None) -> torch.Tensor:
        features = self.pair_encoder(state, pairs, layer_idx, context, policy_context)
        return self._bounded(self.head(features), state.hidden[pairs[:, 0]], state.hidden[pairs[:, 1]], state.mass[pairs[:, 0]], state.mass[pairs[:, 1]])

    def forward_tensors(self, a: torch.Tensor, b: torch.Tensor, ga: torch.Tensor, gb: torch.Tensor, ma: torch.Tensor, mb: torch.Tensor, total_mass: torch.Tensor, context: torch.Tensor, layer_idx: torch.Tensor, policy_context: torch.Tensor) -> torch.Tensor:
        features = self.pair_encoder.forward_tensors(a, b, ga, gb, ma, mb, total_mass, context, layer_idx, policy_context)
        return self._bounded(self.head(features), a, b, ma, mb)


class MergeValueModel(nn.Module):
    def __init__(self, pair_encoder: PairEncoder, n_layers: int, hidden: int = 128, min_log_sigma: float = -5.0, max_log_sigma: float = 2.0):
        super().__init__()
        self.pair_encoder = pair_encoder
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.baseline_bias = nn.Embedding(n_layers, 1)
        self.baseline_slope = nn.Embedding(n_layers, 1)
        nn.init.constant_(self.baseline_bias.weight, -5.0)
        nn.init.constant_(self.baseline_slope.weight, 1.0)
        self.head = nn.Sequential(nn.Linear(pair_encoder.out_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, state: TokenState, pairs: torch.Tensor, layer_idx: int, context: torch.Tensor | None = None, policy_context: torch.Tensor | None = None, epsilon: float | None = None) -> MergePrediction:
        a = state.hidden[pairs[:, 0]]
        b = state.hidden[pairs[:, 1]]
        features = self.pair_encoder(state, pairs, layer_idx, context, policy_context)
        raw_cosine = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8).to(features.dtype)
        return self.predict(features, raw_cosine, layer_idx, epsilon)

    def forward_tensors(self, a: torch.Tensor, b: torch.Tensor, ga: torch.Tensor, gb: torch.Tensor, ma: torch.Tensor, mb: torch.Tensor, total_mass: torch.Tensor, context: torch.Tensor, layer_idx: torch.Tensor, policy_context: torch.Tensor, epsilon: float | None = None) -> MergePrediction:
        features = self.pair_encoder.forward_tensors(a, b, ga, gb, ma, mb, total_mass, context, layer_idx, policy_context)
        raw_cosine = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8).to(features.dtype)
        return self.predict(features, raw_cosine, layer_idx, epsilon)

    def predict(self, features: torch.Tensor, raw_cosine: torch.Tensor, layer_idx: int | torch.Tensor, epsilon: float | None = None) -> MergePrediction:
        idx = torch.as_tensor(layer_idx, device=features.device, dtype=torch.long)
        baseline = self.baseline_bias(idx).squeeze(-1) - torch.nn.functional.softplus(self.baseline_slope(idx).squeeze(-1)) * raw_cosine
        out = self.head(features)
        mu = baseline + out[:, 0]
        log_sigma = out[:, 1].clamp(self.min_log_sigma, self.max_log_sigma)
        safe_probability = None
        if epsilon is not None:
            threshold = math.log(max(epsilon, 1e-12))
            z = (threshold - mu) / log_sigma.exp().clamp_min(1e-8)
            safe_probability = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return MergePrediction(mu, log_sigma, safe_probability)


class MergeModule(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int, critic_dim: int = 64, geometry_dim: int = 6, max_refine_ratio: float = 0.1):
        super().__init__()
        value_encoder = PairEncoder(hidden_size, n_layers, critic_dim, geometry_dim)
        operator_encoder = PairEncoder(hidden_size, n_layers, critic_dim, geometry_dim)
        self.value_model = MergeValueModel(value_encoder, n_layers)
        self.merge_operator = MergeOperator(operator_encoder, max_refine_ratio)

    @property
    def critic(self) -> MergeValueModel:
        return self.value_model

    @property
    def refiner(self) -> MergeOperator:
        return self.merge_operator


MergeCritic = MergeValueModel
MergeRefiner = MergeOperator
