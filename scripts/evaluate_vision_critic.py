from pathlib import Path

import torch
from PIL import Image

from merge_critic.adapters import CLIPVisionAdapter, DINOv3VisionAdapter
from merge_critic.calibration import terminal_damage_batch
from merge_critic.merge_ops import counterfactual_batch
from merge_critic.policy import PolicyContext
from merge_critic.training import load_module


MODEL_KIND = "clip"
MODEL_NAME = "openai/clip-vit-base-patch16"
CHECKPOINT = "artifacts/clip_rate_distortion_policy_v2.pt"
IMAGE_PATH = "example.jpg"
LAYER = 2
MAX_PAIRS = 128
EPSILON = 1e-3
TARGET_FRACTION = 0.5
ROLLOUT_ROUNDS = 2
ROLLOUT_MERGES_PER_ROUND = 4
DEVICE = "cuda"
DTYPE = torch.bfloat16


def rankdata(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def corr(a, b):
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8))


def build_adapter():
    if MODEL_KIND == "clip":
        return CLIPVisionAdapter(MODEL_NAME, DEVICE, DTYPE)
    return DINOv3VisionAdapter(MODEL_NAME, DEVICE, DTYPE)


def main():
    adapter = build_adapter()
    module = load_module(CHECKPOINT, DEVICE).to(dtype=DTYPE).eval()
    image = Image.open(Path(IMAGE_PATH)).convert("RGB")
    batch = adapter.prepare(image)
    with torch.no_grad():
        reference = adapter.reference(batch)
        state = adapter.to_layer(batch, LAYER)
        pairs = adapter.candidates(state)[:MAX_PAIRS]
        target_tokens = max(1, round(state.n_tokens * TARGET_FRACTION))
        policy_context = PolicyContext(LAYER, 0, state.n_tokens, state.n_tokens, target_tokens, EPSILON, adapter.n_layers).tensor(state.hidden.device, state.hidden.dtype)
        pred = module.value_model(state, pairs, LAYER, policy_context=policy_context, epsilon=EPSILON)
        a = state.hidden[pairs[:, 0]]
        b = state.hidden[pairs[:, 1]]
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        true_damage = terminal_damage_batch(adapter, counterfactual_batch(state, pairs), LAYER, reference, ROLLOUT_ROUNDS, ROLLOUT_MERGES_PER_ROUND)
        predicted_damage = pred.mu_log_damage.exp()
        print({
            "value_model_spearman": corr(rankdata(predicted_damage), rankdata(true_damage)),
            "cosine_spearman": corr(rankdata(-cosine), rankdata(true_damage)),
            "value_model_log_pearson": corr(torch.log(predicted_damage.clamp_min(1e-8)), torch.log(true_damage.clamp_min(1e-8))),
            "safe_fraction_true": float((true_damage < EPSILON).float().mean()),
        })


if __name__ == "__main__":
    main()
