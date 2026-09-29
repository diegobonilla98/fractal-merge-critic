import math
from pathlib import Path

import torch
from PIL import Image

from merge_critic.adapters import CLIPVisionAdapter, DINOv3VisionAdapter
from merge_critic.calibration import append_records, collect_records_from_state
from merge_critic.policy import PolicyConfig, PolicyContext, TransformerCostModel
from merge_critic.runtime import merge_round
from merge_critic.training import load_module


MODEL_KIND = "clip"
MODEL_NAME = "openai/clip-vit-base-patch16"
CHECKPOINT = "artifacts/clip_rate_distortion_policy_v2.pt"
CALIBRATION_DIR = "calibration_images"
OUTPUT_PATH = "artifacts/clip_rollout_calibration.pt"
LAYER = 2
MAX_IMAGES = 100
ROUNDS = 3
ACTIVE_PAIRS_PER_ROUND = 12
EPSILON = 1e-3
TARGET_FRACTION = 0.5
MAXIMUM_MERGES_PER_ROUND = 8
FUTURE_ROLLOUT_ROUNDS = 2
ORACLE_STEPS = 6
ORACLE_LR = 0.04
MAX_REFINE_RATIO = 0.10
DEVICE = "cuda"
DTYPE = torch.bfloat16


def images():
    root = Path(CALIBRATION_DIR)
    paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(root.rglob(suffix))
    for path in sorted(paths)[:MAX_IMAGES]:
        yield Image.open(path).convert("RGB")


def build_adapter():
    if MODEL_KIND == "clip":
        return CLIPVisionAdapter(MODEL_NAME, DEVICE, DTYPE)
    return DINOv3VisionAdapter(MODEL_NAME, DEVICE, DTYPE)


def active_indices(pred, count):
    threshold = math.log(EPSILON)
    distance = (pred.mu_log_damage - threshold).abs()
    uncertainty = pred.log_sigma.exp()
    score = uncertainty / (distance + 1e-4)
    return torch.argsort(score, descending=True)[:count]


def main():
    adapter = build_adapter()
    module = load_module(CHECKPOINT, DEVICE).to(dtype=DTYPE).eval()
    cost_model = TransformerCostModel.from_adapter(adapter)
    if Path(OUTPUT_PATH).exists():
        Path(OUTPUT_PATH).unlink()
    for image_idx, image in enumerate(images()):
        batch = adapter.prepare(image)
        with torch.no_grad():
            reference = adapter.reference(batch)
            state = adapter.to_layer(batch, LAYER)
        initial_tokens = state.n_tokens
        target_tokens = max(1, round(initial_tokens * TARGET_FRACTION))
        policy = PolicyConfig(target_tokens=target_tokens, distortion_budget=EPSILON, maximum_rounds_per_site=ROUNDS, maximum_merges_per_round=MAXIMUM_MERGES_PER_ROUND)
        for round_idx in range(ROUNDS):
            pairs = adapter.candidates(state)
            if pairs.shape[0] == 0:
                break
            with torch.no_grad():
                policy_context = PolicyContext(LAYER, round_idx, initial_tokens, state.n_tokens, target_tokens, EPSILON, adapter.n_layers).tensor(state.hidden.device, state.hidden.dtype)
                pred = module.value_model(state, pairs, LAYER, policy_context=policy_context, epsilon=EPSILON)
                idx = active_indices(pred, min(ACTIVE_PAIRS_PER_ROUND, pairs.shape[0]))
            records = collect_records_from_state(
                adapter,
                state,
                LAYER,
                reference,
                pairs[idx],
                ORACLE_STEPS,
                ORACLE_LR,
                MAX_REFINE_RATIO,
                sample_id=image_idx,
                round_idx=round_idx,
                initial_tokens=initial_tokens,
                target_tokens=target_tokens,
                distortion_budget=EPSILON,
                rollout_rounds=FUTURE_ROLLOUT_ROUNDS,
                rollout_merges_per_round=MAXIMUM_MERGES_PER_ROUND,
                cost_model=cost_model,
                rollout_module=module,
                rollout_policy=policy,
            )
            append_records(records, OUTPUT_PATH)
            with torch.no_grad():
                result = merge_round(state, pairs, module, LAYER, round_idx, initial_tokens, policy, cost_model)
            if result.selected_pairs.shape[0] == 0:
                break
            state = result.state
        print(f"image={image_idx + 1}")


if __name__ == "__main__":
    main()
