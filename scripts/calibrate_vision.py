from pathlib import Path

import torch
from PIL import Image

from merge_critic.adapters import CLIPVisionAdapter, DINOv3VisionAdapter
from merge_critic.calibration import append_records, collect_records


MODEL_KIND = "clip"
MODEL_NAME = "openai/clip-vit-base-patch16"
CALIBRATION_DIR = "calibration_images"
OUTPUT_PATH = "artifacts/policy_calibration_v2.pt"
LAYERS = [0, 2, 5, 8]
MAX_IMAGES = 200
PAIRS_PER_IMAGE_LAYER = 20
ORACLE_STEPS = 8
ORACLE_LR = 0.05
MAX_REFINE_RATIO = 0.10
STATE_ROUNDS = 3
ROLLOUT_ROUNDS = 2
ROLLOUT_MERGES_PER_ROUND = 4
TARGET_FRACTION = 0.5
DISTORTION_BUDGET = 1e-3
DEVICE = "cuda"
DTYPE = torch.bfloat16


def load_images():
    paths = []
    root = Path(CALIBRATION_DIR)
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(root.rglob(suffix))
    for path in sorted(paths)[:MAX_IMAGES]:
        yield Image.open(path).convert("RGB")


def build_adapter():
    if MODEL_KIND == "clip":
        return CLIPVisionAdapter(MODEL_NAME, DEVICE, DTYPE)
    if MODEL_KIND == "dinov3":
        return DINOv3VisionAdapter(MODEL_NAME, DEVICE, DTYPE)
    raise ValueError(MODEL_KIND)


def main():
    adapter = build_adapter()
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    if Path(OUTPUT_PATH).exists():
        Path(OUTPUT_PATH).unlink()
    for image_idx, image in enumerate(load_images()):
        batch = adapter.prepare(image)
        for layer in LAYERS:
            records = collect_records(
                adapter,
                batch,
                layer,
                pair_count=PAIRS_PER_IMAGE_LAYER,
                oracle_steps=ORACLE_STEPS,
                oracle_lr=ORACLE_LR,
                max_refine_ratio=MAX_REFINE_RATIO,
                seed=image_idx * 100 + layer,
                sample_id=image_idx,
                state_rounds=STATE_ROUNDS,
                rollout_rounds=ROLLOUT_ROUNDS,
                rollout_merges_per_round=ROLLOUT_MERGES_PER_ROUND,
                target_fraction=TARGET_FRACTION,
                distortion_budget=DISTORTION_BUDGET,
            )
            append_records(records, OUTPUT_PATH)
        print(f"image={image_idx + 1}")


if __name__ == "__main__":
    main()
