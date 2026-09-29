from pathlib import Path

import torch
from PIL import Image

from merge_critic.adapters import CLIPVisionAdapter, DINOv3VisionAdapter
from merge_critic.metrics import timed_cuda
from merge_critic.policy import PolicyConfig
from merge_critic.schedule import run_merge_schedule
from merge_critic.training import load_module


MODEL_KIND = "clip"
MODEL_NAME = "openai/clip-vit-base-patch16"
CHECKPOINT = "artifacts/clip_rate_distortion_policy_v2.pt"
IMAGE_PATH = "example.jpg"
MERGE_SITES = [1, 5]
TARGET_FRACTION = 0.5
ROUNDS_PER_SITE = 4
EPSILON = 1e-3
DEVICE = "cuda"
DTYPE = torch.bfloat16


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
        initial_tokens = adapter.to_layer(batch, 0).n_tokens
        policy = PolicyConfig(target_tokens=max(1, round(initial_tokens * TARGET_FRACTION)), distortion_budget=EPSILON, maximum_rounds_per_site=ROUNDS_PER_SITE)
        merged, stats = run_merge_schedule(
            adapter,
            batch,
            module,
            policy,
            MERGE_SITES,
        )
        damage = adapter.damage(reference, merged)
    baseline_timing = timed_cuda(lambda: adapter.reference(batch))
    merged_timing = timed_cuda(lambda: run_merge_schedule(adapter, batch, module, policy, MERGE_SITES)[0])
    print({"damage": float(damage), "tokens": (stats.initial_tokens, stats.final_tokens), "merges": stats.merges_per_site, "estimated_compute_saving_fraction": stats.estimated_compute_saving_fraction, "trace": stats.trace})
    print({"baseline": baseline_timing, "merged": merged_timing})


if __name__ == "__main__":
    main()
