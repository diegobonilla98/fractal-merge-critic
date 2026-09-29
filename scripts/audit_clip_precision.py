import json
from pathlib import Path

import torch
from PIL import Image

from merge_critic.adapters import CLIPVisionAdapter
from merge_critic.merge_ops import counterfactual_batch
from merge_critic.metrics import spearman_correlation


model_name = "openai/clip-vit-base-patch16"
image_path = sorted(Path("/mnt/hdd/coco2017/val2017").glob("*.jpg"))[0]
output_path = Path("/home/boni/experiments/FractalMergeCritic/clip_precision_audit.json")
layer = 2
pair_count = 32
device = "cuda"


def measure(dtype: torch.dtype) -> dict:
    adapter = CLIPVisionAdapter(model_name, device, dtype)
    with Image.open(image_path) as image:
        batch = adapter.prepare(image.convert("RGB"))
    with torch.no_grad():
        reference = adapter.reference(batch)
        state = adapter.to_layer(batch, layer)
        candidates = adapter.candidates(state)
        indices = torch.linspace(0, candidates.shape[0] - 1, pair_count, device=candidates.device).round().long().unique()
        pairs = candidates[indices]
        states = counterfactual_batch(state, pairs)
        batched_outputs = adapter.from_layer_batch(states, layer)
        sequential_outputs = torch.stack([adapter.from_layer(value, layer) for value in states])
        batched_damage = adapter.damage_batch(reference, batched_outputs)
        sequential_damage = adapter.damage_batch(reference, sequential_outputs)
    result = {
        "pairs": int(pairs.shape[0]),
        "batched_mean_damage": float(batched_damage.float().mean()),
        "sequential_mean_damage": float(sequential_damage.float().mean()),
        "damage_mae": float((batched_damage.float() - sequential_damage.float()).abs().mean()),
        "damage_max_abs": float((batched_damage.float() - sequential_damage.float()).abs().max()),
        "damage_spearman": spearman_correlation(batched_damage, sequential_damage),
        "output_min_cosine": float(torch.nn.functional.cosine_similarity(batched_outputs.float(), sequential_outputs.float(), dim=-1).min()),
        "sequential_damage": sequential_damage.float().cpu().tolist(),
    }
    del adapter
    torch.cuda.empty_cache()
    return result


def main() -> None:
    torch.set_float32_matmul_precision("highest")
    bfloat16 = measure(torch.bfloat16)
    float32 = measure(torch.float32)
    cross_precision = spearman_correlation(torch.tensor(bfloat16["sequential_damage"]), torch.tensor(float32["sequential_damage"]))
    result = {"image": str(image_path), "layer": layer, "bfloat16": bfloat16, "float32": float32, "sequential_cross_precision_spearman": cross_precision}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
