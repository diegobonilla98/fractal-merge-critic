import json
from pathlib import Path

import torch
from PIL import Image

from merge_critic.adapters import DINOv3VisionAdapter
from merge_critic.merge_ops import merge_pair


model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
image_path = sorted(Path("/mnt/hdd/ImageNet100/val").rglob("*.JPEG"))[0]
output_path = Path("/home/boni/experiments/FractalMergeCritic/dinov3_adapter_audit.json")
device = "cuda"
dtype = torch.bfloat16
layer = 2


def main() -> None:
    adapter = DINOv3VisionAdapter(model_name, device, dtype, proportional_attention=True)
    with Image.open(image_path) as image:
        batch = adapter.prepare(image.convert("RGB"))
    with torch.no_grad():
        reference = adapter.reference(batch)
        equivalence = {}
        for selected_layer in sorted(set([0, 1, adapter.n_layers // 2, adapter.n_layers])):
            output = adapter.from_layer(adapter.to_layer(batch, selected_layer), selected_layer)
            equivalence[str(selected_layer)] = {
                "max_abs": float((reference.float() - output.float()).abs().max()),
                "cosine": float(torch.nn.functional.cosine_similarity(reference.float(), output.float(), dim=0)),
            }
        state = adapter.to_layer(batch, layer)
        pair = adapter.candidates(state)[0]
        merged = merge_pair(state, int(pair[0]), int(pair[1]))
        proportional_output = adapter.from_layer(merged, layer)
        proportional_damage = float(adapter.damage(reference, proportional_output))
        adapter.proportional_attention = False
        unweighted_output = adapter.from_layer(merged, layer)
        unweighted_damage = float(adapter.damage(reference, unweighted_output))
    if min(value["cosine"] for value in equivalence.values()) < 0.99999:
        raise RuntimeError("DINOv3 zero-merge equivalence failed")
    result = {
        "image": str(image_path),
        "equivalence": equivalence,
        "tokens": [state.n_tokens, merged.n_tokens],
        "proportional_damage": proportional_damage,
        "unweighted_damage": unweighted_damage,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
