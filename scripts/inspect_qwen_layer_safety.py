import torch

from merge_critic.adapters import Qwen3Adapter
from merge_critic.merge_ops import merge_pair


MODEL_NAME = "Qwen/Qwen3-8B"
TEXT = "The transformer receives a long prompt and should preserve its next token distribution after safe local token merging."
LAYERS = [0, 4, 12, 24, 32]
MAX_PAIRS = 24
DEVICE_MAP = "auto"
DTYPE = torch.bfloat16


def main():
    adapter = Qwen3Adapter(MODEL_NAME, DEVICE_MAP, DTYPE)
    batch = adapter.prepare(TEXT)
    with torch.no_grad():
        reference = adapter.reference(batch)
        for layer in LAYERS:
            state = adapter.to_layer(batch, layer)
            pairs = adapter.candidates(state)[:MAX_PAIRS]
            values = []
            for pair in pairs:
                merged = merge_pair(state, int(pair[0]), int(pair[1]))
                output = adapter.from_layer(merged, layer)
                values.append(float(adapter.damage(reference, output)))
            values.sort()
            print(layer, values[:5], values[-5:] if values else [])


if __name__ == "__main__":
    main()
