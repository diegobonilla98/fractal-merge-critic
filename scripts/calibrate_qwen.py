from pathlib import Path

import torch

from merge_critic.adapters import Qwen3Adapter
from merge_critic.calibration import append_records, collect_records


MODEL_NAME = "Qwen/Qwen3-8B"
TEXT_FILE = "calibration_text.txt"
OUTPUT_PATH = "artifacts/qwen_policy_calibration_v2.pt"
LAYERS = [0]
MAX_SAMPLES = 100
MAX_CHARS = 4000
PAIRS_PER_SAMPLE_LAYER = 16
ORACLE_STEPS = 5
ORACLE_LR = 0.02
MAX_REFINE_RATIO = 0.05
STATE_ROUNDS = 3
ROLLOUT_ROUNDS = 2
ROLLOUT_MERGES_PER_ROUND = 2
TARGET_FRACTION = 0.5
DISTORTION_BUDGET = 1e-2
DEVICE_MAP = "auto"
DTYPE = torch.bfloat16


def texts():
    raw = Path(TEXT_FILE).read_text(encoding="utf-8")
    chunks = [x.strip() for x in raw.split("\n\n") if x.strip()]
    for chunk in chunks[:MAX_SAMPLES]:
        yield chunk[:MAX_CHARS]


def main():
    adapter = Qwen3Adapter(MODEL_NAME, DEVICE_MAP, DTYPE)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    if Path(OUTPUT_PATH).exists():
        Path(OUTPUT_PATH).unlink()
    for sample_idx, text in enumerate(texts()):
        batch = adapter.prepare(text)
        for layer in LAYERS:
            records = collect_records(
                adapter,
                batch,
                layer,
                pair_count=PAIRS_PER_SAMPLE_LAYER,
                oracle_steps=ORACLE_STEPS,
                oracle_lr=ORACLE_LR,
                max_refine_ratio=MAX_REFINE_RATIO,
                seed=sample_idx * 100 + layer,
                sample_id=sample_idx,
                state_rounds=STATE_ROUNDS,
                rollout_rounds=ROLLOUT_ROUNDS,
                rollout_merges_per_round=ROLLOUT_MERGES_PER_ROUND,
                target_fraction=TARGET_FRACTION,
                distortion_budget=DISTORTION_BUDGET,
            )
            append_records(records, OUTPUT_PATH)
        print(f"sample={sample_idx + 1}")


if __name__ == "__main__":
    main()
