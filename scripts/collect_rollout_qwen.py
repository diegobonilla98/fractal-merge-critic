import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from merge_critic.adapters import Qwen3Adapter
from merge_critic.calibration import append_records, collect_records_from_state
from merge_critic.policy import PolicyConfig, PolicyContext, TransformerCostModel
from merge_critic.runtime import merge_round
from merge_critic.training import load_module


MODEL_NAME = "Qwen/Qwen3-0.6B"
TEXT_PATH = Path("/mnt/hdd/contrastive_llm_datasets/multisource_20k/texts.jsonl")
CHECKPOINT = Path("artifacts/qwen_rate_distortion_policy_v2.pt")
RUN_ID = os.environ.get("FRACTAL_MERGE_RUN_ID", datetime.datetime.now().strftime("qwen_policy_rollout_%Y%m%d_%H%M%S"))
OUTPUT_PATH = Path("/home/boni/experiments/FractalMergeCritic") / RUN_ID / "qwen_on_policy_calibration.pt"
STATE_PATH = OUTPUT_PATH.with_suffix(".state.json")
MAX_SAMPLES = 100
MAXIMUM_TOKENS = 256
ROUNDS = 3
ACTIVE_PAIRS_PER_ROUND = 12
FUTURE_ROLLOUT_ROUNDS = 2
TARGET_FRACTION = 0.5
DISTORTION_BUDGET = 1e-2
MAXIMUM_MERGES_PER_ROUND = 4
ORACLE_STEPS = 3
ORACLE_LR = 0.02
MAX_REFINE_RATIO = 0.05
DEVICE = "cuda"
DTYPE = torch.bfloat16
SEED = 3407


def texts() -> list[str]:
    values = []
    with TEXT_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            text = row.get("text", "").strip()
            if text:
                rank = hashlib.sha256(f"{SEED}:{row.get('text_id', row.get('row_index'))}".encode("utf-8")).hexdigest()
                values.append((rank, text))
    values.sort()
    return [text for _, text in values[:MAX_SAMPLES]]


def active_indices(prediction, count: int) -> torch.Tensor:
    distance = (prediction.mu_log_damage - math.log(DISTORTION_BUDGET)).abs()
    score = prediction.log_sigma.exp() / (distance + 1e-4)
    return torch.argsort(score, descending=True)[:count]


def main() -> None:
    torch.manual_seed(SEED)
    adapter = Qwen3Adapter(MODEL_NAME, DEVICE, DTYPE, proportional_attention=True)
    module = load_module(str(CHECKPOINT), adapter.device).to(dtype=DTYPE).eval()
    cost_model = TransformerCostModel.from_adapter(adapter)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    progress = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.is_file() else {}
    completed_samples = int(progress.get("completed_samples", 0))
    record_count = int(progress.get("records", 0))
    if bool(OUTPUT_PATH.is_file()) != bool(completed_samples):
        raise RuntimeError("rollout records and progress state are inconsistent")
    for sample_idx, text in enumerate(texts()[completed_samples:], start=completed_samples):
        encoded = adapter.tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=MAXIMUM_TOKENS)
        batch = {key: value.to(adapter.device) for key, value in encoded.items()}
        with torch.no_grad():
            reference = adapter.reference(batch)
            state = adapter.to_layer(batch, 0)
        initial_tokens = state.n_tokens
        target_tokens = max(1, round(initial_tokens * TARGET_FRACTION))
        policy = PolicyConfig(target_tokens, DISTORTION_BUDGET, maximum_rounds_per_site=ROUNDS, maximum_merges_per_round=MAXIMUM_MERGES_PER_ROUND)
        sample_records = []
        for round_idx in range(ROUNDS):
            pairs = adapter.candidates(state)
            if pairs.numel() == 0:
                break
            with torch.no_grad():
                policy_context = PolicyContext(0, round_idx, initial_tokens, state.n_tokens, target_tokens, DISTORTION_BUDGET, adapter.n_layers).tensor(state.hidden.device, state.hidden.dtype)
                prediction = module.value_model(state, pairs, 0, policy_context=policy_context, epsilon=DISTORTION_BUDGET)
                selected = active_indices(prediction, min(ACTIVE_PAIRS_PER_ROUND, pairs.shape[0]))
            records = collect_records_from_state(
                adapter,
                state,
                0,
                reference,
                pairs[selected],
                ORACLE_STEPS,
                ORACLE_LR,
                MAX_REFINE_RATIO,
                sample_id=sample_idx,
                round_idx=round_idx,
                initial_tokens=initial_tokens,
                target_tokens=target_tokens,
                distortion_budget=DISTORTION_BUDGET,
                rollout_rounds=FUTURE_ROLLOUT_ROUNDS,
                rollout_merges_per_round=MAXIMUM_MERGES_PER_ROUND,
                cost_model=cost_model,
                rollout_module=module,
                rollout_policy=policy,
            )
            sample_records.extend(records)
            with torch.no_grad():
                result = merge_round(state, pairs, module, 0, round_idx, initial_tokens, policy, cost_model)
            if result.stop_reason != "merged":
                break
            state = result.state
        append_records(sample_records, str(OUTPUT_PATH))
        record_count += len(sample_records)
        temporary = STATE_PATH.with_suffix(STATE_PATH.suffix + ".tmp")
        temporary.write_text(json.dumps({"completed_samples": sample_idx + 1, "records": record_count}, indent=2), encoding="utf-8")
        os.replace(temporary, STATE_PATH)
        print(f"sample={sample_idx + 1} records={record_count}", flush=True)


if __name__ == "__main__":
    main()
