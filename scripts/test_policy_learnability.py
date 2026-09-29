import datetime
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import default_collate

from merge_critic.metrics import selection_regret, spearman_correlation
from merge_critic.training import load_module, train_module


RUN_ID = os.environ.get("FRACTAL_MERGE_RUN_ID", datetime.datetime.now().strftime("policy_learnability_%Y%m%d_%H%M%S"))
OUTPUT_ROOT = Path("/home/boni/experiments/FractalMergeCritic") / RUN_ID
HIDDEN_SIZE = 32
N_LAYERS = 6
CRITIC_DIM = 32
MAX_REFINE_RATIO = 0.10
TRAINING_SAMPLES = int(os.environ.get("FRACTAL_MERGE_TRAINING_SAMPLES", "96"))
HELDOUT_SAMPLES = int(os.environ.get("FRACTAL_MERGE_HELDOUT_SAMPLES", "32"))
PAIRS_PER_SAMPLE = int(os.environ.get("FRACTAL_MERGE_PAIRS_PER_SAMPLE", "12"))
BATCH_SIZE = int(os.environ.get("FRACTAL_MERGE_BATCH_SIZE", "128"))
MERGE_OPERATOR_EPOCHS = int(os.environ.get("FRACTAL_MERGE_REFINER_EPOCHS", "20"))
VALUE_MODEL_EPOCHS = int(os.environ.get("FRACTAL_MERGE_CRITIC_EPOCHS", "20"))
LEARNING_RATE = 1e-3
RANKING_WEIGHT = 1.0
SELECTION_WEIGHT = 1.0
SEED = 3407


def records(sample_count: int, seed_offset: int) -> list[dict]:
    generator = torch.Generator().manual_seed(SEED + seed_offset)
    projection = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED + 99)) / math.sqrt(HIDDEN_SIZE)
    output = []
    for sample_idx in range(sample_count):
        context = torch.randn(HIDDEN_SIZE, generator=generator)
        layer_idx = sample_idx % N_LAYERS
        round_idx = sample_idx % 3
        initial_tokens = 64
        current_tokens = initial_tokens - round_idx * 8
        target_tokens = 24
        distortion_budget = 1e-3
        policy_context = torch.tensor([
            layer_idx / (N_LAYERS - 1),
            (N_LAYERS - layer_idx) / N_LAYERS,
            round_idx / 40,
            current_tokens / initial_tokens,
            target_tokens / initial_tokens,
            (current_tokens - target_tokens) / initial_tokens,
            math.log10(distortion_budget) / 12,
        ])
        for _ in range(PAIRS_PER_SAMPLE):
            a = torch.randn(HIDDEN_SIZE, generator=generator)
            b = torch.randn(HIDDEN_SIZE, generator=generator)
            ga = torch.randn(6, generator=generator)
            gb = torch.randn(6, generator=generator)
            ma = torch.randint(1, 5, (), generator=generator).float()
            mb = torch.randint(1, 5, (), generator=generator).float()
            cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
            contextual_pair = torch.tanh(context[0]) * (a[0] - b[0]).abs() + 0.5 * torch.tanh(context[1]) * (ga[0] + gb[0])
            log_damage = -6.0 - 2.0 * cosine + 1.6 * contextual_pair + 0.35 * layer_idx + 0.6 * round_idx
            damage_terminal = log_damage.clamp(-11.0, -0.5).exp()
            damage_raw = (-6.0 - 2.0 * cosine + 0.2 * layer_idx).clamp(-11.0, -0.5).exp()
            base = (ma * a + mb * b) / (ma + mb)
            direction = torch.tanh((a + b + 0.5 * context) @ projection)
            oracle_delta = direction / direction.norm().clamp_min(1e-8) * (0.04 * base.norm())
            output.append({
                "a": a,
                "b": b,
                "ga": ga,
                "gb": gb,
                "ma": ma,
                "mb": mb,
                "total_mass": torch.tensor(float(initial_tokens)),
                "context": context,
                "policy_context": policy_context,
                "layer_idx": layer_idx,
                "round_idx": round_idx,
                "initial_tokens": initial_tokens,
                "current_tokens": current_tokens,
                "target_tokens": target_tokens,
                "distortion_budget": distortion_budget,
                "compute_saving": torch.tensor(0.01 * (N_LAYERS - layer_idx) / N_LAYERS),
                "damage_raw": damage_raw,
                "damage_terminal": damage_terminal,
                "damage_oracle": damage_raw * 0.8,
                "oracle_delta": oracle_delta,
                "rollout_rounds": 2,
                "sample_id": sample_idx,
                "schema_version": 2,
            })
    return output


def evaluate(module, heldout: list[dict]) -> dict:
    batch = default_collate(heldout)
    features = (
        batch["a"],
        batch["b"],
        batch["ga"],
        batch["gb"],
        batch["ma"],
        batch["mb"],
        batch["total_mass"],
        batch["context"],
        batch["layer_idx"],
        batch["policy_context"],
    )
    with torch.no_grad():
        value = module.value_model.forward_tensors(*features).mu_log_damage
        delta = module.merge_operator.forward_tensors(*features)
    target_damage = batch["damage_terminal"]
    cosine_score = -torch.nn.functional.cosine_similarity(batch["a"], batch["b"], dim=-1)
    value_score = value.exp()
    oracle_delta = batch["oracle_delta"]
    merge_mse = torch.nn.functional.mse_loss(delta, oracle_delta)
    zero_mse = oracle_delta.pow(2).mean()
    value_regret = []
    cosine_regret = []
    value_within_state_spearman = []
    cosine_within_state_spearman = []
    for sample_idx in range(HELDOUT_SAMPLES):
        start = sample_idx * PAIRS_PER_SAMPLE
        end = start + PAIRS_PER_SAMPLE
        value_regret.append(selection_regret(value_score[start:end], target_damage[start:end], 0.25))
        cosine_regret.append(selection_regret(cosine_score[start:end], target_damage[start:end], 0.25))
        value_within_state_spearman.append(spearman_correlation(value_score[start:end], target_damage[start:end]))
        cosine_within_state_spearman.append(spearman_correlation(cosine_score[start:end], target_damage[start:end]))
    result = {
        "records": len(heldout),
        "value_spearman": spearman_correlation(value_score, target_damage),
        "cosine_spearman": spearman_correlation(cosine_score, target_damage),
        "value_mean_within_state_spearman": float(torch.tensor(value_within_state_spearman).mean()),
        "cosine_mean_within_state_spearman": float(torch.tensor(cosine_within_state_spearman).mean()),
        "value_regret_at_0.25": float(torch.tensor(value_regret).mean()),
        "cosine_regret_at_0.25": float(torch.tensor(cosine_regret).mean()),
        "merge_operator_mse": float(merge_mse),
        "zero_delta_mse": float(zero_mse),
        "merge_operator_error_ratio": float(merge_mse / zero_mse.clamp_min(1e-12)),
    }
    result["checks"] = {
        "value_beats_cosine_spearman_by_0.20": result["value_spearman"] >= result["cosine_spearman"] + 0.20,
        "value_beats_cosine_regret": result["value_regret_at_0.25"] < result["cosine_regret_at_0.25"],
        "merge_operator_halves_zero_error": result["merge_operator_error_ratio"] < 0.5,
    }
    result["passed"] = all(result["checks"].values())
    return result


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data_path = OUTPUT_ROOT / "synthetic_policy_calibration.pt"
    checkpoint_path = OUTPUT_ROOT / "synthetic_policy.pt"
    torch.save(records(TRAINING_SAMPLES, 0), data_path)
    train_module(
        str(data_path),
        str(checkpoint_path),
        HIDDEN_SIZE,
        N_LAYERS,
        CRITIC_DIM,
        MAX_REFINE_RATIO,
        BATCH_SIZE,
        MERGE_OPERATOR_EPOCHS,
        VALUE_MODEL_EPOCHS,
        LEARNING_RATE,
        "damage_terminal",
        RANKING_WEIGHT,
        SELECTION_WEIGHT,
        "cpu",
        seed=SEED,
    )
    module = load_module(str(checkpoint_path), "cpu").eval()
    report = evaluate(module, records(HELDOUT_SAMPLES, 10000))
    (OUTPUT_ROOT / "FINAL_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("controlled policy learnability checks failed")


if __name__ == "__main__":
    main()
