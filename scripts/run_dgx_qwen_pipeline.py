import datetime
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TelegramBot

from merge_critic.adapters import Qwen3Adapter
from merge_critic.baselines import replay_cosine_trace
from merge_critic.calibration import collect_records, terminal_damage_batch
from merge_critic.losses import symmetric_kl
from merge_critic.metrics import binary_auroc, expected_calibration_error, false_safe_rate, selection_regret, spearman_correlation, timed_cuda
from merge_critic.merge_ops import counterfactual_batch
from merge_critic.policy import PolicyConfig, PolicyContext, TransformerCostModel
from merge_critic.runtime import merge_round
from merge_critic.sampling import stratified_pairs
from merge_critic.schedule import run_merge_schedule
from merge_critic.training import atomic_torch_save, load_module, train_module


project_root = Path("/home/boni/projects/FractalMergeCritic")
text_path = Path("/mnt/hdd/contrastive_llm_datasets/multisource_20k/texts.jsonl")
run_id = os.environ.get("FRACTAL_MERGE_RUN_ID", datetime.datetime.now().strftime("qwen_%Y%m%d_%H%M%S"))
smoke_mode = os.environ.get("FRACTAL_MERGE_SMOKE", "0") == "1"
telegram_enabled = os.environ.get("FRACTAL_MERGE_TELEGRAM", "1") == "1"
output_root = Path("/home/boni/experiments/FractalMergeCritic") / run_id
model_name = "Qwen/Qwen3-0.6B"
device_map = os.environ.get("FRACTAL_MERGE_DEVICE", "cuda")
dtype = torch.bfloat16 if device_map == "cuda" else torch.float32
seed = 3407
training_samples = 4 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_TRAINING_SAMPLES", "48"))
evaluation_samples = 2 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_EVALUATION_SAMPLES", "12"))
maximum_tokens = 96 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_MAXIMUM_TOKENS", "256"))
pairs_per_sample_layer = 3 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_PAIRS_PER_SAMPLE_LAYER", "8"))
oracle_steps = 1 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_ORACLE_STEPS", "3"))
oracle_learning_rate = 0.02
max_refine_ratio = 0.05
batch_size = 8 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_BATCH_SIZE", "128"))
refiner_epochs = 1 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_REFINER_EPOCHS", "8"))
critic_epochs = 1 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_CRITIC_EPOCHS", "8"))
critic_dim = 48
learning_rate = 3e-4
evaluation_pairs = 6 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_EVALUATION_PAIRS", "32"))
safety_epsilons = [1e-4, 1e-3, 1e-2]
calibration_state_rounds = 2 if smoke_mode else 3
calibration_rollout_rounds = 1 if smoke_mode else 2
rollout_merges_per_round = 1 if smoke_mode else 2
target_fraction = 0.5
distortion_budget = 1e-2
compute_reward_multipliers = [1.0, 64.0, 4096.0]


def clean_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def atomic_json(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(values), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class Reporter:
    def __init__(self):
        self.bot = None
        if telegram_enabled:
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
            self.bot = TelegramBot(poll_updates=False, enable_terminal_commands=False, working_directory=str(project_root))

    def send(self, message: str) -> None:
        print(message, flush=True)
        if self.bot is not None:
            self.bot.send_message(text=message, wait=False)

    def close(self) -> None:
        if self.bot is not None:
            self.bot.stop(wait=True)


def selected_texts() -> tuple[list[dict], list[dict]]:
    rows = []
    with text_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            text = row.get("text", "").strip()
            if text:
                digest = hashlib.sha256(f"{seed}:{row.get('text_id', row.get('row_index'))}".encode("utf-8")).hexdigest()
                rows.append({"rank": digest, "text_id": row.get("text_id"), "source": row.get("source"), "text": text})
    rows.sort(key=lambda row: row["rank"])
    selected = rows[:training_samples + evaluation_samples]
    return selected[:training_samples], selected[training_samples:]


def prepare(adapter: Qwen3Adapter, text: str) -> dict[str, torch.Tensor]:
    encoded = adapter.tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=maximum_tokens)
    return {key: value.to(adapter.device) for key, value in encoded.items()}


@torch.no_grad()
def native_greedy(adapter: Qwen3Adapter, batch: dict[str, torch.Tensor], count: int) -> list[int]:
    outputs = adapter.model(**batch, use_cache=True, return_dict=True)
    cache = outputs.past_key_values
    logits = outputs.logits[0, -1]
    generated = []
    original_length = batch["input_ids"].shape[1]
    for step in range(count):
        token = logits.argmax(dim=-1, keepdim=True)
        generated.append(int(token.item()))
        position_ids = torch.tensor([[original_length + step]], device=adapter.device, dtype=torch.long)
        cache_position = torch.tensor([original_length + step], device=adapter.device, dtype=torch.long)
        outputs = adapter.model(
            input_ids=token.reshape(1, 1),
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1]
    return generated


def zero_merge_validation(adapter: Qwen3Adapter, text: str) -> dict:
    batch = prepare(adapter, text)
    layers = sorted(set([0, 1, adapter.n_layers // 2, adapter.n_layers]))
    with torch.no_grad():
        native = adapter.model(**batch).logits[0, -1]
        layer_results = {}
        for layer in layers:
            output = adapter.from_layer(adapter.to_layer(batch, layer), layer)
            layer_results[str(layer)] = {
                "max_abs": float((output.float() - native.float()).abs().max()),
                "symmetric_kl": float(symmetric_kl(native.unsqueeze(0), output.unsqueeze(0))[0]),
            }
        state = adapter.to_layer(batch, 0)
        prefill, _ = adapter.compressed_prefill(state)
        prefill_kl = float(symmetric_kl(native.unsqueeze(0), prefill.unsqueeze(0))[0])
        native_tokens = native_greedy(adapter, batch, 4)
        adapter_tokens = adapter.generate_token_ids_from_layer0_state(state, 4)
    if max(value["symmetric_kl"] for value in layer_results.values()) > 1e-5 or prefill_kl > 1e-5 or native_tokens != adapter_tokens:
        raise RuntimeError("Qwen zero-merge equivalence failed")
    return {"layers": layer_results, "prefill_symmetric_kl": prefill_kl, "native_tokens": native_tokens, "adapter_tokens": adapter_tokens}


def collect_calibration(adapter: Qwen3Adapter, rows: list[dict], layers: list[int], reporter: Reporter) -> Path:
    path = output_root / "qwen_calibration.pt"
    state_path = output_root / "calibration_state.json"
    records = torch.load(path, weights_only=False) if path.is_file() else []
    calibration_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    completed_samples = int(calibration_state.get("completed_samples", 0))
    if bool(records) != bool(completed_samples):
        raise RuntimeError("calibration records and progress state are inconsistent")
    started = time.perf_counter()
    for sample_index in range(completed_samples, len(rows)):
        row = rows[sample_index]
        batch = prepare(adapter, row["text"])
        for layer in layers:
            records.extend(
                record.cpu_dict()
                for record in collect_records(
                    adapter,
                    batch,
                    layer,
                    pair_count=pairs_per_sample_layer,
                    oracle_steps=oracle_steps,
                    oracle_lr=oracle_learning_rate,
                    max_refine_ratio=max_refine_ratio,
                    seed=seed + sample_index * 100 + layer,
                    sample_id=sample_index,
                    state_rounds=calibration_state_rounds,
                    rollout_rounds=calibration_rollout_rounds,
                    rollout_merges_per_round=rollout_merges_per_round,
                    target_fraction=target_fraction,
                    distortion_budget=distortion_budget,
                )
            )
        atomic_torch_save(records, path)
        elapsed = time.perf_counter() - started
        processed = sample_index + 1 - completed_samples
        samples_per_second = processed / max(elapsed, 1e-8)
        eta_minutes = (len(rows) - sample_index - 1) / max(samples_per_second, 1e-8) / 60.0
        atomic_json(state_path, {"completed_samples": sample_index + 1, "records": len(records), "samples_per_second": samples_per_second, "eta_minutes": eta_minutes})
        reporter.send(f"Qwen calibration {sample_index + 1}/{len(rows)} samples, {len(records)} records, ETA {eta_minutes:.1f} min")
    return path


def evaluate(adapter: Qwen3Adapter, module, rows: list[dict], layers: list[int]) -> dict:
    layer = layers[-1]
    predictions = []
    mus = []
    sigmas = []
    cosines = []
    damages = []
    refined_damages = []
    critic_regrets = []
    cosine_regrets = []
    runtime = None
    cost_model = TransformerCostModel.from_adapter(adapter)
    frontier_values = {str(multiplier): {"initial_tokens": [], "tokens": [], "policy_damage": [], "cosine_damage": []} for multiplier in compute_reward_multipliers}
    module.eval()
    for sample_index, row in enumerate(rows):
        batch = prepare(adapter, row["text"])
        with torch.no_grad():
            reference = adapter.reference(batch)
            state = adapter.to_layer(batch, layer)
            pairs = stratified_pairs(state.hidden, adapter.candidates(state), evaluation_pairs, seed + 10000 + sample_index)
            target_tokens = max(1, round(state.n_tokens * target_fraction))
            policy_context = PolicyContext(layer, 0, state.n_tokens, state.n_tokens, target_tokens, distortion_budget, adapter.n_layers).tensor(state.hidden.device, state.hidden.dtype)
            prediction = module.value_model(state, pairs, layer, policy_context=policy_context)
            cosine = -torch.nn.functional.cosine_similarity(state.hidden[pairs[:, 0]].float(), state.hidden[pairs[:, 1]].float(), dim=-1)
            raw_states = counterfactual_batch(state, pairs)
            damage = terminal_damage_batch(adapter, raw_states, layer, reference, calibration_rollout_rounds, rollout_merges_per_round)
            refined_deltas = module.merge_operator(state, pairs, layer, policy_context=policy_context)
            refined_states = counterfactual_batch(state, pairs, refined_deltas)
            refined_damage = terminal_damage_batch(adapter, refined_states, layer, reference, calibration_rollout_rounds, rollout_merges_per_round)
            initial_tokens = state.n_tokens
            frontier_target = max(1, round(initial_tokens * target_fraction))
            frontier_base = PolicyConfig(frontier_target, distortion_budget).resolved_compute_reward(cost_model, initial_tokens)
            for multiplier in compute_reward_multipliers:
                frontier_policy = PolicyConfig(frontier_target, distortion_budget, compute_reward=frontier_base * multiplier, maximum_rounds_per_site=3, maximum_merges_per_round=max(1, initial_tokens // 8))
                policy_output, policy_stats = run_merge_schedule(adapter, batch, module, frontier_policy, layers, cost_model)
                cosine_output, cosine_tokens, _ = replay_cosine_trace(adapter, batch, policy_stats.trace)
                frontier_values[str(multiplier)]["initial_tokens"].append(initial_tokens)
                frontier_values[str(multiplier)]["tokens"].append(policy_stats.final_tokens)
                frontier_values[str(multiplier)]["policy_damage"].append(float(adapter.damage(reference, policy_output)))
                frontier_values[str(multiplier)]["cosine_damage"].append(float(adapter.damage(reference, cosine_output)))
                if cosine_tokens != policy_stats.final_tokens:
                    raise RuntimeError("cosine trace replay did not match policy token count")
            if runtime is None and adapter.device.type == "cuda":
                layer_zero_state = adapter.to_layer(batch, 0)
                runtime_pairs = adapter.candidates(layer_zero_state)
                runtime_policy = PolicyConfig(target_tokens=max(1, round(layer_zero_state.n_tokens * target_fraction)), distortion_budget=distortion_budget, maximum_rounds_per_site=2, maximum_merges_per_round=max(1, layer_zero_state.n_tokens // 4))
                forced_round = merge_round(layer_zero_state, runtime_pairs, module, 0, 0, layer_zero_state.n_tokens, runtime_policy, cost_model)
                baseline_timing = timed_cuda(lambda: adapter.model(**batch, use_cache=True), warmup=1, repeats=5)
                critic_round_timing = timed_cuda(lambda: merge_round(layer_zero_state, runtime_pairs, module, 0, 0, layer_zero_state.n_tokens, runtime_policy, cost_model), warmup=2, repeats=8)
                compressed_timing = timed_cuda(lambda: adapter.compressed_prefill(forced_round.state), warmup=1, repeats=5)
                compressed_logits, _ = adapter.compressed_prefill(forced_round.state)
                total_compressed_ms = critic_round_timing["median_ms"] + compressed_timing["median_ms"]
                runtime = {
                    "baseline_prefill": baseline_timing,
                    "scoring_matching_refining_round": critic_round_timing,
                    "compressed_prefill": compressed_timing,
                    "end_to_end_estimated_speedup": baseline_timing["median_ms"] / total_compressed_ms,
                    "initial_tokens": layer_zero_state.n_tokens,
                    "final_tokens": forced_round.state.n_tokens,
                    "next_token_symmetric_kl": float(adapter.damage(reference, compressed_logits)),
                }
        predictions.append(prediction.mu_log_damage.exp())
        mus.append(prediction.mu_log_damage)
        sigmas.append(prediction.log_sigma.exp())
        cosines.append(cosine)
        damages.append(damage)
        refined_damages.append(refined_damage)
        critic_regrets.append(selection_regret(prediction.mu_log_damage, damage, 0.25))
        cosine_regrets.append(selection_regret(cosine, damage, 0.25))
    predicted = torch.cat(predictions)
    mu = torch.cat(mus).float()
    sigma = torch.cat(sigmas).float().clamp_min(1e-8)
    cosine = torch.cat(cosines)
    damage = torch.cat(damages)
    refined_damage = torch.cat(refined_damages)
    safety = {}
    for epsilon in safety_epsilons:
        probabilities = 0.5 * (1.0 + torch.erf((math.log(epsilon) - mu) / sigma / math.sqrt(2.0)))
        labels = damage < epsilon
        false_safe, selected_safe = false_safe_rate(probabilities, labels, 0.95)
        safety[str(epsilon)] = {
            "safe_fraction": float(labels.float().mean()),
            "auroc": binary_auroc(probabilities, labels),
            "calibration_error": expected_calibration_error(probabilities, labels),
            "false_safe_rate_at_0.95": false_safe,
            "selected_safe_pairs_at_0.95": selected_safe,
        }
    critic_spearman = spearman_correlation(predicted, damage)
    cosine_spearman = spearman_correlation(cosine, damage)
    eligible = not smoke_mode and damage.numel() >= 300
    frontier = {}
    compressed_frontier_wins = []
    for multiplier, values in frontier_values.items():
        policy_damage = np.asarray(values["policy_damage"])
        cosine_damage = np.asarray(values["cosine_damage"])
        mean_tokens = float(np.mean(values["tokens"]))
        mean_initial_tokens = float(np.mean(values["initial_tokens"]))
        frontier[multiplier] = {
            "mean_initial_tokens": mean_initial_tokens,
            "mean_tokens": mean_tokens,
            "mean_policy_damage": float(policy_damage.mean()),
            "mean_cosine_damage": float(cosine_damage.mean()),
            "policy_win_fraction": float((policy_damage < cosine_damage).mean()),
            "maximum_policy_damage": float(policy_damage.max()),
            "maximum_cosine_damage": float(cosine_damage.max()),
            "maximum_policy_excess_damage": float((policy_damage - cosine_damage).max()),
            "policy_damage_per_sample": policy_damage.tolist(),
            "cosine_damage_per_sample": cosine_damage.tolist(),
        }
        if mean_tokens < 0.95 * mean_initial_tokens:
            compressed_frontier_wins.append(policy_damage.mean() < cosine_damage.mean())
    checks = {
        "critic_beats_cosine_spearman_by_0.05": critic_spearman >= cosine_spearman + 0.05,
        "critic_beats_cosine_regret_at_0.25": float(np.mean(critic_regrets)) < float(np.mean(cosine_regrets)),
        "policy_beats_cosine_on_compressed_frontier": bool(compressed_frontier_wins) and all(compressed_frontier_wins),
    }
    return {
        "pairs": int(damage.numel()),
        "critic_spearman": critic_spearman,
        "cosine_spearman": cosine_spearman,
        "mean_damage": float(damage.float().mean()),
        "mean_refined_damage": float(refined_damage.float().mean()),
        "refiner_improved_fraction": float((refined_damage < damage).float().mean()),
        "target_kind": "multi_round_terminal_damage",
        "rate_distortion_frontier": frontier,
        "critic_regret_at_0.25": float(np.mean(critic_regrets)),
        "cosine_regret_at_0.25": float(np.mean(cosine_regrets)),
        "safety": safety,
        "runtime_one_round_aggressive": runtime,
        "scientific_gate": {
            "eligible": eligible,
            "passed": eligible and all(checks.values()),
            "checks": checks,
            "decision": "green" if eligible and all(checks.values()) else ("smoke_only" if not eligible else "red"),
        },
    }


def main() -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    failure_path = output_root / "failure.json"
    final_path = output_root / "FINAL_REPORT.json"
    previous_elapsed_seconds = 0.0
    if failure_path.is_file():
        previous_elapsed_seconds += float(json.loads(failure_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
    if final_path.is_file():
        previous_elapsed_seconds += float(json.loads(final_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
    reporter = Reporter()
    started = time.perf_counter()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_float32_matmul_precision("high")
        training_rows, evaluation_rows = selected_texts()
        atomic_json(output_root / "text_manifest.json", {"training": training_rows, "evaluation": evaluation_rows})
        reporter.send(f"Starting Qwen {'smoke' if smoke_mode else 'pilot'} with {len(training_rows)} train and {len(evaluation_rows)} held-out texts")
        adapter = Qwen3Adapter(model_name, device_map, dtype, proportional_attention=True)
        equivalence = zero_merge_validation(adapter, training_rows[0]["text"])
        atomic_json(output_root / "zero_merge_equivalence.json", equivalence)
        layers = [0] if smoke_mode else sorted(set([0, adapter.n_layers // 3, 2 * adapter.n_layers // 3, adapter.n_layers - 1]))
        records_path = collect_calibration(adapter, training_rows, layers, reporter)
        checkpoint_path = output_root / "qwen_merge_module.pt"
        train_module(
            str(records_path),
            str(checkpoint_path),
            hidden_size=adapter.hidden_size,
            n_layers=adapter.n_layers,
            critic_dim=critic_dim,
            max_refine_ratio=max_refine_ratio,
            batch_size=batch_size,
            refiner_epochs=refiner_epochs,
            critic_epochs=critic_epochs,
            learning_rate=learning_rate,
            device=str(adapter.device),
            seed=seed,
            resume=True,
            epoch_callback=lambda metric: reporter.send(f"Qwen {metric['phase']} epoch {metric['epoch']}: train {metric['training_loss']:.5f}, validation {metric['validation_loss']:.5f}"),
        )
        module = load_module(str(checkpoint_path), adapter.device).to(dtype=dtype).eval()
        evaluation = evaluate(adapter, module, evaluation_rows, layers)
        attempt_elapsed_seconds = time.perf_counter() - started
        elapsed_seconds = previous_elapsed_seconds + attempt_elapsed_seconds
        report = {"status": "complete", "elapsed_seconds": elapsed_seconds, "attempt_elapsed_seconds": attempt_elapsed_seconds, "model_name": model_name, "evaluation": evaluation}
        atomic_json(final_path, report)
        if failure_path.is_file():
            recovered_path = output_root / f"failure.recovered_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.replace(failure_path, recovered_path)
        reporter.send(f"Qwen run complete: critic Spearman {evaluation['critic_spearman']:.4f}, cosine {evaluation['cosine_spearman']:.4f}")
    except Exception as error:
        failure = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - started}
        atomic_json(failure_path, failure)
        reporter.send(f"Qwen run failed: {error!r}")
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
