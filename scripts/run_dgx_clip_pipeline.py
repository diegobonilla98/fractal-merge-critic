import datetime
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TelegramBot

from merge_critic.adapters import CLIPVisionAdapter
from merge_critic.baselines import replay_cosine_trace
from merge_critic.calibration import collect_records, terminal_damage_batch
from merge_critic.metrics import binary_auroc, expected_calibration_error, false_safe_rate, selection_regret, spearman_correlation, timed_cuda
from merge_critic.merge_ops import counterfactual_batch
from merge_critic.policy import PolicyConfig, PolicyContext, TransformerCostModel
from merge_critic.sampling import stratified_pairs
from merge_critic.schedule import run_merge_schedule
from merge_critic.training import atomic_torch_save, load_module, train_module
from merge_critic.runtime import merge_round


project_root = Path("/home/boni/projects/FractalMergeCritic")
run_id = os.environ.get("FRACTAL_MERGE_RUN_ID", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
smoke_mode = os.environ.get("FRACTAL_MERGE_SMOKE", "0") == "1"
telegram_enabled = os.environ.get("FRACTAL_MERGE_TELEGRAM", "1") == "1"
output_root = Path("/home/boni/experiments/FractalMergeCritic") / run_id
staging_root = Path("/home/boni/datasets/FractalMergeCritic") / run_id
training_roots = {
    "imagenet100": Path("/mnt/hdd/ImageNet100/train"),
    "coco2017": Path("/mnt/hdd/coco2017/train2017"),
}
validation_roots = {
    "imagenet100": Path("/mnt/hdd/ImageNet100/val"),
    "coco2017": Path("/mnt/hdd/coco2017/val2017"),
}
model_name = "openai/clip-vit-base-patch16"
device = os.environ.get("FRACTAL_MERGE_DEVICE", "cuda")
dtype = torch.bfloat16 if device == "cuda" else torch.float32
seed = 3407
maximum_images_per_source = 2 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_IMAGES_PER_SOURCE", "100"))
validation_images_per_source = 1 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_VALIDATION_IMAGES_PER_SOURCE", "20"))
layers = [0, 2] if smoke_mode else [0, 2, 5, 8]
pairs_per_image_layer = 4 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_PAIRS_PER_IMAGE_LAYER", "20"))
oracle_steps = 2 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_ORACLE_STEPS", "8"))
oracle_learning_rate = 0.05
max_refine_ratio = 0.10
calibration_checkpoint_images = 1 if smoke_mode else 10
batch_size = 16 if smoke_mode else 256
refiner_epochs = 1 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_REFINER_EPOCHS", "8"))
critic_epochs = 1 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_CRITIC_EPOCHS", "8"))
critic_dim = 64
learning_rate = 3e-4
evaluation_layer = 2
evaluation_pairs = 8 if smoke_mode else int(os.environ.get("FRACTAL_MERGE_EVALUATION_PAIRS", "128"))
safety_epsilons = [1e-5, 1e-4, 1e-3]
selection_fractions = [0.10, 0.25, 0.50]
calibration_state_rounds = 2 if smoke_mode else 3
calibration_rollout_rounds = 1 if smoke_mode else 2
rollout_merges_per_round = 1 if smoke_mode else 4
target_fraction = 0.5
distortion_budget = 1e-3
compute_reward_multipliers = [1.0, 64.0, 4096.0]


def atomic_json(path: Path, values) -> None:
    def clean(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(values), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


class Reporter:
    def __init__(self):
        self.state = {}
        self.state_path = output_root / "run_state.json"
        self.bot = None
        if telegram_enabled:
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
            self.bot = TelegramBot(poll_updates=False, enable_terminal_commands=False, working_directory=str(project_root))

    def update(self, **values) -> None:
        self.state.update(values)
        self.state["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        atomic_json(self.state_path, self.state)
        if self.bot is not None:
            self.bot.set_state(self.state)

    def send(self, message: str) -> None:
        print(message, flush=True)
        if self.bot is not None:
            self.bot.send_message(text=message, wait=False)

    def close(self) -> None:
        if self.bot is not None:
            self.bot.stop(wait=True)


def image_paths(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def rank_path(path: Path, selected_seed: int) -> str:
    return hashlib.sha256(f"{selected_seed}:{path}".encode("utf-8")).hexdigest()


def stage_training_images() -> list[dict[str, str]]:
    manifest_path = output_root / "training_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = []
    for source_index, (source, root) in enumerate(training_roots.items()):
        selected = sorted(image_paths(root), key=lambda path: rank_path(path, seed + source_index))[:maximum_images_per_source]
        destination_root = staging_root / source
        destination_root.mkdir(parents=True, exist_ok=True)
        for image_index, path in enumerate(selected):
            destination = destination_root / f"{image_index:06d}{path.suffix.lower()}"
            if not destination.is_file():
                shutil.copy2(path, destination)
            manifest.append({"source": source, "original": str(path), "staged": str(destination)})
    manifest.sort(key=lambda value: rank_path(Path(value["original"]), seed))
    atomic_json(manifest_path, manifest)
    return manifest


def zero_merge_validation(adapter: CLIPVisionAdapter, image_path: Path) -> dict[str, dict[str, float]]:
    with Image.open(image_path) as image:
        batch = adapter.prepare(image.convert("RGB"))
    with torch.no_grad():
        native = adapter.model(**batch).image_embeds[0]
        native = torch.nn.functional.normalize(native, dim=-1)
        results = {}
        for layer in [0, 1, 6, adapter.n_layers]:
            output = adapter.from_layer(adapter.to_layer(batch, layer), layer)
            results[str(layer)] = {
                "max_abs": float((output.float() - native.float()).abs().max()),
                "cosine": float(torch.nn.functional.cosine_similarity(output.float(), native.float(), dim=0)),
            }
    parameter_dtype = next(adapter.model.parameters()).dtype
    reduced_precision = parameter_dtype in {torch.bfloat16, torch.float16}
    minimum_cosine = 0.9999 if reduced_precision else 0.99999
    maximum_absolute_error = 0.005 if reduced_precision else 1e-5
    if min(value["cosine"] for value in results.values()) < minimum_cosine or max(value["max_abs"] for value in results.values()) > maximum_absolute_error:
        raise RuntimeError(f"zero-merge CLIP equivalence failed: {results}")
    return results


def collect_calibration(adapter: CLIPVisionAdapter, manifest: list[dict[str, str]], reporter: Reporter) -> Path:
    records_path = output_root / "clip_calibration.pt"
    state_path = output_root / "calibration_state.json"
    records = torch.load(records_path, weights_only=False) if records_path.is_file() else []
    calibration_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    completed_images = int(calibration_state.get("completed_images", 0))
    if bool(records) != bool(completed_images):
        raise RuntimeError("calibration records and progress state are inconsistent")
    starting_image = completed_images
    started = time.perf_counter()
    reporter.update(phase="calibration", completed_images=completed_images, total_images=len(manifest), records=len(records))
    for image_index in range(completed_images, len(manifest)):
        item = manifest[image_index]
        with Image.open(item["staged"]) as image:
            batch = adapter.prepare(image.convert("RGB"))
        image_records = []
        for layer in layers:
            image_records.extend(
                collect_records(
                    adapter,
                    batch,
                    layer,
                    pair_count=pairs_per_image_layer,
                    oracle_steps=oracle_steps,
                    oracle_lr=oracle_learning_rate,
                    max_refine_ratio=max_refine_ratio,
                    seed=seed + image_index * 100 + layer,
                    sample_id=image_index,
                    state_rounds=calibration_state_rounds,
                    rollout_rounds=calibration_rollout_rounds,
                    rollout_merges_per_round=rollout_merges_per_round,
                    target_fraction=target_fraction,
                    distortion_budget=distortion_budget,
                )
            )
        records.extend(record.cpu_dict() for record in image_records)
        completed_images = image_index + 1
        elapsed = max(time.perf_counter() - started, 1e-6)
        images_per_second = (completed_images - starting_image) / elapsed
        eta_minutes = (len(manifest) - completed_images) / max(images_per_second, 1e-9) / 60.0
        reporter.update(
            phase="calibration",
            completed_images=completed_images,
            total_images=len(manifest),
            records=len(records),
            images_per_second=images_per_second,
            eta_minutes=eta_minutes,
        )
        if completed_images % calibration_checkpoint_images == 0 or completed_images == len(manifest):
            atomic_torch_save(records, records_path)
            atomic_json(state_path, {"completed_images": completed_images, "records": len(records)})
            reporter.send(
                f"CLIP calibration {completed_images}/{len(manifest)} images, {len(records):,} records, ETA {eta_minutes:.1f} min"
            )
    return records_path


def evaluate(adapter: CLIPVisionAdapter, module, reporter: Reporter) -> dict:
    predicted_values = []
    mu_values = []
    sigma_values = []
    cosine_values = []
    damage_values = []
    refined_damage_values = []
    regret_values = {
        str(fraction): {"critic": [], "cosine": []}
        for fraction in selection_fractions
    }
    runtime = None
    cost_model = TransformerCostModel.from_adapter(adapter)
    frontier_values = {str(multiplier): {"initial_tokens": [], "tokens": [], "policy_damage": [], "cosine_damage": []} for multiplier in compute_reward_multipliers}
    module.eval()
    for source_index, (source, root) in enumerate(validation_roots.items()):
        selected = sorted(image_paths(root), key=lambda path: rank_path(path, seed + 1000 + source_index))[:validation_images_per_source]
        for path in selected:
            with Image.open(path) as image:
                batch = adapter.prepare(image.convert("RGB"))
            with torch.no_grad():
                reference = adapter.reference(batch)
                state = adapter.to_layer(batch, evaluation_layer)
                pairs = stratified_pairs(state.hidden, adapter.candidates(state), evaluation_pairs, seed + source_index)
                target_tokens = max(1, round(state.n_tokens * target_fraction))
                policy_context = PolicyContext(evaluation_layer, 0, state.n_tokens, state.n_tokens, target_tokens, distortion_budget, adapter.n_layers).tensor(state.hidden.device, state.hidden.dtype)
                with torch.autocast(device_type=adapter.device.type, dtype=torch.bfloat16, enabled=adapter.device.type == "cuda"):
                    prediction = module.value_model(state, pairs, evaluation_layer, policy_context=policy_context)
                a = state.hidden[pairs[:, 0]]
                b = state.hidden[pairs[:, 1]]
                cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
                raw_states = counterfactual_batch(state, pairs)
                damage = terminal_damage_batch(adapter, raw_states, evaluation_layer, reference, calibration_rollout_rounds, rollout_merges_per_round)
                refined_deltas = module.merge_operator(state, pairs, evaluation_layer, policy_context=policy_context)
                refined_states = counterfactual_batch(state, pairs, refined_deltas)
                refined_damage = terminal_damage_batch(adapter, refined_states, evaluation_layer, reference, calibration_rollout_rounds, rollout_merges_per_round)
                initial_tokens = adapter.to_layer(batch, 0).n_tokens
                frontier_target = max(1, round(initial_tokens * target_fraction))
                frontier_base = PolicyConfig(frontier_target, distortion_budget).resolved_compute_reward(cost_model, initial_tokens)
                for multiplier in compute_reward_multipliers:
                    frontier_policy = PolicyConfig(frontier_target, distortion_budget, compute_reward=frontier_base * multiplier, maximum_rounds_per_site=2, maximum_merges_per_round=max(1, initial_tokens // 8))
                    policy_output, policy_stats = run_merge_schedule(adapter, batch, module, frontier_policy, layers, cost_model)
                    cosine_output, cosine_tokens, _ = replay_cosine_trace(adapter, batch, policy_stats.trace)
                    frontier_values[str(multiplier)]["initial_tokens"].append(initial_tokens)
                    frontier_values[str(multiplier)]["tokens"].append(policy_stats.final_tokens)
                    frontier_values[str(multiplier)]["policy_damage"].append(float(adapter.damage(reference, policy_output)))
                    frontier_values[str(multiplier)]["cosine_damage"].append(float(adapter.damage(reference, cosine_output)))
                    if cosine_tokens != policy_stats.final_tokens:
                        raise RuntimeError("cosine trace replay did not match policy token count")
                if runtime is None and adapter.device.type == "cuda":
                    runtime_pairs = adapter.candidates(state)
                    runtime_policy = PolicyConfig(target_tokens=target_tokens, distortion_budget=distortion_budget, maximum_rounds_per_site=2, maximum_merges_per_round=max(1, state.n_tokens // 4))
                    critic_timing = timed_cuda(lambda: module.value_model(state, runtime_pairs, evaluation_layer, policy_context=policy_context), warmup=5, repeats=20)
                    block_timing = timed_cuda(lambda: adapter.advance(state, evaluation_layer, evaluation_layer + 1), warmup=3, repeats=12)
                    merge_round_timing = timed_cuda(lambda: merge_round(state, runtime_pairs, module, evaluation_layer, 0, state.n_tokens, runtime_policy, cost_model), warmup=3, repeats=12)
                    baseline_timing = timed_cuda(lambda: adapter.reference(batch), warmup=2, repeats=8)
                    compressed_timing = timed_cuda(
                        lambda: run_merge_schedule(adapter, batch, module, runtime_policy, [evaluation_layer], cost_model)[0],
                        warmup=2,
                        repeats=8,
                    )
                    compressed_output, compressed_stats = run_merge_schedule(adapter, batch, module, runtime_policy, [evaluation_layer], cost_model)
                    runtime = {
                        "baseline": baseline_timing,
                        "compressed": compressed_timing,
                        "value_model": critic_timing,
                        "one_backbone_block": block_timing,
                        "scoring_matching_refining_round": merge_round_timing,
                        "speedup": baseline_timing["median_ms"] / compressed_timing["median_ms"],
                        "initial_tokens": compressed_stats.initial_tokens,
                        "final_tokens": compressed_stats.final_tokens,
                        "damage": float(adapter.damage(reference, compressed_output)),
                    }
            predicted_values.append(prediction.mu_log_damage.exp())
            mu_values.append(prediction.mu_log_damage)
            sigma_values.append(prediction.log_sigma.exp())
            cosine_values.append(-cosine)
            damage_values.append(damage)
            refined_damage_values.append(refined_damage)
            for fraction in selection_fractions:
                regret_values[str(fraction)]["critic"].append(selection_regret(prediction.mu_log_damage, damage, fraction))
                regret_values[str(fraction)]["cosine"].append(selection_regret(-cosine, damage, fraction))
    predicted = torch.cat(predicted_values)
    mu = torch.cat(mu_values).float()
    sigma = torch.cat(sigma_values).float().clamp_min(1e-8)
    cosine = torch.cat(cosine_values)
    damage = torch.cat(damage_values)
    refined_damage = torch.cat(refined_damage_values)
    safety = {}
    for epsilon in safety_epsilons:
        threshold = torch.tensor(epsilon, device=mu.device).log()
        probabilities = 0.5 * (1.0 + torch.erf((threshold - mu) / sigma / np.sqrt(2.0)))
        labels = damage < epsilon
        false_safe, selected_safe = false_safe_rate(probabilities, labels, 0.95)
        safety[str(epsilon)] = {
            "safe_fraction": float(labels.float().mean()),
            "auroc": binary_auroc(probabilities, labels),
            "calibration_error": expected_calibration_error(probabilities, labels),
            "false_safe_rate_at_0.95": false_safe,
            "selected_safe_pairs_at_0.95": selected_safe,
        }
    regrets = {
        fraction: {
            method: float(np.mean(values))
            for method, values in methods.items()
        }
        for fraction, methods in regret_values.items()
    }
    critic_spearman = spearman_correlation(predicted, damage)
    cosine_spearman = spearman_correlation(cosine, damage)
    safety_candidates = [
        values for values in safety.values()
        if np.isfinite(values["auroc"]) and values["selected_safe_pairs_at_0.95"] >= 20
    ]
    eligible = not smoke_mode and damage.numel() >= 1000
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
        "critic_beats_cosine_regret_at_0.25": regrets["0.25"]["critic"] < regrets["0.25"]["cosine"],
        "safety_auroc_at_least_0.65": any(values["auroc"] >= 0.65 for values in safety_candidates),
        "false_safe_rate_at_most_0.10": any(values["false_safe_rate_at_0.95"] <= 0.10 for values in safety_candidates),
        "calibration_error_at_most_0.15": any(values["calibration_error"] <= 0.15 for values in safety_candidates),
        "policy_beats_cosine_on_compressed_frontier": bool(compressed_frontier_wins) and all(compressed_frontier_wins),
    }
    gate = {
        "eligible": eligible,
        "passed": eligible and all(checks.values()),
        "checks": checks,
        "decision": "green" if eligible and all(checks.values()) else ("smoke_only" if not eligible else "red"),
    }
    result = {
        "pairs": int(damage.numel()),
        "critic_spearman": critic_spearman,
        "cosine_spearman": cosine_spearman,
        "mean_damage": float(damage.float().mean()),
        "mean_refined_damage": float(refined_damage.float().mean()),
        "refiner_improved_fraction": float((refined_damage < damage).float().mean()),
        "target_kind": "multi_round_terminal_damage",
        "rate_distortion_frontier": frontier,
        "safety": safety,
        "selection_regret": regrets,
        "scientific_gate": gate,
        "runtime_one_round_aggressive": runtime,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if adapter.device.type == "cuda" else 0.0,
    }
    atomic_json(output_root / "evaluation.json", result)
    reporter.send(f"Held-out CLIP: critic Spearman={result['critic_spearman']:.4f}, cosine={result['cosine_spearman']:.4f}")
    return result


def resolved_configuration() -> dict:
    packages = {}
    for package in ["torch", "transformers", "accelerate", "numpy", "pillow"]:
        packages[package] = importlib.metadata.version(package)
    return {
        "run_id": run_id,
        "smoke_mode": smoke_mode,
        "model_name": model_name,
        "dtype": str(dtype),
        "seed": seed,
        "maximum_images_per_source": maximum_images_per_source,
        "validation_images_per_source": validation_images_per_source,
        "layers": layers,
        "pairs_per_image_layer": pairs_per_image_layer,
        "oracle_steps": oracle_steps,
        "batch_size": batch_size,
        "refiner_epochs": refiner_epochs,
        "critic_epochs": critic_epochs,
        "packages": packages,
        "safety_epsilons": safety_epsilons,
        "selection_fractions": selection_fractions,
        "calibration_state_rounds": calibration_state_rounds,
        "calibration_rollout_rounds": calibration_rollout_rounds,
        "rollout_merges_per_round": rollout_merges_per_round,
        "target_fraction": target_fraction,
        "distortion_budget": distortion_budget,
        "compute_reward_multipliers": compute_reward_multipliers,
    }


def report_training_epoch(reporter: Reporter, metric: dict) -> None:
    reporter.update(phase="training", training_phase=metric["phase"], epoch=metric["epoch"], training_loss=metric["training_loss"], validation_loss=metric["validation_loss"])
    reporter.send(f"CLIP {metric['phase']} epoch {metric['epoch']}: train {metric['training_loss']:.5f}, validation {metric['validation_loss']:.5f}")


def main() -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    failure_path = output_root / "failure.json"
    final_path = output_root / "FINAL_REPORT.json"
    previous_elapsed_hours = 0.0
    if failure_path.is_file():
        previous_elapsed_hours += float(json.loads(failure_path.read_text(encoding="utf-8")).get("elapsed_hours", 0.0))
    if final_path.is_file():
        previous_elapsed_hours += float(json.loads(final_path.read_text(encoding="utf-8")).get("elapsed_hours", 0.0))
    reporter = Reporter()
    started = time.perf_counter()
    try:
        atomic_json(output_root / "resolved_configuration.json", resolved_configuration())
        reporter.update(status="running", phase="staging", run_id=run_id, output_root=str(output_root))
        reporter.send(f"Starting FractalMergeCritic CLIP {'smoke' if smoke_mode else 'pilot'} run {run_id}")
        manifest = stage_training_images()
        reporter.send(f"Staged {len(manifest)} balanced images on internal NVMe")
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_float32_matmul_precision("high")
        adapter = CLIPVisionAdapter(model_name, device, dtype)
        equivalence = zero_merge_validation(adapter, Path(manifest[0]["staged"]))
        atomic_json(output_root / "zero_merge_equivalence.json", equivalence)
        reporter.send("CLIP zero-merge equivalence passed at layers 0, 1, 6, and 12")
        records_path = collect_calibration(adapter, manifest, reporter)
        reporter.update(phase="training")
        checkpoint_path = output_root / "clip_merge_module.pt"
        module = train_module(
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
            device=device,
            seed=seed,
            resume=True,
            epoch_callback=lambda metric: report_training_epoch(reporter, metric),
        )
        module = load_module(str(checkpoint_path), device).to(dtype=dtype).eval()
        reporter.update(phase="evaluation")
        evaluation = evaluate(adapter, module, reporter)
        attempt_elapsed_hours = (time.perf_counter() - started) / 3600.0
        elapsed_hours = previous_elapsed_hours + attempt_elapsed_hours
        final = {"status": "complete", "elapsed_hours": elapsed_hours, "attempt_elapsed_hours": attempt_elapsed_hours, "evaluation": evaluation}
        atomic_json(final_path, final)
        if failure_path.is_file():
            recovered_path = output_root / f"failure.recovered_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.replace(failure_path, recovered_path)
        reporter.update(status="complete", phase="complete", elapsed_hours=elapsed_hours, attempt_elapsed_hours=attempt_elapsed_hours)
        reporter.send(f"FractalMergeCritic CLIP run complete in {attempt_elapsed_hours:.2f}h this attempt, {elapsed_hours:.2f}h cumulative")
    except Exception as error:
        failure = {
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "elapsed_hours": (time.perf_counter() - started) / 3600.0,
        }
        atomic_json(failure_path, failure)
        reporter.update(status="failed", phase="failed", error=repr(error))
        reporter.send(f"FractalMergeCritic failed: {error!r}")
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
