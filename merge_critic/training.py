from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler

from .dataset import PairCalibrationDataset
from .losses import delta_distillation, grouped_pairwise_ranking_loss, grouped_topk_selection_loss, log_damage_nll
from .model import MergeModule


POLICY_CHECKPOINT_VERSION = 2


class GroupedBatchSampler(Sampler[list[int]]):
    def __init__(self, records: list[dict[str, Any]], indices: list[int], batch_size: int, seed: int, shuffle: bool):
        self.batch_size = batch_size
        groups = {}
        for index in indices:
            record = records[index]
            sample_id = int(record.get("sample_id", -1))
            key = (sample_id, int(record.get("layer_idx", -1)), int(record.get("round_idx", -1))) if sample_id >= 0 else (index, -1, -1)
            groups.setdefault(key, []).append(index)
        values = list(groups.values())
        if shuffle:
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(values), generator=generator).tolist()
            values = [values[index] for index in order]
        self.batches = []
        current = []
        for group in values:
            if current and len(current) + len(group) > batch_size:
                self.batches.append(current)
                current = []
            current.extend(group)
            if len(current) >= batch_size:
                self.batches.append(current)
                current = []
        if current:
            self.batches.append(current)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def atomic_torch_save(values: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    os.replace(temporary, path)


def file_signature(path: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"bytes": os.path.getsize(path), "sha256": digest.hexdigest()}


def grouped_split_indices(records: list[dict[str, Any]], validation_fraction: float, seed: int) -> tuple[list[int], list[int], bool]:
    sample_ids = [int(record.get("sample_id", -1)) for record in records]
    grouped = all(sample_id >= 0 for sample_id in sample_ids) and len(set(sample_ids)) >= 2
    generator = torch.Generator().manual_seed(seed)
    if grouped:
        unique_samples = sorted(set(sample_ids))
        validation_count = min(max(1, round(len(unique_samples) * validation_fraction)), len(unique_samples) - 1)
        permutation = torch.randperm(len(unique_samples), generator=generator).tolist()
        validation_samples = {unique_samples[index] for index in permutation[:validation_count]}
        validation_indices = [index for index, sample_id in enumerate(sample_ids) if sample_id in validation_samples]
        training_indices = [index for index, sample_id in enumerate(sample_ids) if sample_id not in validation_samples]
        return training_indices, validation_indices, True
    validation_count = min(max(1, round(len(records) * validation_fraction)), len(records) - 1)
    permutation = torch.randperm(len(records), generator=generator).tolist()
    return permutation[validation_count:], permutation[:validation_count], False


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


def batch_features(values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (
        values["a"],
        values["b"],
        values["ga"],
        values["gb"],
        values["ma"],
        values["mb"],
        values["total_mass"],
        values["context"],
        values["layer_idx"],
        values["policy_context"],
    )


def batch_loss(module: MergeModule, batch: dict[str, torch.Tensor], phase: str, value_target: str, ranking_weight: float, selection_weight: float, device: torch.device) -> torch.Tensor:
    values = batch_to_device(batch, device)
    features = batch_features(values)
    if phase == "merge_operator":
        prediction = module.merge_operator.forward_tensors(*features)
        return delta_distillation(prediction.float(), values["oracle_delta"].float())
    prediction = module.value_model.forward_tensors(*features)
    target = values[value_target].float()
    regression = log_damage_nll(prediction.mu_log_damage.float(), prediction.log_sigma.float(), target)
    group = values["sample_id"].long() * 1000000 + values["layer_idx"].long() * 1000 + values["round_idx"].long()
    ranking = grouped_pairwise_ranking_loss(prediction.mu_log_damage.float(), torch.log(target.clamp_min(1e-8)), group)
    selection = grouped_topk_selection_loss(prediction.mu_log_damage.float(), target, group)
    return regression + ranking_weight * ranking + selection_weight * selection


def evaluate_epoch(module: MergeModule, loader: DataLoader, phase: str, value_target: str, ranking_weight: float, selection_weight: float, device: torch.device) -> float:
    module.eval()
    total = torch.zeros((), device=device)
    count = 0
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = batch_loss(module, batch, phase, value_target, ranking_weight, selection_weight, device)
            batch_count = int(batch["layer_idx"].shape[0])
            total += loss.detach() * batch_count
            count += batch_count
    return float((total / max(count, 1)).cpu())


def train_module(
    data_path: str,
    output_path: str,
    hidden_size: int,
    n_layers: int,
    critic_dim: int = 64,
    max_refine_ratio: float = 0.1,
    batch_size: int = 256,
    refiner_epochs: int = 8,
    critic_epochs: int = 8,
    learning_rate: float = 3e-4,
    critic_target: str = "damage_terminal",
    ranking_weight: float = 0.5,
    selection_weight: float = 0.5,
    device: str | None = None,
    validation_fraction: float = 0.1,
    seed: int = 3407,
    resume: bool = True,
    epoch_callback=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    selected_device = torch.device(device)
    dataset = PairCalibrationDataset(data_path)
    if len(dataset) < 2:
        raise ValueError("calibration dataset must contain at least two records")
    training_indices, validation_indices, grouped_split = grouped_split_indices(dataset.records, validation_fraction, seed)
    validation_sampler = GroupedBatchSampler(dataset.records, validation_indices, batch_size, seed, False)
    validation_loader = DataLoader(dataset, batch_sampler=validation_sampler)
    module = MergeModule(hidden_size, n_layers, critic_dim, max_refine_ratio=max_refine_ratio).to(selected_device)
    data_signature = file_signature(data_path)
    training_configuration = {
        "checkpoint_version": POLICY_CHECKPOINT_VERSION,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "critic_dim": critic_dim,
        "max_refine_ratio": max_refine_ratio,
        "batch_size": batch_size,
        "merge_operator_epochs": refiner_epochs,
        "value_model_epochs": critic_epochs,
        "learning_rate": learning_rate,
        "value_target": critic_target,
        "ranking_weight": ranking_weight,
        "selection_weight": selection_weight,
        "validation_fraction": validation_fraction,
        "seed": seed,
    }
    output = Path(output_path)
    last_path = output.with_name(output.stem + ".last" + output.suffix)
    metrics_path = output.with_name(output.stem + ".metrics.jsonl")
    phases = [("merge_operator", refiner_epochs), ("value_model", critic_epochs)]
    resume_phase = "merge_operator"
    resume_epoch = 0
    resume_optimizer = None
    best_validation = float("inf")
    if resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=selected_device, weights_only=False)
        if int(checkpoint.get("checkpoint_version", 0)) != POLICY_CHECKPOINT_VERSION:
            raise RuntimeError("legacy isolated-damage checkpoint is incompatible with the sequential policy")
        if checkpoint.get("data_signature") != data_signature:
            raise RuntimeError("calibration data changed since the checkpoint was created")
        if checkpoint.get("training_configuration") != training_configuration:
            raise RuntimeError("training configuration changed; use a new output path")
        module.load_state_dict(checkpoint["state_dict"])
        resume_phase = checkpoint["phase"]
        resume_epoch = int(checkpoint["epoch"])
        resume_optimizer = checkpoint.get("optimizer_state")
        best_validation = float(checkpoint.get("best_validation", best_validation))
        if checkpoint.get("status") == "complete":
            return module
    resume_phase_index = 0 if resume_phase == "merge_operator" else 1
    for phase_index, (phase, epochs) in enumerate(phases):
        if phase_index < resume_phase_index:
            continue
        parameters = module.merge_operator.parameters() if phase == "merge_operator" else module.value_model.parameters()
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4, fused=selected_device.type == "cuda")
        starting_epoch = resume_epoch if phase == resume_phase else 0
        if phase == resume_phase and resume_optimizer is not None:
            optimizer.load_state_dict(resume_optimizer)
        if phase != resume_phase:
            best_validation = float("inf")
        for epoch in range(starting_epoch, epochs):
            module.train()
            training_sampler = GroupedBatchSampler(dataset.records, training_indices, batch_size, seed + phase_index * 100000 + epoch, True)
            training_loader = DataLoader(dataset, batch_sampler=training_sampler)
            total = torch.zeros((), device=selected_device)
            count = 0
            for batch in training_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=selected_device.type, dtype=torch.bfloat16, enabled=selected_device.type == "cuda"):
                    loss = batch_loss(module, batch, phase, critic_target, ranking_weight, selection_weight, selected_device)
                loss.backward()
                optimizer.step()
                batch_count = int(batch["layer_idx"].shape[0])
                total += loss.detach() * batch_count
                count += batch_count
            training_loss = float((total / max(count, 1)).cpu())
            validation_loss = evaluate_epoch(module, validation_loader, phase, critic_target, ranking_weight, selection_weight, selected_device)
            metric = {
                "phase": phase,
                "epoch": epoch + 1,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "training_records": len(training_indices),
                "validation_records": len(validation_indices),
                "grouped_split": grouped_split,
                "data_signature": data_signature,
                "training_configuration": training_configuration,
            }
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metric) + "\n")
            print(json.dumps(metric), flush=True)
            if epoch_callback is not None:
                epoch_callback(metric)
            checkpoint = {
                "checkpoint_version": POLICY_CHECKPOINT_VERSION,
                "model_kind": "sequential_rate_distortion_policy",
                "status": "running",
                "phase": phase,
                "epoch": epoch + 1,
                "state_dict": module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_validation": min(best_validation, validation_loss),
                "hidden_size": hidden_size,
                "n_layers": n_layers,
                "critic_dim": critic_dim,
                "max_refine_ratio": max_refine_ratio,
                "value_target": critic_target,
                "seed": seed,
                "training_indices": training_indices,
                "validation_indices": validation_indices,
                "grouped_split": grouped_split,
                "data_signature": data_signature,
                "training_configuration": training_configuration,
            }
            atomic_torch_save(checkpoint, last_path)
            if validation_loss < best_validation:
                best_validation = validation_loss
                atomic_torch_save(checkpoint, output)
        resume_epoch = 0
        resume_optimizer = None
    final_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    final_checkpoint["status"] = "complete"
    final_checkpoint["optimizer_state"] = None
    atomic_torch_save(final_checkpoint, last_path)
    return module


def load_module(path: str, device: str | torch.device = "cpu") -> MergeModule:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if int(checkpoint.get("checkpoint_version", 0)) != POLICY_CHECKPOINT_VERSION:
        raise RuntimeError("legacy isolated-damage checkpoint is incompatible with the sequential policy")
    module = MergeModule(checkpoint["hidden_size"], checkpoint["n_layers"], checkpoint["critic_dim"], max_refine_ratio=checkpoint["max_refine_ratio"]).to(device)
    module.load_state_dict(checkpoint["state_dict"])
    return module
