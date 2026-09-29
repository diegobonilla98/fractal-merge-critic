import json
from pathlib import Path

import pytest
import torch

from merge_critic.training import GroupedBatchSampler, grouped_split_indices, load_module, train_module
from merge_critic.dataset import PairCalibrationDataset


def calibration_record(index: int) -> dict[str, torch.Tensor | int]:
    generator = torch.Generator().manual_seed(index)
    return {
        "a": torch.randn(16, generator=generator),
        "b": torch.randn(16, generator=generator),
        "ga": torch.randn(6, generator=generator),
        "gb": torch.randn(6, generator=generator),
        "ma": torch.tensor(1.0),
        "mb": torch.tensor(1.0),
        "total_mass": torch.tensor(12.0),
        "context": torch.randn(16, generator=generator),
        "policy_context": torch.randn(7, generator=generator),
        "layer_idx": index % 4,
        "round_idx": index % 2,
        "initial_tokens": 12,
        "current_tokens": 12 - index % 2,
        "target_tokens": 6,
        "distortion_budget": 1e-3,
        "compute_saving": torch.tensor(0.01),
        "damage_raw": torch.tensor(1e-3 + index * 1e-5),
        "damage_terminal": torch.tensor(2e-3 + index * 1e-5),
        "damage_oracle": torch.tensor(5e-4 + index * 1e-5),
        "oracle_delta": torch.randn(16, generator=generator) * 0.01,
        "rollout_rounds": 2,
        "sample_id": index // 4,
        "schema_version": 2,
    }


def test_grouped_split_keeps_samples_disjoint():
    records = [calibration_record(index) for index in range(24)]
    training, validation, grouped = grouped_split_indices(records, 0.25, 17)
    training_samples = {records[index]["sample_id"] for index in training}
    validation_samples = {records[index]["sample_id"] for index in validation}
    assert grouped
    assert training_samples.isdisjoint(validation_samples)
    assert training_samples | validation_samples == set(range(6))


def test_grouped_batch_sampler_never_splits_competing_state_candidates():
    records = []
    for sample_id in range(6):
        for pair_index in range(4):
            record = calibration_record(sample_id * 4 + pair_index)
            record["sample_id"] = sample_id
            record["layer_idx"] = sample_id % 2
            record["round_idx"] = 0
            records.append(record)
    sampler = GroupedBatchSampler(records, list(range(24)), batch_size=10, seed=7, shuffle=True)
    batch_for_index = {}
    for batch_number, batch in enumerate(sampler):
        for index in batch:
            batch_for_index[index] = batch_number
    groups = {}
    for index, record in enumerate(records):
        key = (record["sample_id"], record["layer_idx"], record["round_idx"])
        groups.setdefault(key, []).append(index)
    assert all(len({batch_for_index[index] for index in indices}) == 1 for indices in groups.values())


def test_vectorized_training_checkpoint_and_resume(tmp_path: Path):
    data_path = tmp_path / "calibration.pt"
    output_path = tmp_path / "merge_module.pt"
    torch.save([calibration_record(index) for index in range(24)], data_path)
    first = train_module(
        str(data_path),
        str(output_path),
        hidden_size=16,
        n_layers=4,
        critic_dim=8,
        batch_size=8,
        refiner_epochs=1,
        critic_epochs=1,
        device="cpu",
    )
    last_path = tmp_path / "merge_module.last.pt"
    metrics_path = tmp_path / "merge_module.metrics.jsonl"
    assert output_path.is_file()
    assert last_path.is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])["phase"] == "value_model"
    assert torch.load(last_path, weights_only=False)["status"] == "complete"
    resumed = train_module(
        str(data_path),
        str(output_path),
        hidden_size=16,
        n_layers=4,
        critic_dim=8,
        batch_size=8,
        refiner_epochs=1,
        critic_epochs=1,
        device="cpu",
    )
    loaded = load_module(str(output_path))
    for expected, actual, restored in zip(first.parameters(), resumed.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
        assert torch.equal(expected, restored)


def test_resume_rejects_changed_calibration_data(tmp_path: Path):
    data_path = tmp_path / "calibration.pt"
    output_path = tmp_path / "merge_module.pt"
    records = [calibration_record(index) for index in range(16)]
    torch.save(records, data_path)
    train_module(str(data_path), str(output_path), hidden_size=16, n_layers=4, critic_dim=8, batch_size=8, refiner_epochs=1, critic_epochs=1, device="cpu")
    records.append(calibration_record(16))
    torch.save(records, data_path)
    with pytest.raises(RuntimeError, match="calibration data changed"):
        train_module(str(data_path), str(output_path), hidden_size=16, n_layers=4, critic_dim=8, batch_size=8, refiner_epochs=1, critic_epochs=1, device="cpu")


def test_legacy_data_and_checkpoint_fail_closed(tmp_path: Path):
    legacy_data = tmp_path / "legacy_calibration.pt"
    legacy_checkpoint = tmp_path / "legacy_module.pt"
    torch.save([{"damage_raw": torch.tensor(1e-3)}], legacy_data)
    torch.save({"hidden_size": 16, "n_layers": 4, "critic_dim": 8}, legacy_checkpoint)
    with pytest.raises(ValueError, match="legacy"):
        PairCalibrationDataset(str(legacy_data))
    with pytest.raises(RuntimeError, match="legacy"):
        load_module(str(legacy_checkpoint))
