from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import numpy as np
import pytest


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    def mean(self, dim=None):
        return _Tensor(self.values.mean(axis=dim))

    def tolist(self):
        return self.values.tolist()


class _Device:
    def __init__(self, value):
        self.value = str(value)
        self.type = "cuda" if self.value.startswith("cuda") else "cpu"

    def __str__(self):
        return self.value


def _install_torch_stub() -> None:
    torch_stub = types.ModuleType("torch")
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_utils_data_stub = types.ModuleType("torch.utils.data")
    torch_stub.Tensor = _Tensor
    torch_stub.float32 = object()
    torch_stub.long = object()
    torch_stub.manual_seed = lambda _seed: None
    torch_stub.no_grad = lambda: (lambda func: func)
    torch_stub.device = _Device
    torch_stub.tensor = lambda values, dtype=None: _Tensor(values)
    torch_stub.topk = lambda tensor, k, dim=-1: (
        _Tensor(np.sort(tensor.values, axis=dim)[..., -k:]),
        None,
    )
    torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda _device: None,
    )
    torch_stub.nn = torch_nn_stub
    torch_stub.utils = torch_utils_stub
    torch_nn_stub.Module = object
    torch_utils_stub.data = torch_utils_data_stub
    torch_utils_data_stub.DataLoader = object
    torch_utils_data_stub.Dataset = object
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_nn_stub
    sys.modules["torch.utils"] = torch_utils_stub
    sys.modules["torch.utils.data"] = torch_utils_data_stub


def _restore_modules(previous_modules):
    for name, module in previous_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_recent_runner_with_torch_stub():
    module_names = ("torch", "torch.nn", "torch.utils", "torch.utils.data")
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    _install_torch_stub()
    module_path = Path(__file__).resolve().parents[1] / "runners" / "run_recent_official_fusiontrack.py"
    spec = importlib.util.spec_from_file_location("recent_official_runner_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        _restore_modules(previous_modules)
    return module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

recent_runner = _load_recent_runner_with_torch_stub()
torch = recent_runner.torch
SequenceSample = recent_runner.SequenceSample
_aggregate_time_scores = recent_runner._aggregate_time_scores
_cap_features_len = recent_runner._cap_features_len
_patch_config = recent_runner._patch_config
_resample = recent_runner._resample
_score_row = recent_runner._score_row


def test_recent_runner_uses_expected_cutaddpaste_feature_lengths() -> None:
    assert _cap_features_len(16) == 4
    assert _cap_features_len(32) == 6
    assert _cap_features_len(64) == 10


def test_recent_runner_patch_config_stays_inside_window() -> None:
    patch_size, patch_stride = _patch_config(16)

    assert patch_size == 16
    assert patch_stride == 8


def test_recent_runner_resamples_short_sequences_to_fixed_window() -> None:
    values = np.asarray([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)

    resampled = _resample(values, 3)

    assert resampled.shape == (3, 2)
    assert resampled[1].tolist() == [5.0, 10.0]


def test_recent_runner_aggregates_top_fraction_scores() -> None:
    scores = torch.tensor([[1.0, 2.0, 10.0, 4.0]])

    aggregated = _aggregate_time_scores(scores, 0.5)

    assert aggregated.tolist() == [7.0]


def test_recent_runner_score_row_keeps_group_window_id() -> None:
    args = SimpleNamespace(method="timemixer", task="group")
    sample = SequenceSample(
        sample_id="seq:track",
        sequence="seq",
        track_id="track",
        values=np.zeros((2, 2), dtype=np.float32),
        metadata={"window_id": "w1"},
    )

    row = _score_row(args, sample, 0.25)

    assert row["sample_id"] == "seq:track"
    assert row["window_id"] == "w1"
    assert row["source"] == "official_timemixer:group"
    assert row["score"] == 0.25


def test_recent_runner_scores_optional_score_jsonl_instead_of_clean_validation(
    tmp_path, monkeypatch
) -> None:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "clean_val.jsonl"
    score_path = tmp_path / "injected_score.jsonl"
    output_dir = tmp_path / "out"
    captured = {}

    def sample(sample_id: str, offset: float) -> SequenceSample:
        values = np.asarray([[offset, offset + 1.0], [offset + 2.0, offset + 3.0]], dtype=np.float32)
        return SequenceSample(
            sample_id=sample_id,
            sequence="seq",
            track_id=sample_id,
            values=values,
            metadata={},
        )

    def fake_load_samples(path: Path, task: str, win_size: int) -> list[SequenceSample]:
        assert task == "individual"
        assert win_size == 8
        if path == train_path:
            return [sample("train_sample", 1.0)]
        if path == val_path:
            return [sample("clean_validation_sample", 10.0)]
        if path == score_path:
            return [sample("injected_score_sample", 20.0)]
        raise AssertionError(f"unexpected sample path: {path}")

    def fake_run_timemixer(args, train_samples, val_samples, score_samples, device):
        captured["train_ids"] = [item.sample_id for item in train_samples]
        captured["val_ids"] = [item.sample_id for item in val_samples]
        captured["score_ids"] = [item.sample_id for item in score_samples]
        captured["device"] = str(device)
        return (
            [{"epoch": 1, "train_loss": 0.3, "val_loss": 0.2}],
            [_score_row(args, item, 0.75) for item in score_samples],
            {"mock": True},
        )

    monkeypatch.setattr(recent_runner, "_load_samples", fake_load_samples)
    monkeypatch.setattr(recent_runner, "_run_timemixer", fake_run_timemixer)

    result = recent_runner.main(
        [
            "--method",
            "timemixer",
            "--task",
            "individual",
            "--official-root",
            str(tmp_path / "official"),
            "--train-jsonl",
            str(train_path),
            "--val-jsonl",
            str(val_path),
            "--score-jsonl",
            str(score_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--win-size",
            "8",
        ]
    )

    assert result == 0
    assert captured["train_ids"] == ["train_sample"]
    assert captured["val_ids"] == ["clean_validation_sample"]
    assert captured["score_ids"] == ["injected_score_sample"]
    score_rows = (output_dir / "official_timemixer_individual_scores.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(score_rows) == 1
    assert "injected_score_sample" in score_rows[0]
    assert "clean_validation_sample" not in score_rows[0]
    manifest = recent_runner.json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["score_input_jsonl"] == str(score_path)
    assert manifest["num_val"] == 1
    assert manifest["num_score"] == 1
    assert manifest["manifest_schema_version"] == 2
    assert manifest["generated_at_utc"].endswith("Z")
    assert set(manifest["git"]) >= {"commit", "branch", "dirty"}
    assert set(manifest["environment"]) >= {"python_version", "platform"}
    assert manifest["protocol"] == {
        "method": "timemixer",
        "task": "individual",
        "seed": 42,
        "win_size": 8,
        "score_input_is_validation": False,
    }
    score_jsonl = output_dir / "official_timemixer_individual_scores.jsonl"
    score_csv = output_dir / "official_timemixer_individual_scores.csv"
    assert manifest["artifacts"]["score_jsonl"]["path"] == str(score_jsonl)
    assert manifest["artifacts"]["score_jsonl"]["sha256"] == hashlib.sha256(
        score_jsonl.read_bytes()
    ).hexdigest()
    assert manifest["artifacts"]["score_csv"]["path"] == str(score_csv)
