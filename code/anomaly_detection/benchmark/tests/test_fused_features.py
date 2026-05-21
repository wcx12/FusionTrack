from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import types
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "individual"))

from fusiontrack.fused_features import (
    FeatureBuildConfig,
    build_fused_feature_sets,
    save_fused_feature_sets,
)


def _install_torch_stub_if_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        return

    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_nn_utils = types.ModuleType("torch.nn.utils")
    torch_nn_utils_rnn = types.ModuleType("torch.nn.utils.rnn")

    class _Module:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def to(self, *_args, **_kwargs):
            return self

        def train(self, *_args, **_kwargs) -> None:
            pass

    class _Layer(_Module):
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("torch stub layer is not executable")

    class _Dataset:
        pass

    class _DataLoader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    nn.Module = _Module
    nn.LSTM = _Layer
    nn.Linear = _Layer
    torch_utils_data.Dataset = _Dataset
    torch_utils_data.DataLoader = _DataLoader
    torch_nn_utils_rnn.pad_sequence = lambda *_args, **_kwargs: None
    torch.Tensor = object
    torch.nn = nn
    torch.utils = torch_utils
    torch_utils.data = torch_utils_data
    torch_nn_utils.rnn = torch_nn_utils_rnn
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data
    sys.modules["torch.nn.utils"] = torch_nn_utils
    sys.modules["torch.nn.utils.rnn"] = torch_nn_utils_rnn


_install_torch_stub_if_missing()

from mtf_ba.feature_training import (
    infer_feature_columns,
    infer_feature_min_length,
    FEATURE_NUM_LAYERS,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _fused_trajectory() -> dict:
    return {
        "sample_id": "seq_a:track_1",
        "sequence": "seq_a",
        "track_id": "track_1",
        "points": [
            {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
            {"frame_id": 2, "fused": {"center_xy": [3.0, 4.0]}},
            {"frame_id": 3, "fused": {"center_xy": [7.0, 4.0]}},
            {"frame_id": 4, "fused": {"center_xy": [7.0, 10.0]}},
        ],
    }


def test_fused_features_imports_with_only_benchmark_on_pythonpath(tmp_path: Path) -> None:
    benchmark_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(benchmark_root)
    code = (
        "import fusiontrack.fused_features as fused_features; "
        "assert hasattr(fused_features, 'build_fused_feature_sets')"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_build_fused_feature_sets_exports_route_speed_and_shape(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "fused.jsonl"
    _write_jsonl(jsonl_path, [_fused_trajectory()])

    feature_sets = build_fused_feature_sets(
        jsonl_path,
        config=FeatureBuildConfig(
            route_step_size=5.0,
            shape_time_step=0.1,
            min_points_per_modality=3,
            shape_min_total_length=0.1,
            shape_min_nonzero_steps=2,
        ),
    )

    assert set(feature_sets) == {"route_fused", "speed_fused", "shape_fused"}
    assert list(feature_sets["route_fused"]["seq_a:track_1"].columns) == [
        "latitude",
        "longitude",
    ]
    assert list(feature_sets["speed_fused"]["seq_a:track_1"].columns) == ["speed"]
    assert list(feature_sets["shape_fused"]["seq_a:track_1"].columns) == [
        "delta_x",
        "delta_y",
    ]


def test_save_fused_feature_sets_writes_expected_pickle_names(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "fused.jsonl"
    _write_jsonl(jsonl_path, [_fused_trajectory()])
    feature_sets = build_fused_feature_sets(jsonl_path)

    paths = save_fused_feature_sets(feature_sets, tmp_path / "features", split="train")

    assert set(paths) == {"route_fused", "speed_fused", "shape_fused"}
    assert Path(paths["route_fused"]).name == "route_fused_train.pkl"
    assert Path(paths["speed_fused"]).name == "speed_fused_train.pkl"
    assert Path(paths["shape_fused"]).name == "shape_fused_train.pkl"
    assert all(Path(path).exists() for path in paths.values())


def test_feature_training_metadata_supports_fused_features() -> None:
    assert infer_feature_columns("route_fused") == ["latitude", "longitude"]
    assert infer_feature_columns("speed_fused") == ["speed"]
    assert infer_feature_columns("shape_fused") == ["delta_x", "delta_y"]
    assert infer_feature_min_length("route_fused", None) == 10
    assert infer_feature_min_length("speed_fused", None) == 10
    assert infer_feature_min_length("shape_fused", None) == 2
    assert FEATURE_NUM_LAYERS["route_fused"] == 2
    assert FEATURE_NUM_LAYERS["speed_fused"] == 1
    assert FEATURE_NUM_LAYERS["shape_fused"] == 2
