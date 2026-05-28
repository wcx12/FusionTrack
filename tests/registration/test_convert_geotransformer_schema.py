import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_converter():
    module_path = Path(__file__).resolve().parents[2] / "code" / "registration" / "convert_geotransformer_schema.py"
    spec = importlib.util.spec_from_file_location("convert_geotransformer_schema_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load_converter()


def _write_pair(path: Path, pred: np.ndarray, target: np.ndarray) -> None:
    src = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ref = converter.transform_points(src, target).astype(np.float32)
    np.savez(
        path,
        src_points=src,
        ref_points=ref,
        src_points_f=src,
        ref_points_f=ref,
        src_points_c=src,
        ref_points_c=ref,
        estimated_transform=pred.astype(np.float32),
        transform=target.astype(np.float32),
    )


def test_read_pair_metrics_zero_for_matching_transform(tmp_path: Path) -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [0.25, -0.5, 0.75]
    pair_path = tmp_path / "0_1.npz"
    _write_pair(pair_path, transform, transform)

    metrics = converter.read_pair_metrics(pair_path, "fine")

    assert metrics["rotation_error_deg"] == pytest.approx(0.0)
    assert metrics["translation_error"] == pytest.approx(0.0)
    assert metrics["chamfer_distance"] == pytest.approx(0.0)


def test_rotation_error_projects_slightly_non_orthogonal_inputs() -> None:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag([0.9995, 0.9997, 0.9996])

    assert converter.rotation_error_deg(transform, transform) == pytest.approx(0.0)


def test_comparison_schema_adds_pose_metric() -> None:
    metrics = {
        "rotation_error_deg_mean": 10.0,
        "rotation_error_deg_rmse": 10.0,
        "translation_error_mean": 0.2,
        "translation_error_rmse": 0.2,
        "chamfer_distance_mean": 0.01,
        "chamfer_distance_rmse": 0.01,
        "num_pairs": 2,
    }

    schema = converter.to_comparison_schema(metrics, pose_trans_weight=50.0)

    assert schema["pose_metric"] == pytest.approx(20.0)


def test_validate_relative_paths_rejects_absolute_features_dir() -> None:
    args = type("Args", (), {"features_dir": "/tmp/geotransformer", "output_dir": "runs/out"})()

    with pytest.raises(ValueError, match="features_dir"):
        converter.validate_relative_paths(args)
