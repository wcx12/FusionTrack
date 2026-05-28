import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "code" / "registration" / "convert_roitr_schema.py"
    spec = importlib.util.spec_from_file_location("convert_roitr_schema_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


convert_roitr_schema = _load_module()


def _write_log(path: Path, tx: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "0 1 60",
                "1 0 0 " + str(tx),
                "0 1 0 0",
                "0 0 1 0",
                "0 0 0 1",
                "",
            ]
        )
    )


def test_parse_registration_log_reads_pairs_and_matrices(tmp_path: Path) -> None:
    log_path = tmp_path / "est.log"
    _write_log(log_path, tx=0.25)

    entries = convert_roitr_schema.parse_registration_log(log_path)

    assert entries[0][0] == (0, 1)
    assert entries[0][1][0, 3] == 0.25


def test_summarize_roitr_outputs_project_pose_metric(tmp_path: Path) -> None:
    est_root = tmp_path / "est" / "3DMatch" / "2500"
    gt_root = tmp_path / "benchmarks" / "3DMatch"
    _write_log(est_root / "scene-a" / "est.log", tx=0.25)
    _write_log(gt_root / "scene-a" / "gt.log", tx=0.20)

    summary = convert_roitr_schema.summarize_roitr(est_root, gt_root)

    assert summary["pairs"] == 1
    assert summary["num_pairs"] == 1
    assert summary["rotation_error_deg_mean"] == pytest.approx(0.0)
    assert summary["rotation_error_deg_rmse"] == pytest.approx(0.0)
    assert summary["translation_error_mean"] == pytest.approx(0.05)
    assert summary["translation_error_rmse"] == pytest.approx(0.05)
    assert summary["chamfer_distance_mean"] is None
    assert summary["pose_metric"] == pytest.approx(2.5)


def test_absolute_paths_are_rejected() -> None:
    with pytest.raises(ValueError):
        convert_roitr_schema._reject_absolute("/tmp/roitr", "est_root")
