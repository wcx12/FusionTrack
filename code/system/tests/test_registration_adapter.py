from __future__ import annotations

from pathlib import Path
import json

from fusiontrack.config import FusionTrackPaths
from fusiontrack.pipeline import run_registration_experiment
from fusiontrack.registration_adapter import build_registration_experiment_bundle


def _build_registration_summary() -> dict:
    return {
        "benchmark": {
            "icp_point_to_point": {
                "num_pairs": 2,
                "chamfer_distance_mean": 0.8,
                "rotation_error_mean": 3.0,
                "translation_error_mean": 0.12,
                "runtime_sec_mean": 0.2,
            },
            "fpfh_ransac": {
                "num_pairs": 2,
                "chamfer_distance_mean": 1.2,
                "rotation_error_mean": 4.1,
                "translation_error_mean": 0.22,
                "runtime_sec_mean": 0.5,
            },
        },
        "pair_results": [
            {
                "batch_idx": 0,
                "sample_idx": 0,
                "group_ref_idx": 0,
                "method": "icp_point_to_point",
                "rotation_error_deg": 2.1,
                "translation_error": 0.06,
                "chamfer_distance": 0.7,
                "success": True,
                "registration_points": {
                    "source": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    "reference": [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
                    "aligned": [[0.08, 0.0, 0.0], [1.08, 0.0, 0.0]],
                },
            },
            {
                "batch_idx": 0,
                "sample_idx": 1,
                "group_ref_idx": 0,
                "method": "fpfh_ransac",
                "rotation_error_deg": 8.4,
                "translation_error": 0.31,
                "chamfer_distance": 1.2,
                "success": False,
            },
        ],
        "methods": ["icp_point_to_point", "fpfh_ransac"],
        "args": {"seed": 42},
    }


def test_build_registration_experiment_bundle(tmp_path: Path) -> None:
    summary_path = tmp_path / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(_build_registration_summary()),
        encoding="utf-8",
    )
    bundle = build_registration_experiment_bundle(summary_path, tmp_path)

    assert bundle["num_methods"] == 2
    assert bundle["num_scores"] == 2
    assert Path(bundle["manifest_path"]).exists()
    assert Path(bundle["fused_jsonl"]).exists()
    score_file = tmp_path / next(path for path in bundle["score_files"] if Path(path).name.startswith("icp_"))
    row = json.loads(score_file.read_text(encoding="utf-8").splitlines()[0])
    assert row["registration_points"] == {
        "source": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "reference": [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
        "aligned": [[0.08, 0.0, 0.0], [1.08, 0.0, 0.0]],
    }
    assert row["metadata"]["registration_point_source"] == "benchmark_row"


def test_run_registration_experiment_chain(tmp_path: Path) -> None:
    summary_path = tmp_path / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(_build_registration_summary()),
        encoding="utf-8",
    )
    result = run_registration_experiment(
        paths=FusionTrackPaths.defaults(
            data_root=tmp_path / "data",
            work_root=tmp_path / "work",
        ),
        benchmark_summary=summary_path,
        split="test",
        result_method="icp_point_to_point",
    )

    assert result["mode"] == "experiment_report"
    assert result["experiment"]["method_name"] == "icp_point_to_point"
    assert (tmp_path / "work" / "report" / "index.html").exists()
