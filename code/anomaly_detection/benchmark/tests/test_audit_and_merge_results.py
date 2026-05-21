from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


def _write_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_audit_and_merge_results_cli_fails_on_alignment_issues(tmp_path: Path) -> None:
    metric_path = tmp_path / "bad_metrics.json"
    output_csv = tmp_path / "summary.csv"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/audit_and_merge_results.py")
    _write_metrics(
        metric_path,
        {
            "method": "bad",
            "num_duplicate_label_keys": 0,
            "num_duplicate_score_keys": 1,
            "num_missing_score_keys": 0,
            "num_extra_score_keys": 0,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(metric_path),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Alignment audit failed" in result.stderr
    assert "num_duplicate_score_keys=1" in result.stderr
    assert not output_csv.exists()


def test_audit_and_merge_results_cli_merges_multiple_metrics(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    first = metrics_dir / "a_metrics.json"
    nested = metrics_dir / "nested"
    second = nested / "b_metrics.json"
    output_csv = tmp_path / "summary.csv"
    output_json = tmp_path / "summary.json"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/audit_and_merge_results.py")
    clean_alignment = {
        "num_duplicate_label_keys": 0,
        "num_duplicate_score_keys": 0,
        "num_missing_score_keys": 0,
        "num_extra_score_keys": 0,
    }
    _write_metrics(
        first,
        {
            "method": "baseline_a",
            "split": "val",
            "seed": 7,
            "auroc": 0.75,
            **clean_alignment,
        },
    )
    _write_metrics(
        second,
        {
            "method": "baseline_b",
            "split": "test",
            "seed": 11,
            "auroc": 0.85,
            **clean_alignment,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(metrics_dir),
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with output_csv.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    json_rows = json.loads(output_json.read_text(encoding="utf-8"))
    assert [row["method"] for row in csv_rows] == ["baseline_a", "baseline_b"]
    assert [row["method"] for row in json_rows] == ["baseline_a", "baseline_b"]
    assert json_rows[1]["auroc"] == 0.85
