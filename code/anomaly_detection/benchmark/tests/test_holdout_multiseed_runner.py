from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.run_fusiontrack_holdout_multiseed import (
    aggregate_summary_rows,
    best_by_metric,
    build_holdout_manifest,
)


def test_aggregate_summary_rows_reports_mean_std_and_seeds() -> None:
    rows = [
        {
            "level": "individual",
            "method": "fusiontrack",
            "task": "fusiontrack_individual_ensemble",
            "seed": 42,
            "auprc": "0.10",
            "auroc": "0.60",
            "f1": "0.20",
            "precision_at_k": "0.30",
            "recall_at_k": "0.40",
            "num_missing_score_keys": "0",
            "num_extra_score_keys": "0",
        },
        {
            "level": "individual",
            "method": "fusiontrack",
            "task": "fusiontrack_individual_ensemble",
            "seed": 43,
            "auprc": "0.20",
            "auroc": "0.80",
            "f1": "0.40",
            "precision_at_k": "0.50",
            "recall_at_k": "0.60",
            "num_missing_score_keys": "0",
            "num_extra_score_keys": "0",
        },
    ]

    [aggregate] = aggregate_summary_rows(rows)

    assert aggregate["level"] == "individual"
    assert aggregate["method"] == "fusiontrack"
    assert aggregate["num_runs"] == 2
    assert aggregate["seeds"] == "42,43"
    assert aggregate["auprc_mean"] == pytest.approx(0.15)
    assert aggregate["auprc_std"] > 0.0
    assert aggregate["num_missing_score_keys_mean"] == 0.0
    assert aggregate["num_extra_score_keys_mean"] == 0.0


def test_best_by_metric_uses_metric_mean_fields() -> None:
    rows = [
        {
            "level": "individual",
            "method": "a",
            "task": "task",
            "num_runs": 3,
            "auprc_mean": 0.2,
            "auprc_std": 0.01,
        },
        {
            "level": "group",
            "method": "b",
            "task": "task",
            "num_runs": 3,
            "auprc_mean": 0.3,
            "auprc_std": 0.02,
        },
    ]

    best = best_by_metric(rows)

    assert best["auprc"]["method"] == "b"
    assert best["auprc"]["auprc_mean"] == 0.3


def test_build_holdout_manifest_records_traceability_fields(tmp_path: Path) -> None:
    all_runs_csv = tmp_path / "all_runs.csv"
    aggregate_csv = tmp_path / "aggregate.csv"
    best_json = tmp_path / "best_by_metric.json"
    output_root = tmp_path / "out"
    work_root = tmp_path / "work"
    data_root = tmp_path / "data"
    for directory in (output_root, work_root, data_root):
        directory.mkdir()
    with all_runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "auroc"])
        writer.writeheader()
        writer.writerow({"method": "fusiontrack", "auroc": "0.7"})
    aggregate_csv.write_text("method,auroc_mean\nfusiontrack,0.7\n", encoding="utf-8")
    best_json.write_text(json.dumps({"auroc": {"method": "fusiontrack"}}), encoding="utf-8")
    args = argparse.Namespace(
        data_root=data_root,
        output_root=output_root,
        work_root=work_root,
        train_source_split="train",
        eval_source_split="test",
        individual_anomaly_fraction=0.1,
        group_anomaly_fraction=0.2,
        window_size=16,
        stride=8,
        smoke_max_train=5,
        smoke_max_eval=7,
    )

    manifest = build_holdout_manifest(
        args=args,
        seeds=[42, 43],
        levels=["individual", "group"],
        split_name="test",
        output_root=output_root,
        work_root=work_root,
        all_runs_csv=all_runs_csv,
        aggregate_csv=aggregate_csv,
        best_json=best_json,
    )

    assert manifest["manifest_schema_version"] == 2
    assert manifest["generated_at_utc"].endswith("Z")
    assert manifest["seeds"] == [42, 43]
    assert manifest["levels"] == ["individual", "group"]
    assert set(manifest["git"]) >= {"commit", "branch", "dirty"}
    assert set(manifest["environment"]) >= {"python_version", "platform"}
    assert manifest["protocol"] == {
        "train_source_split": "train",
        "eval_source_split": "test",
        "split_name": "test",
        "individual_anomaly_fraction": 0.1,
        "group_anomaly_fraction": 0.2,
        "window_size": 16,
        "stride": 8,
        "smoke_max_train": 5,
        "smoke_max_eval": 7,
    }
    assert manifest["artifacts"]["all_runs_csv"]["path"] == str(all_runs_csv)
    assert manifest["artifacts"]["all_runs_csv"]["sha256"] == hashlib.sha256(
        all_runs_csv.read_bytes()
    ).hexdigest()
    assert manifest["artifacts"]["aggregate_csv"]["path"] == str(aggregate_csv)
    assert manifest["artifacts"]["best_by_metric_json"]["path"] == str(best_json)
