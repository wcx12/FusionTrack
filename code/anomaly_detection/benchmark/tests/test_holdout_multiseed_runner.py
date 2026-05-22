from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.run_fusiontrack_holdout_multiseed import (
    aggregate_summary_rows,
    best_by_metric,
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
