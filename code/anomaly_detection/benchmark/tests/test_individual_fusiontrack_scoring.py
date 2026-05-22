from __future__ import annotations

from pathlib import Path
import math
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusiontrack.individual_scoring import (
    _feature_stratified_rank01,
    _rank01,
    run_individual_fusiontrack_baseline,
    run_individual_fusiontrack_ensemble,
)


def _trajectory(track_id: str, centers: list[list[float]]) -> dict:
    return {
        "sample_id": f"seq_a:{track_id}",
        "sequence": "seq_a",
        "track_id": track_id,
        "points": [
            {"frame_id": index + 1, "fused": {"center_xy": center}}
            for index, center in enumerate(centers)
        ],
    }


def test_individual_fusiontrack_nearest_feature_scores_rank_jump_higher() -> None:
    train = [
        _trajectory("train_a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        _trajectory("train_b", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
    ]
    normal = _trajectory("normal", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    jump = _trajectory("jump", [[0.0, 0.0], [1.0, 0.0], [20.0, 0.0]])

    rows = run_individual_fusiontrack_baseline(train, [normal, jump], n_neighbors=1)
    scores = {row["track_id"]: row["score"] for row in rows}

    assert scores["jump"] > scores["normal"]
    for row in rows:
        assert row["source"] == "fusiontrack_individual:nearest_feature"
        assert isinstance(row["score"], float)
        assert math.isfinite(row["score"])
        assert row["metadata"]["n_neighbors"] == 1


def test_individual_fusiontrack_ensemble_combines_ranked_components() -> None:
    train = [
        _trajectory("train_a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        _trajectory("train_b", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
        _trajectory("train_c", [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]]),
    ]
    normal = _trajectory("normal", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    jump = _trajectory("jump", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [25.0, 0.0]])

    rows = run_individual_fusiontrack_ensemble(train, [normal, jump], n_neighbors=1)
    scores = {row["track_id"]: row["score"] for row in rows}

    assert scores["jump"] > scores["normal"]
    for row in rows:
        assert row["source"] == "fusiontrack_individual:ensemble"
        assert isinstance(row["score"], float)
        assert math.isfinite(row["score"])
        assert set(row["component_scores"]) == {
            "nearest_feature_rank",
            "lof_novelty_rank",
            "isolation_forest_rank",
        }
        assert row["metadata"]["method"] == "fusiontrack_individual_ensemble"


def test_feature_stratified_rank_suppresses_scale_only_false_positives() -> None:
    raw_scores = [0.10, 0.20, 0.30, 0.95, 0.96, 0.97]
    feature_df = pd.DataFrame(
        {
            "mean_speed": [1.0, 1.1, 1.2, 20.0, 20.1, 20.2],
        }
    )

    calibrated = _feature_stratified_rank01(
        raw_scores,
        feature_df,
        columns=("mean_speed",),
        bins=2,
        global_weight=0.30,
    )
    global_rank = _rank01(raw_scores)

    assert calibrated[3] < global_rank[3]
    assert calibrated[5] > calibrated[3]
    assert all(0.0 <= score <= 1.0 for score in calibrated)


def test_individual_fusiontrack_ensemble_records_calibration_config() -> None:
    train = [
        _trajectory("train_a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        _trajectory("train_b", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
        _trajectory("train_c", [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]]),
    ]
    normal = _trajectory("normal", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    jump = _trajectory("jump", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [25.0, 0.0]])

    rows = run_individual_fusiontrack_ensemble(
        train,
        [normal, jump],
        n_neighbors=1,
        calibration_columns=("mean_speed",),
        calibration_bins=2,
        calibration_global_weight=0.30,
    )

    for row in rows:
        assert row["metadata"]["calibration"] == {
            "enabled": True,
            "columns": ["mean_speed"],
            "bins": 2,
            "global_weight": 0.30,
        }
        assert "uncalibrated_ensemble_rank" in row["component_scores"]
