from __future__ import annotations

from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusiontrack.context_aware_individual import (
    build_context_feature_table,
    run_context_aware_fusiontrack_baseline,
)


def _trajectory(sequence: str, track_id: str, centers: list[list[float]]) -> dict:
    return {
        "sample_id": f"{sequence}:{track_id}",
        "sequence": sequence,
        "track_id": track_id,
        "points": [
            {"frame_id": index + 1, "fused": {"center_xy": center}}
            for index, center in enumerate(centers)
        ],
    }


def _object(sequence: str, track_id: str, centers: list[list[float]]) -> dict:
    return {
        "sample_id": f"{sequence}:{track_id}",
        "sequence": sequence,
        "track_id": track_id,
        "states": [
            {"frame_id": index + 1, "fused": {"center_xy": center}}
            for index, center in enumerate(centers)
        ],
    }


def _window(sequence: str, objects: list[dict]) -> dict:
    return {
        "sample_id": f"{sequence}:window",
        "sequence": sequence,
        "objects": objects,
    }


def _normal_window(sequence: str) -> dict:
    return _window(
        sequence,
        [
            _object(sequence, "a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            _object(sequence, "b", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
            _object(sequence, "c", [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]),
        ],
    )


def _isolated_window(sequence: str) -> dict:
    return _window(
        sequence,
        [
            _object(sequence, "normal", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            _object(sequence, "neighbor", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
            _object(sequence, "isolated", [[50.0, 50.0], [51.0, 50.0], [52.0, 50.0]]),
        ],
    )


def _single_object_window(sequence: str) -> dict:
    return _window(
        sequence,
        [_object(sequence, "solo", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])],
    )


def test_context_feature_table_marks_isolated_object_as_farther_from_group() -> None:
    table = build_context_feature_table([_isolated_window("seq_score")])
    rows = {row["track_id"]: row for row in table.to_dict("records")}

    isolated = rows["isolated"]
    normal = rows["normal"]

    assert isolated["context_nearest_distance_mean"] > normal["context_nearest_distance_mean"]
    assert isolated["context_neighbor_distance_mean"] > normal["context_neighbor_distance_mean"]
    assert isolated["context_isolation_ratio"] > normal["context_isolation_ratio"]
    assert isolated["context_group_dispersion_mean"] >= normal["context_group_dispersion_mean"]


def test_context_feature_table_marks_single_object_window_as_isolated() -> None:
    table = build_context_feature_table([_single_object_window("seq_solo")])
    row = table.iloc[0].to_dict()

    assert row["track_id"] == "solo"
    assert row["context_nearest_distance_mean"] > 0.0
    assert row["context_neighbor_distance_mean"] > 0.0
    assert row["context_isolation_ratio"] == 1.0


def test_context_aware_baseline_scores_context_outlier_above_normal_neighbor() -> None:
    train_trajectories = [
        _trajectory("seq_train", "a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        _trajectory("seq_train", "b", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
        _trajectory("seq_train", "c", [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]),
    ]
    score_trajectories = [
        _trajectory("seq_score", "isolated", [[50.0, 50.0], [51.0, 50.0], [52.0, 50.0]]),
        _trajectory("seq_score", "normal", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    ]

    rows = run_context_aware_fusiontrack_baseline(
        train_trajectories,
        score_trajectories,
        [_normal_window("seq_train")],
        [_isolated_window("seq_score")],
        n_neighbors=1,
    )
    scores = {row["track_id"]: row["score"] for row in rows}

    assert [row["sample_id"] for row in rows] == ["seq_score:isolated", "seq_score:normal"]
    assert scores["isolated"] > scores["normal"]


def test_context_aware_baseline_outputs_schema_and_handles_missing_context() -> None:
    train_trajectories = [
        _trajectory("seq_train", "a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        _trajectory("seq_train", "b", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
    ]
    score_trajectories = [
        _trajectory("seq_missing", "z", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    ]

    rows = run_context_aware_fusiontrack_baseline(
        train_trajectories,
        score_trajectories,
        [_normal_window("seq_train")],
        [],
        n_neighbors=3,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == "seq_missing:z"
    assert row["sequence"] == "seq_missing"
    assert row["track_id"] == "z"
    assert row["source"] == "fusiontrack_individual:context_aware"
    assert isinstance(row["score"], float)
    assert math.isfinite(row["score"])
    assert row["component_scores"] == {"context_aware_distance": row["score"]}
    assert row["metadata"]["n_neighbors"] == 3
    assert "feature_columns" in row["metadata"]
    assert "individual_feature_columns" in row["metadata"]
    assert "context_feature_columns" in row["metadata"]
    assert set(row["metadata"]["context_feature_columns"]) <= set(
        row["metadata"]["feature_columns"]
    )
