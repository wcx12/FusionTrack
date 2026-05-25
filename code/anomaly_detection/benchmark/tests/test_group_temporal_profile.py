from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusiontrack.group_temporal_profile import (
    GROUP_EVENT_COMPONENT_SCORE_KEYS,
    _residual_gated_rank_fusion,
    fit_group_temporal_knn,
    run_group_hybrid_fusiontrack,
    run_group_temporal_knn,
)


def _object(
    track_id: str,
    points: list[tuple[float, float]],
    sample_id: str | None = None,
) -> dict:
    obj = {
        "track_id": track_id,
        "states": [
            {"frame_id": frame_id, "fused": {"center_xy": [x, y]}}
            for frame_id, (x, y) in enumerate(points, start=1)
        ],
    }
    if sample_id is not None:
        obj["sample_id"] = sample_id
    return obj


def _stable_window(window_id: str, sequence: str = "seq_score") -> dict:
    return {
        "window_id": window_id,
        "sequence": sequence,
        "frame_start": 1,
        "frame_end": 4,
        "objects": [
            _object("a", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]),
            _object("b", [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]),
            _object("c", [(0.0, 2.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0)]),
        ],
    }


def _anomalous_window(window_id: str) -> dict:
    return {
        "window_id": window_id,
        "sequence": "seq_score",
        "frame_start": 1,
        "frame_end": 4,
        "objects": [
            _object("a", [(0.0, 0.0), (1.0, 0.0), (12.0, 8.0), (22.0, 15.0)]),
            _object("b", [(0.0, 1.0), (1.0, 1.0), (-10.0, 12.0), (-20.0, 22.0)]),
            _object("c", [(0.0, 2.0), (1.0, 2.0), (2.0, -16.0), (3.0, -26.0)]),
        ],
    }


def _train_windows() -> list[dict]:
    return [_stable_window(f"train_{index}", sequence="seq_train") for index in range(6)]


def test_anomalous_group_window_scores_higher_than_stable_window() -> None:
    rows = run_group_temporal_knn(
        _train_windows(),
        [_stable_window("stable"), _anomalous_window("broken")],
        n_neighbors=3,
    )

    scores_by_window = {}
    for row in rows:
        scores_by_window.setdefault(row["metadata"]["window_id"], row["score"])

    assert scores_by_window["broken"] > scores_by_window["stable"]
    assert scores_by_window["broken"] > 0.0


def test_group_hybrid_combines_prediction_graph_and_temporal_profile() -> None:
    rows = run_group_hybrid_fusiontrack(
        _train_windows(),
        [_stable_window("stable"), _anomalous_window("broken")],
        n_neighbors=3,
        prediction_weight=0.8,
        graph_weight=0.1,
        temporal_weight=0.1,
    )

    scores_by_window = {}
    for row in rows:
        scores_by_window.setdefault(row["metadata"]["window_id"], row["score"])

    assert scores_by_window["broken"] > scores_by_window["stable"]
    for row in rows:
        assert row["source"] == "fusiontrack_group_hybrid"
        assert set(row["component_scores"]) >= {
            "prediction_residual_rank",
            "graph_rank",
            "temporal_profile_rank",
        }
        assert row["metadata"]["method"] == "fusiontrack_group_hybrid"


def test_residual_gate_suppresses_side_components_when_residual_is_low() -> None:
    ungated = _residual_gated_rank_fusion(
        prediction_rank=[0.0, 1.0],
        graph_rank=[1.0, 0.0],
        temporal_rank=[1.0, 0.0],
        weights=(0.50, 0.25, 0.25),
        enabled=False,
        gate_power=2.0,
        gate_floor=0.0,
    )
    gated = _residual_gated_rank_fusion(
        prediction_rank=[0.0, 1.0],
        graph_rank=[1.0, 0.0],
        temporal_rank=[1.0, 0.0],
        weights=(0.50, 0.25, 0.25),
        enabled=True,
        gate_power=2.0,
        gate_floor=0.0,
    )

    assert gated[0] < ungated[0]
    assert gated[1] > gated[0]
    assert all(0.0 <= score <= 1.0 for score in gated)


def test_group_hybrid_records_residual_gate_config() -> None:
    rows = run_group_hybrid_fusiontrack(
        _train_windows(),
        [_stable_window("stable"), _anomalous_window("broken")],
        n_neighbors=3,
        prediction_weight=0.8,
        graph_weight=0.1,
        temporal_weight=0.1,
        use_residual_gate=True,
        residual_gate_power=2.0,
        residual_gate_floor=0.05,
    )

    for row in rows:
        assert row["metadata"]["residual_gate"] == {
            "enabled": True,
            "power": 2.0,
            "floor": 0.05,
        }


def test_group_hybrid_exposes_event_components_and_score_sources() -> None:
    rows = run_group_hybrid_fusiontrack(
        _train_windows(),
        [_stable_window("stable"), _anomalous_window("broken")],
        n_neighbors=3,
        prediction_weight=0.6,
        graph_weight=0.2,
        temporal_weight=0.2,
    )

    broken = next(row for row in rows if row["metadata"]["window_id"] == "broken")

    assert GROUP_EVENT_COMPONENT_SCORE_KEYS <= set(broken["component_scores"])
    assert broken["event_score"] == max(item["score"] for item in broken["frame_event_scores"])
    assert broken["event_segments"]
    assert all(
        {"frame", "score", "dominant_reason", "component_scores"} <= set(item)
        for item in broken["frame_event_scores"]
    )
    assert broken["metadata"]["event_component_schema_version"] == 1
    assert broken["metadata"]["event_component_keys"] == sorted(GROUP_EVENT_COMPONENT_SCORE_KEYS)
    assert set(broken["metadata"]["score_sources"]) == {
        "prediction",
        "graph",
        "temporal_profile",
    }
    assert "prediction_residual" in broken["metadata"]["score_sources"]["prediction"]["component_scores"]
    assert "temporal_profile_distance" in broken["metadata"]["score_sources"]["temporal_profile"]["component_scores"]
    assert broken["metadata"]["score_sources"]["graph"]["dominant_reason"]


def test_outputs_object_window_schema_and_preserves_sample_id() -> None:
    score_window = _stable_window("score_schema")
    score_window["objects"][0]["sample_id"] = "custom:a"

    rows = run_group_temporal_knn(_train_windows(), [score_window], n_neighbors=2)

    assert len(rows) == 3
    assert [row["sample_id"] for row in rows] == ["custom:a", "seq_score:b", "seq_score:c"]
    for row in rows:
        assert set(row) == {
            "sample_id",
            "window_id",
            "sequence",
            "track_id",
            "frame_start",
            "frame_end",
            "source",
            "score",
            "component_scores",
            "metadata",
        }
        assert row["window_id"] == "score_schema"
        assert row["sequence"] == "seq_score"
        assert row["frame_start"] == 1
        assert row["frame_end"] == 4
        assert row["source"] == "fusiontrack_group_temporal_knn"
        assert isinstance(row["score"], float)
        assert math.isfinite(row["score"])
        assert set(row["component_scores"]) >= {
            "temporal_profile_distance",
            "mean_speed",
            "mean_dispersion",
            "neighbor_churn",
        }
        assert row["metadata"]["window_id"] == "score_schema"
        assert row["metadata"]["n_neighbors"] == 2
        assert "mean_speed" in row["metadata"]["feature_columns"]


def test_neighbor_count_is_clamped_to_training_window_count() -> None:
    model = fit_group_temporal_knn([_stable_window("only_train")], n_neighbors=99)

    assert model["n_neighbors"] == 1


def test_empty_training_windows_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="Cannot fit group temporal KNN with no training windows"):
        fit_group_temporal_knn([], n_neighbors=3)
