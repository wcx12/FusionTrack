from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.group_temporal_autoencoder import (
    FEATURE_COLUMNS,
    build_temporal_graph_signature_table,
    fit_temporal_graph_autoencoder,
    run_temporal_graph_autoencoder,
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
            _object("a", [(0.0, 0.0), (1.0, 0.0), (2.0, 8.0), (3.0, 16.0)]),
            _object("b", [(0.0, 1.0), (1.0, 1.0), (18.0, 1.0), (28.0, 1.0)]),
            _object("c", [(0.0, 2.0), (1.0, 2.0), (-15.0, 2.0), (-25.0, 2.0)]),
        ],
    }


def _train_windows() -> list[dict]:
    return [_stable_window(f"train_{index}", sequence="seq_train") for index in range(5)]


def test_anomalous_group_window_scores_higher_than_stable_window() -> None:
    rows = run_temporal_graph_autoencoder(
        _train_windows(),
        [_stable_window("stable"), _anomalous_window("broken")],
        n_components=3,
    )

    scores_by_window = {}
    for row in rows:
        scores_by_window.setdefault(row["metadata"]["window_id"], row["score"])

    assert scores_by_window["broken"] > scores_by_window["stable"]
    assert scores_by_window["broken"] > 0.0


def test_outputs_object_schema_and_preserves_sample_id() -> None:
    score_window = _stable_window("score_schema")
    score_window["objects"][0]["sample_id"] = "custom:a"

    rows = run_temporal_graph_autoencoder(_train_windows(), [score_window], n_components=2)

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
        assert row["source"] == "group_temporal_autoencoder:pca_reconstruction"
        assert isinstance(row["score"], float)
        assert math.isfinite(row["score"])
        assert set(row["component_scores"]) >= {
            "reconstruction_error",
            "temporal_churn",
            "dispersion",
        }
        assert row["metadata"]["window_id"] == "score_schema"
        assert row["metadata"]["n_components"] == 2
        assert "num_objects" in row["metadata"]["feature_columns"]
        assert "edge_density" in row["metadata"]["feature_columns"]


def test_single_training_window_clamps_n_components() -> None:
    train_windows = [_stable_window("only_train")]
    model = fit_temporal_graph_autoencoder(train_windows, n_components=99)
    expected_components = min(
        99,
        len(build_temporal_graph_signature_table(train_windows)),
        len(FEATURE_COLUMNS),
    )

    assert model["n_components"] == expected_components

    rows = run_temporal_graph_autoencoder(
        train_windows,
        [_stable_window("score")],
        n_components=99,
    )
    assert len(rows) == 3
    assert {row["metadata"]["n_components"] for row in rows} == {expected_components}


def test_empty_training_windows_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="Cannot fit temporal graph autoencoder with no training windows"):
        fit_temporal_graph_autoencoder([], n_components=3)
