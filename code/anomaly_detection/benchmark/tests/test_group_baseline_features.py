from __future__ import annotations

from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.group_features import build_group_feature_row, build_group_feature_table


def _window() -> dict:
    return {
        "window_id": "w1",
        "sample_id": "window_sample",
        "sequence": "seq_a",
        "frame_start": 10,
        "frame_end": 12,
        "objects": [
            {
                "track_id": "a",
                "states": [
                    {"frame_id": 10, "fused": {"center_xy": [0.0, 0.0]}},
                    {"frame_id": 11, "fused": {"center_xy": [1.0, 0.0]}},
                    {"frame_id": 12, "fused": {"center_xy": [2.0, 0.0]}},
                ],
            },
            {
                "track_id": "b",
                "states": [
                    {"frame_id": 10, "rgb": {"center_xy": [0.0, 2.0]}},
                    {"frame_id": 11, "rgb": {"center_xy": [1.0, 2.0]}},
                    {"frame_id": 12, "rgb": {"center_xy": [2.0, 2.0]}},
                ],
            },
            {
                "track_id": "missing_center",
                "states": [{"frame_id": 10, "fused": {}}],
            },
        ],
    }


def test_build_group_feature_row_summarizes_window_dynamics_and_ignores_missing_centers() -> None:
    row = build_group_feature_row(_window())

    assert row["window_id"] == "w1"
    assert row["sequence"] == "seq_a"
    assert row["frame_start"] == 10
    assert row["frame_end"] == 12
    assert row["num_objects"] == 2
    assert row["num_frames"] == 3
    assert row["mean_group_size"] == 2.0
    assert row["max_group_size"] == 2.0
    assert row["mean_speed"] == 1.0
    assert row["std_speed"] == 0.0
    assert row["mean_dispersion"] > 0.0
    assert row["max_dispersion"] >= row["mean_dispersion"]
    assert row["neighbor_churn"] == 0.0
    for value in row.values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_build_group_feature_table_returns_stable_numeric_feature_columns() -> None:
    table = build_group_feature_table([_window(), {**_window(), "window_id": "w2"}])

    assert table["window_id"].tolist() == ["w1", "w2"]
    assert table["sequence"].tolist() == ["seq_a", "seq_a"]
    assert table["num_objects"].tolist() == [2, 2]
    assert table["mean_speed"].tolist() == [1.0, 1.0]
