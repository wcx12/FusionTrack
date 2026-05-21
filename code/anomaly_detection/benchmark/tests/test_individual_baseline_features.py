from pathlib import Path
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.individual_features import (
    build_handcrafted_feature_row,
    build_handcrafted_feature_table,
    extract_center_sequence,
)


def _trajectory() -> dict:
    return {
        "sample_id": "seq_a:track_1",
        "sequence": "seq_a",
        "track_id": "track_1",
        "points": [
            {
                "frame_id": 3,
                "fused": {
                    "center_xy": [6.0, 4.0],
                    "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                },
            },
            {
                "frame_id": 1,
                "fused": {
                    "center_xy": [0.0, 0.0],
                    "bbox_xyxy": [0.0, 0.0, 2.0, 3.0],
                },
                "rgb": {"center_xy": [1.0, 0.0]},
                "thermal": {"center_xy": [0.0, 0.0]},
            },
            {
                "frame_id": 2,
                "rgb": {
                    "center_xy": [3.0, 4.0],
                    "bbox_xyxy": [1.0, 1.0, 4.0, 5.0],
                },
                "thermal": {"center_xy": [6.0, 8.0]},
            },
        ],
    }


def test_extract_center_sequence_uses_preferred_modalities_and_frame_order() -> None:
    sequence = extract_center_sequence(_trajectory())

    assert sequence == [(1, 0.0, 0.0), (2, 3.0, 4.0), (3, 6.0, 4.0)]


def test_build_handcrafted_feature_row_computes_motion_and_modal_features() -> None:
    row = build_handcrafted_feature_row(_trajectory())

    assert row["sample_id"] == "seq_a:track_1"
    assert row["sequence"] == "seq_a"
    assert row["track_id"] == "track_1"
    assert row["duration_frames"] == 3
    assert row["num_points"] == 3
    assert row["path_length"] == 8.0
    assert row["displacement"] == math.sqrt(52.0)
    assert row["mean_speed"] == 4.0
    assert row["max_speed"] == 5.0
    assert row["std_speed"] == 1.0
    assert row["mean_acceleration"] == 2.0
    assert row["max_acceleration"] == 2.0
    assert row["mean_turn_angle"] == math.acos(0.6)
    assert row["max_turn_angle"] == math.acos(0.6)
    assert row["bbox_area_mean"] == np.mean([6.0, 12.0, 1.0])
    assert row["bbox_area_std"] == np.std([6.0, 12.0, 1.0])
    assert row["modal_offset_mean"] == np.mean([1.0, 0.0, 1.0, 5.0])
    assert row["modal_offset_max"] == 5.0


def test_build_handcrafted_feature_table_sorts_by_sample_id_and_uses_numeric_defaults() -> None:
    trajectories = [
        {
            "sample_id": "seq_b:track_2",
            "sequence": "seq_b",
            "track_id": "track_2",
            "points": [{"frame_id": 5, "fused": {"center_xy": [2.0, 2.0]}}],
        },
        {
            "sample_id": "seq_a:track_missing",
            "sequence": "seq_a",
            "track_id": "track_missing",
            "points": [{"frame_id": 1}],
        },
    ]

    table = build_handcrafted_feature_table(trajectories)

    assert table["sample_id"].tolist() == ["seq_a:track_missing", "seq_b:track_2"]
    missing = table.iloc[0].to_dict()
    assert missing["duration_frames"] == 0
    assert missing["num_points"] == 0
    assert missing["path_length"] == 0.0
    assert missing["bbox_area_mean"] == 0.0
    assert missing["modal_offset_mean"] == 0.0
    numeric = table.drop(columns=["sample_id", "sequence", "track_id"])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()
