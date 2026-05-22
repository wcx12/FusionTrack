from __future__ import annotations

import copy
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusiontrack.fused_trajectories import build_fused_trajectory, fuse_state


def test_fuse_state_averages_two_modal_centers_and_scores_offset() -> None:
    point = {
        "rgb": {"center_xy": [10.0, 20.0]},
        "thermal": {"center_xy": [14.0, 28.0]},
        "modal": {"modal_offset_distance": 5.0},
    }

    fused = fuse_state(point, offset_scale=25.0)

    assert fused is not None
    assert fused["center_xy"] == [12.0, 24.0]
    assert fused["confidence"] == 1.0 / (1.0 + 5.0 / 25.0)
    assert fused["source_modalities"] == ["rgb", "thermal"]
    assert fused["component_scores"]["modal_offset_distance"] == 5.0


def test_fuse_state_uses_single_modality_fallback() -> None:
    fused = fuse_state({"thermal": {"center_xy": [7.0, 9.0]}})

    assert fused is not None
    assert fused["center_xy"] == [7.0, 9.0]
    assert fused["confidence"] == 0.55
    assert fused["source_modalities"] == ["thermal"]


def test_fuse_state_lower_confidence_for_larger_offsets() -> None:
    near = fuse_state(
        {
            "rgb": {"center_xy": [0.0, 0.0]},
            "thermal": {"center_xy": [3.0, 4.0]},
        }
    )
    far = fuse_state(
        {
            "rgb": {"center_xy": [0.0, 0.0]},
            "thermal": {"center_xy": [30.0, 40.0]},
        }
    )

    assert near is not None
    assert far is not None
    assert near["confidence"] > far["confidence"]
    assert near["component_scores"]["modal_offset_distance"] == 5.0
    assert far["component_scores"]["modal_offset_distance"] == 50.0


def test_build_fused_trajectory_preserves_input_and_marks_missing_fused_state() -> None:
    trajectory = {
        "sample_id": "seq_a:track_1",
        "sequence": "seq_a",
        "track_id": "track_1",
        "category_id": 2,
        "category_name": "person",
        "points": [
            {
                "frame_id": 1,
                "rgb": {"center_xy": [0.0, 0.0]},
                "thermal": {"center_xy": [2.0, 2.0]},
            },
            {"frame_id": 2, "rgb": {}, "thermal": {}},
        ],
    }
    original = copy.deepcopy(trajectory)

    fused_trajectory = build_fused_trajectory(trajectory)

    assert trajectory == original
    assert fused_trajectory is not trajectory
    assert fused_trajectory["sample_id"] == "seq_a:track_1"
    assert fused_trajectory["category_name"] == "person"
    assert fused_trajectory["points"][0]["rgb"] == {"center_xy": [0.0, 0.0]}
    assert fused_trajectory["points"][0]["thermal"] == {"center_xy": [2.0, 2.0]}
    assert fused_trajectory["points"][0]["fused"]["center_xy"] == [1.0, 1.0]
    assert math.isclose(
        fused_trajectory["points"][0]["fused"]["component_scores"]["rgb_weight"],
        0.5,
    )
    assert fused_trajectory["points"][1]["frame_id"] == 2
    assert fused_trajectory["points"][1]["fused"] is None
