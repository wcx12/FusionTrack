from __future__ import annotations

import pytest

from mtf_ba.observation_standardization import (
    standardize_observation_row,
    standardize_observation_rows,
)


def test_standardize_observation_row_keeps_modalities_and_quality_flags() -> None:
    row = {
        "dataset": "M3OT",
        "sequence": "SEQ-A",
        "track_id": "7",
        "category_id": "1",
        "category_name": "vehicle",
        "fps": "25",
        "frame_id": "12",
        "rgb_file": "SEQ-A/RGB/000012.jpg",
        "rgb_x": "10",
        "rgb_y": "20",
        "rgb_w": "30",
        "rgb_h": "40",
        "rgb_cx": "25",
        "rgb_cy": "40",
        "rgb_confidence": "0.9",
        "rgb_visibility": "0.8",
        "rgb_vx_px_per_frame": "2.5",
        "rgb_vy_px_per_frame": "-1",
        "rgb_speed_px_per_frame": "2.692582",
        "thermal_file": "SEQ-A/IR/000012.jpg",
        "thermal_x": "13",
        "thermal_y": "24",
        "thermal_w": "30",
        "thermal_h": "40",
        "thermal_cx": "28",
        "thermal_cy": "44",
        "thermal_confidence": "0.7",
        "thermal_visibility": "0.6",
        "modal_offset_dx_thermal_minus_rgb": "3",
        "modal_offset_dy_thermal_minus_rgb": "4",
        "modal_offset_distance": "5",
        "modal_bbox_iou": "0.72",
    }

    normalized = standardize_observation_row(row)

    assert normalized["sample_id"] == "SEQ-A:7"
    assert normalized["frame_id"] == 12
    assert normalized["fps"] == 25.0
    assert normalized["modalities"]["rgb"]["available"] is True
    assert normalized["modalities"]["rgb"]["bbox_xywh"] == [10.0, 20.0, 30.0, 40.0]
    assert normalized["modalities"]["rgb"]["center_xy"] == [25.0, 40.0]
    assert normalized["modalities"]["thermal"]["available"] is True
    assert normalized["modalities"]["thermal"]["center_xy"] == [28.0, 44.0]
    assert normalized["modal_relation"]["available"] is True
    assert normalized["modal_relation"]["offset_distance"] == 5.0
    assert normalized["quality"] == {
        "num_available_modalities": 2,
        "available_modalities": ["rgb", "thermal"],
        "missing_modalities": [],
        "has_cross_modal_relation": True,
    }


def test_standardize_observation_row_marks_missing_modality_without_dropping_row() -> None:
    normalized = standardize_observation_row(
        {
            "sequence": "SEQ-B",
            "track_id": "3",
            "frame_id": "2",
            "rgb_cx": "11",
            "rgb_cy": "22",
        }
    )

    assert normalized["modalities"]["rgb"]["available"] is True
    assert normalized["modalities"]["thermal"]["available"] is False
    assert normalized["modalities"]["thermal"]["center_xy"] is None
    assert normalized["modal_relation"]["available"] is False
    assert normalized["quality"]["num_available_modalities"] == 1
    assert normalized["quality"]["missing_modalities"] == ["thermal"]


def test_standardize_observation_rows_sorts_by_sequence_track_and_frame() -> None:
    rows = standardize_observation_rows(
        [
            {"sequence": "SEQ-B", "track_id": "10", "frame_id": "3", "rgb_cx": "1", "rgb_cy": "1"},
            {"sequence": "SEQ-A", "track_id": "2", "frame_id": "9", "rgb_cx": "1", "rgb_cy": "1"},
            {"sequence": "SEQ-A", "track_id": "2", "frame_id": "1", "rgb_cx": "1", "rgb_cy": "1"},
        ]
    )

    assert [(row["sequence"], row["track_id"], row["frame_id"]) for row in rows] == [
        ("SEQ-A", "2", 1),
        ("SEQ-A", "2", 9),
        ("SEQ-B", "10", 3),
    ]


def test_standardize_observation_row_reports_bad_numeric_fields() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        standardize_observation_row(
            {
                "sequence": "SEQ-C",
                "track_id": "1",
                "frame_id": "bad",
                "rgb_cx": "1",
                "rgb_cy": "1",
            }
        )
