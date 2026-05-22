from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.complementary_trajectory import run_complementary_baseline


def _trajectory(
    sample_id: str,
    centers: list[list[float]],
    *,
    include_modal: bool = True,
) -> dict:
    points = []
    for frame_id, center in enumerate(centers, start=1):
        point = {
            "frame_id": frame_id,
            "fused": {
                "center_xy": center,
                "bbox_xyxy": [center[0], center[1], center[0] + 2.0, center[1] + 3.0],
            },
        }
        if include_modal:
            point["rgb"] = {
                "center_xy": [center[0] + 0.1, center[1]],
                "bbox_xyxy": [center[0], center[1], center[0] + 2.0, center[1] + 3.0],
            }
        points.append(point)
    sequence, track_id = sample_id.split(":", 1)
    return {
        "sample_id": sample_id,
        "sequence": sequence,
        "track_id": track_id,
        "points": points,
    }


def _normal_train() -> list[dict]:
    return [
        _trajectory(f"train:normal_{index}", [[0.0, y], [1.0, y], [2.0, y], [3.0, y]])
        for index, y in enumerate([0.0, 1.0, 2.0, 3.0, 4.0])
    ]


def test_complementary_baseline_outputs_schema_and_component_scores() -> None:
    rows = run_complementary_baseline(
        _normal_train(),
        [
            _trajectory("score:normal", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            _trajectory("score:jump", [[0.0, 0.0], [12.0, 0.0], [24.0, 0.0], [36.0, 0.0]]),
        ],
        seed=7,
        contamination=0.1,
        n_neighbors=1,
    )

    assert [row["sample_id"] for row in rows] == ["score:jump", "score:normal"]
    for row in rows:
        assert row["source"] == "individual_complementary:cetrajad_style"
        assert isinstance(row["score"], float)
        assert math.isfinite(row["score"])
        assert set(row["component_scores"]) == {"route", "speed", "shape", "modal"}
        assert row["metadata"]["seed"] == 7
        assert row["metadata"]["contamination"] == 0.1
        assert row["metadata"]["n_neighbors"] == 1
        assert set(row["metadata"]["feature_columns"]) == {"route", "speed", "shape", "modal"}


def test_complementary_baseline_ranks_speed_spike_above_normal() -> None:
    rows = run_complementary_baseline(
        _normal_train(),
        [
            _trajectory("score:normal", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            _trajectory("score:jump", [[0.0, 0.0], [1.0, 0.0], [40.0, 0.0], [41.0, 0.0]]),
        ],
        seed=11,
        contamination=0.1,
        n_neighbors=1,
    )

    scores = {row["sample_id"]: row["score"] for row in rows}
    assert scores["score:jump"] > scores["score:normal"]


def test_complementary_baseline_handles_missing_modal_features() -> None:
    rows = run_complementary_baseline(
        _normal_train(),
        [
            _trajectory(
                "score:normal",
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                include_modal=False,
            )
        ],
        seed=3,
        contamination=0.1,
        n_neighbors=1,
    )

    assert rows[0]["component_scores"]["modal"] == 0.0
    assert math.isfinite(rows[0]["score"])


def test_complementary_baseline_rejects_too_few_training_rows_with_clear_error() -> None:
    with pytest.raises(ValueError, match="shape detector requires at least 2 training rows"):
        run_complementary_baseline(
            [_trajectory("train:only", [[0.0, 0.0], [1.0, 0.0]])],
            [_trajectory("score:normal", [[0.0, 0.0], [1.0, 0.0]])],
            seed=3,
            contamination=0.1,
            n_neighbors=1,
        )
