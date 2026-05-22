from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.individual_classical import CLASSICAL_METHODS, run_classical_baseline


def _normal_trajectories() -> list[dict]:
    trajectories = []
    for track_index in range(8):
        trajectories.append(
            {
                "sample_id": f"seq_train:normal_{track_index}",
                "sequence": "seq_train",
                "track_id": f"normal_{track_index}",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [0.0, float(track_index)]}},
                    {"frame_id": 2, "fused": {"center_xy": [1.0, float(track_index)]}},
                    {"frame_id": 3, "fused": {"center_xy": [2.0, float(track_index)]}},
                    {"frame_id": 4, "fused": {"center_xy": [3.0, float(track_index)]}},
                ],
            }
        )
    return trajectories


def _score_trajectories() -> list[dict]:
    return [
        {
            "sample_id": "seq_score:normal",
            "sequence": "seq_score",
            "track_id": "normal",
            "points": [
                {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
                {"frame_id": 2, "fused": {"center_xy": [1.0, 0.0]}},
                {"frame_id": 3, "fused": {"center_xy": [2.0, 0.0]}},
                {"frame_id": 4, "fused": {"center_xy": [3.0, 0.0]}},
            ],
        },
        {
            "sample_id": "seq_score:jump",
            "sequence": "seq_score",
            "track_id": "jump",
            "points": [
                {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
                {"frame_id": 2, "fused": {"center_xy": [20.0, 0.0]}},
                {"frame_id": 3, "fused": {"center_xy": [40.0, 0.0]}},
                {"frame_id": 4, "fused": {"center_xy": [60.0, 0.0]}},
            ],
        },
    ]


@pytest.mark.parametrize("method", CLASSICAL_METHODS)
def test_run_classical_baseline_returns_finite_scores_for_each_method(method: str) -> None:
    rows = run_classical_baseline(
        _normal_trajectories(),
        _score_trajectories(),
        method=method,
        seed=11,
        contamination=0.1,
    )

    assert [row["sample_id"] for row in rows] == ["seq_score:jump", "seq_score:normal"]
    assert {row["sequence"] for row in rows} == {"seq_score"}
    assert {row["track_id"] for row in rows} == {"jump", "normal"}
    for row in rows:
        assert row["source"] == f"individual_classical:{method}"
        assert isinstance(row["score"], float)
        assert math.isfinite(row["score"])
        assert row["component_scores"] == {method: row["score"]}
        assert row["metadata"]["method"] == method


def test_run_classical_baseline_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown classical detector"):
        run_classical_baseline(
            _normal_trajectories(),
            _score_trajectories(),
            method="unknown",
        )
