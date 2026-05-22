from pathlib import Path
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.physics_informed import (
    build_physics_feature_row,
    fit_physics_profile,
    run_physics_informed_baseline,
)


def _trajectory(sample_id: str, xs: list[float], ys: list[float] | None = None) -> dict:
    if ys is None:
        ys = [0.0] * len(xs)
    return {
        "sample_id": sample_id,
        "sequence": "seq_a",
        "track_id": sample_id.split(":")[-1],
        "points": [
            {"frame_id": frame_id, "fused": {"center_xy": [x, y]}}
            for frame_id, (x, y) in enumerate(zip(xs, ys), start=1)
        ],
    }


def test_physics_profile_scores_speed_spikes_above_smooth_trajectories() -> None:
    train = [
        _trajectory("seq_a:train_1", [0, 1, 2, 3, 4, 5]),
        _trajectory("seq_a:train_2", [1, 2, 3, 4, 5, 6]),
        _trajectory("seq_a:train_3", [0, 1, 2, 3, 4, 5]),
    ]
    normal = _trajectory("seq_a:normal", [0, 1, 2, 3, 4, 5])
    spike = _trajectory("seq_a:spike", [0, 1, 2, 12, 13, 14])

    results = run_physics_informed_baseline(train, [normal, spike])
    by_id = {result["sample_id"]: result for result in results}

    assert by_id["seq_a:spike"]["score"] > by_id["seq_a:normal"]["score"]
    assert by_id["seq_a:spike"]["component_scores"]["speed"] > by_id["seq_a:normal"]["component_scores"]["speed"]
    assert by_id["seq_a:spike"]["component_scores"]["acceleration"] > by_id["seq_a:normal"]["component_scores"]["acceleration"]


def test_physics_profile_output_schema_preserves_ids_and_finite_scores() -> None:
    train = [_trajectory("seq_a:train", [0, 1, 2, 3, 4])]
    score = _trajectory("seq_a:track_7", [0, 1, 3, 6, 10])

    result = run_physics_informed_baseline(train, [score])[0]

    assert result["sample_id"] == "seq_a:track_7"
    assert result["sequence"] == "seq_a"
    assert result["track_id"] == "track_7"
    assert result["source"] == "individual_physics:kinematic_prior"
    assert math.isfinite(result["score"])
    assert set(result["component_scores"]) == {
        "speed",
        "acceleration",
        "jerk",
        "turn",
        "smoothness",
    }
    assert all(math.isfinite(value) for value in result["component_scores"].values())
    assert result["metadata"]["profile"] == "median_mad"
    assert "mean_speed" in result["metadata"]["feature_columns"]


def test_physics_profile_preserves_score_input_order() -> None:
    train = [_trajectory("seq_a:train", [0, 1, 2, 3, 4])]
    score = [
        _trajectory("seq_a:z_track", [0, 1, 2, 3, 4]),
        _trajectory("seq_a:a_track", [0, 2, 4, 6, 8]),
    ]

    results = run_physics_informed_baseline(train, score)

    assert [row["sample_id"] for row in results] == ["seq_a:z_track", "seq_a:a_track"]


def test_fit_physics_profile_rejects_empty_training_set() -> None:
    with pytest.raises(ValueError, match="train_trajectories"):
        fit_physics_profile([])


def test_single_point_trajectory_uses_conservative_finite_defaults() -> None:
    single = _trajectory("seq_a:single", [5])

    row = build_physics_feature_row(single)
    result = run_physics_informed_baseline([single], [single])[0]

    numeric_values = [
        value
        for key, value in row.items()
        if key not in {"sample_id", "sequence", "track_id"}
    ]
    assert np.isfinite(np.asarray(numeric_values, dtype=float)).all()
    assert row["mean_speed"] == 0.0
    assert row["max_acceleration"] == 0.0
    assert math.isfinite(result["score"])


def test_zero_mad_profile_uses_conservative_scale_floor() -> None:
    train = [
        _trajectory("seq_a:train_1", [0, 1, 2, 3, 4]),
        _trajectory("seq_a:train_2", [0, 1, 2, 3, 4]),
    ]
    tiny_change = _trajectory("seq_a:tiny_change", [0, 1.001, 2.002, 3.003, 4.004])

    result = run_physics_informed_baseline(train, [tiny_change])[0]

    assert result["component_scores"]["speed"] < 5.0
    assert result["score"] < 5.0
