from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.trajectory_language_model import (
    fit_ngram_language_model,
    run_ngram_language_model,
    score_ngram_language_model,
    trajectory_to_tokens,
)


def _trajectory(sample_id: str, centers: list[list[float]]) -> dict:
    sequence, track_id = sample_id.split(":", 1)
    return {
        "sample_id": sample_id,
        "sequence": sequence,
        "track_id": track_id,
        "points": [
            {"frame_id": index + 1, "fused": {"center_xy": center}}
            for index, center in enumerate(centers)
        ],
    }


def test_trajectory_to_tokens_separates_straight_turn_and_jump() -> None:
    straight = trajectory_to_tokens(_trajectory("seq:straight", [[0, 0], [1, 0], [2, 0]]))
    turn = trajectory_to_tokens(_trajectory("seq:turn", [[0, 0], [1, 0], [1, 2]]))
    jump = trajectory_to_tokens(_trajectory("seq:jump", [[0, 0], [1, 0], [7, 0]]))

    assert "dir:E" in straight
    assert "dir:N" in turn
    assert any(token.startswith("turn:") and token != "turn:0" for token in turn)
    assert any(token.startswith("spd:") for token in jump)
    assert jump != straight
    assert turn != straight


def test_trajectory_to_tokens_places_turn_between_local_motion_steps() -> None:
    tokens = trajectory_to_tokens(_trajectory("seq:turn", [[0, 0], [1, 0], [1, 2]]))

    assert tokens == ["dir:E", "spd:1", "turn:2", "dir:N", "spd:2"]


def test_ngram_language_model_scores_jump_and_turn_above_normal() -> None:
    train = [
        _trajectory("train:a", [[0, 0], [1, 0], [2, 0], [3, 0]]),
        _trajectory("train:b", [[4, 1], [5, 1], [6, 1], [7, 1]]),
    ]
    normal = _trajectory("score:normal", [[10, 0], [11, 0], [12, 0], [13, 0]])
    jump = _trajectory("score:jump", [[10, 0], [11, 0], [20, 0], [21, 0]])
    turn = _trajectory("score:turn", [[10, 0], [11, 0], [11, 2], [11, 4]])

    rows = run_ngram_language_model(train, [normal, jump, turn], ngram_order=2, alpha=0.1)
    scores = {row["sample_id"]: row["score"] for row in rows}

    assert scores["score:jump"] > scores["score:normal"]
    assert scores["score:turn"] > scores["score:normal"]


def test_ngram_language_model_output_schema_preserves_order_and_scores_are_finite() -> None:
    train = [_trajectory("train:a", [[0, 0], [1, 0], [2, 0]])]
    score = [
        _trajectory("score:z", [[0, 0]]),
        _trajectory("score:a", [[0, 0], [1, 0], [2, 0]]),
    ]

    rows = run_ngram_language_model(
        train,
        score,
        ngram_order=2,
        alpha=1.0,
        grid_size=16,
        seed=123,
    )

    assert [row["sample_id"] for row in rows] == ["score:z", "score:a"]
    for row in rows:
        assert row["sequence"] == "score"
        assert row["source"] == "individual_trajectory_lm:ngram"
        assert math.isfinite(row["score"])
        assert row["component_scores"]["negative_log_likelihood"] == row["score"]
        assert row["component_scores"]["num_tokens"] >= 0
        assert row["metadata"]["ngram_order"] == 2
        assert row["metadata"]["alpha"] == 1.0
        assert row["metadata"]["grid_size"] == 16
        assert row["metadata"]["seed"] == 123
        assert row["metadata"]["vocab_size"] >= 1


def test_ngram_language_model_smoothing_scores_unseen_token_and_context() -> None:
    model = fit_ngram_language_model(
        [_trajectory("train:a", [[0, 0], [1, 0], [2, 0]])],
        ngram_order=3,
        alpha=0.5,
    )
    rows = score_ngram_language_model(
        model,
        [_trajectory("score:unseen", [[0, 0], [0, 3], [4, 3], [4, -1]])],
    )

    assert len(rows) == 1
    assert math.isfinite(rows[0]["score"])
    assert rows[0]["score"] > 0.0
