from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baselines.group_classical as group_classical
from baselines.group_classical import CLASSICAL_METHODS, run_classical_baseline


def _object(track_id: str, y: float, jump: float = 0.0) -> dict:
    return {
        "track_id": track_id,
        "states": [
            {"frame_id": 1, "fused": {"center_xy": [0.0, y]}},
            {"frame_id": 2, "fused": {"center_xy": [1.0 + jump, y]}},
            {"frame_id": 3, "fused": {"center_xy": [2.0 + jump, y]}},
        ],
    }


def _window(window_id: str, sequence: str = "seq_score", jump: float = 0.0) -> dict:
    return {
        "window_id": window_id,
        "sequence": sequence,
        "frame_start": 1,
        "frame_end": 3,
        "objects": [
            _object("a", 0.0, jump=jump),
            _object("b", 1.0, jump=jump),
        ],
    }


def _train_windows() -> list[dict]:
    return [_window(f"train_{index}", sequence="seq_train") for index in range(8)]


@pytest.mark.parametrize("method", CLASSICAL_METHODS)
def test_run_classical_baseline_returns_object_level_rows_with_window_scores(method: str) -> None:
    rows = run_classical_baseline(
        _train_windows(),
        [_window("score_normal"), _window("score_jump", jump=10.0)],
        method=method,
        seed=7,
        contamination=0.1,
    )

    assert len(rows) == 4
    assert [row["sample_id"] for row in rows] == [
        "seq_score:a",
        "seq_score:b",
        "seq_score:a",
        "seq_score:b",
    ]
    assert {row["frame_start"] for row in rows} == {1}
    assert {row["frame_end"] for row in rows} == {3}
    assert {row["source"] for row in rows} == {f"group_classical:{method}"}
    for row in rows:
        assert row["sequence"] == "seq_score"
        assert row["window_id"] in {"score_normal", "score_jump"}
        assert row["track_id"] in {"a", "b"}
        assert isinstance(row["score"], float)
        assert math.isfinite(row["score"])
        assert row["component_scores"] == {method: row["score"]}
        assert row["metadata"]["method"] == method
        assert row["metadata"]["window_id"] in {"score_normal", "score_jump"}


def test_run_classical_baseline_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown classical detector"):
        run_classical_baseline(_train_windows(), [_window("score")], method="unknown")


def test_lof_requires_at_least_two_training_windows() -> None:
    with pytest.raises(ValueError, match="at least 2 training feature rows"):
        run_classical_baseline([_window("only_train")], [_window("score")], method="lof")


def test_run_classical_baseline_binds_scores_by_window_order_when_ids_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _window("duplicate")
    second = _window("duplicate", jump=10.0)
    first.pop("window_id")
    second.pop("window_id")

    monkeypatch.setattr(
        group_classical,
        "fit_classical_detector",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        group_classical,
        "score_classical_detector",
        lambda *args, **kwargs: [0.1, 0.9],
    )

    rows = group_classical.run_classical_baseline(
        _train_windows(),
        [first, second],
        method="isolation_forest",
    )

    assert [row["score"] for row in rows] == [0.1, 0.1, 0.9, 0.9]


def test_run_classical_baseline_keeps_scores_aligned_with_unsorted_window_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal = _window("z_normal")
    jump = _window("a_jump", jump=10.0)

    monkeypatch.setattr(
        group_classical,
        "fit_classical_detector",
        lambda *args, **kwargs: object(),
    )

    def score_by_window_id(*args, **kwargs) -> list[float]:
        feature_df = args[1]
        return [
            0.9 if str(window_id) == "a_jump" else 0.1
            for window_id in feature_df["window_id"].tolist()
        ]

    monkeypatch.setattr(group_classical, "score_classical_detector", score_by_window_id)

    rows = group_classical.run_classical_baseline(
        _train_windows(),
        [normal, jump],
        method="isolation_forest",
    )

    assert [row["metadata"]["window_id"] for row in rows] == [
        "z_normal",
        "z_normal",
        "a_jump",
        "a_jump",
    ]
    assert [row["score"] for row in rows] == [0.1, 0.1, 0.9, 0.9]
