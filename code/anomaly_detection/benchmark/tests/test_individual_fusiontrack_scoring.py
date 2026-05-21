from __future__ import annotations

from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusiontrack.individual_scoring import run_individual_fusiontrack_baseline


def _trajectory(track_id: str, centers: list[list[float]]) -> dict:
    return {
        "sample_id": f"seq_a:{track_id}",
        "sequence": "seq_a",
        "track_id": track_id,
        "points": [
            {"frame_id": index + 1, "fused": {"center_xy": center}}
            for index, center in enumerate(centers)
        ],
    }


def test_individual_fusiontrack_nearest_feature_scores_rank_jump_higher() -> None:
    train = [
        _trajectory("train_a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        _trajectory("train_b", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
    ]
    normal = _trajectory("normal", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    jump = _trajectory("jump", [[0.0, 0.0], [1.0, 0.0], [20.0, 0.0]])

    rows = run_individual_fusiontrack_baseline(train, [normal, jump], n_neighbors=1)
    scores = {row["track_id"]: row["score"] for row in rows}

    assert scores["jump"] > scores["normal"]
    for row in rows:
        assert row["source"] == "fusiontrack_individual:nearest_feature"
        assert isinstance(row["score"], float)
        assert math.isfinite(row["score"])
        assert row["metadata"]["n_neighbors"] == 1
