from __future__ import annotations

from pathlib import Path
import math
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.group_prediction import run_prediction_baseline


def _object(track_id: str, centers: list[list[float]]) -> dict:
    return {
        "track_id": track_id,
        "states": [
            {"frame_id": index + 1, "fused": {"center_xy": center}}
            for index, center in enumerate(centers)
        ],
    }


def _object_with_sample_id(track_id: str, sample_id: str) -> dict:
    obj = _object(track_id, [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    obj["sample_id"] = sample_id
    return obj


def _window() -> dict:
    return {
        "window_id": "w1",
        "sequence": "seq_a",
        "objects": [
            _object("steady", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            _object("jump", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [20.0, 1.0]]),
            {"track_id": "missing_center", "states": [{"frame_id": 1, "fused": {}}]},
        ],
    }


def test_prediction_baseline_outputs_object_rows_and_ranks_jump_higher() -> None:
    rows = run_prediction_baseline([_window()])
    scores = {row["track_id"]: row["score"] for row in rows}

    assert [row["sample_id"] for row in rows] == [
        "seq_a:jump",
        "seq_a:missing_center",
        "seq_a:steady",
    ]
    assert scores["jump"] > scores["steady"]
    assert scores["jump"] > 0.0
    assert scores["steady"] == 0.0
    assert scores["missing_center"] == 0.0
    for row in rows:
        assert row["window_id"] == "w1"
        assert row["sequence"] == "seq_a"
        assert row["frame_start"] == 1
        assert row["frame_end"] == 4
        assert row["source"] == "group_prediction:linear"
        assert isinstance(row["score"], float)
        assert math.isfinite(row["score"])
        assert row["component_scores"] == {"prediction_residual": row["score"]}


def test_run_group_baselines_help_works_as_direct_script() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_group_baselines.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--baseline" in result.stdout


def test_prediction_baseline_preserves_object_sample_id_and_rejects_duplicates() -> None:
    rows = run_prediction_baseline(
        [
            {
                "window_id": "w_custom",
                "sequence": "seq_a",
                "objects": [_object_with_sample_id("a", "custom:a")],
            }
        ]
    )

    assert rows[0]["sample_id"] == "custom:a"

    with pytest.raises(ValueError, match="Duplicate track_id"):
        run_prediction_baseline(
            [
                {
                    "window_id": "w_duplicate",
                    "sequence": "seq_a",
                    "objects": [
                        _object("a", [[0, 0], [1, 0], [2, 0]]),
                        _object("a", [[0, 1], [1, 1], [2, 1]]),
                    ],
                }
            ]
        )
