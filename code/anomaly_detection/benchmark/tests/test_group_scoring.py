from __future__ import annotations

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusiontrack.group_scoring import score_group_windows


EXPECTED_COMPONENTS = {
    "leave",
    "motion",
    "neighbor",
    "count",
    "dispersion",
    "split_merge",
    "object_group",
    "group_event",
}


def _object(track_id: str, centers: list[list[float]]) -> dict:
    return {
        "sample_id": f"seq_a:{track_id}",
        "sequence": "seq_a",
        "track_id": track_id,
        "category_name": "person",
        "states": [
            {"frame_id": index + 1, "fused": {"center_xy": center}}
            for index, center in enumerate(centers)
        ],
    }


def _normal_window() -> dict:
    return {
        "sample_id": "normal_window",
        "sequence": "seq_a",
        "objects": [
            _object("a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            _object("b", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
            _object("c", [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]),
        ],
    }


def _anomaly_window() -> dict:
    return {
        "sample_id": "anomaly_window",
        "sequence": "seq_a",
        "objects": [
            _object("a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            _object("b", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
            _object("c", [[0.0, 2.0], [1.0, 2.0], [-6.0, 8.0]]),
        ],
    }


def test_score_group_windows_outputs_object_rows_with_complete_schema() -> None:
    rows = score_group_windows([_normal_window()], k_neighbors=2)

    assert len(rows) == 3
    row = rows[0]
    assert set(row) == {
        "sample_id",
        "window_id",
        "sequence",
        "track_id",
        "frame_start",
        "frame_end",
        "source",
        "score",
        "component_scores",
        "metadata",
    }
    assert row["window_id"] == "normal_window"
    assert row["frame_start"] == 1
    assert row["frame_end"] == 3
    assert row["source"] == "fusiontrack_group_graph"
    assert EXPECTED_COMPONENTS <= set(row["component_scores"])
    assert row["metadata"]["num_frames"] == 3
    assert "dominant_reason" in row["metadata"]


def test_score_group_windows_ranks_leaving_or_against_motion_object_higher() -> None:
    rows = score_group_windows([_anomaly_window()], k_neighbors=2)
    scores = {row["track_id"]: row["score"] for row in rows}

    assert scores["c"] > scores["a"]
    assert scores["c"] > scores["b"]
    anomalous = next(row for row in rows if row["track_id"] == "c")
    assert anomalous["component_scores"]["leave"] > 0.0
    assert anomalous["component_scores"]["motion"] > 0.0


def test_score_group_windows_ignores_objects_without_centers() -> None:
    window = _normal_window()
    window["objects"].append(
        {
            "sample_id": "seq_a:no_center",
            "sequence": "seq_a",
            "track_id": "no_center",
            "states": [{"frame_id": 1, "fused": {}}],
        }
    )

    rows = score_group_windows([window], k_neighbors=2)

    assert {row["track_id"] for row in rows} == {"a", "b", "c"}


def test_score_group_windows_preserves_object_sample_id_and_uses_sequence_track_fallback() -> None:
    window = _normal_window()
    window["sample_id"] = "window_1"
    window["objects"][0]["sample_id"] = "custom:a"
    window["objects"][1].pop("sample_id")

    rows = score_group_windows([window], k_neighbors=2)
    sample_ids = {row["track_id"]: row["sample_id"] for row in rows}

    assert sample_ids["a"] == "custom:a"
    assert sample_ids["b"] == "seq_a:b"


def test_run_group_method_help_works_as_direct_script() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/fusiontrack/run_group_method.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--k-neighbors" in result.stdout
