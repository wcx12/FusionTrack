from __future__ import annotations

import json
from pathlib import Path

from fusiontrack.simple_detectors import score_fused_trajectories_simple


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def trajectory(sample_id: str, centers: list[tuple[float, float]]) -> dict:
    sequence, track_id = sample_id.split(":")
    return {
        "sample_id": sample_id,
        "sequence": sequence,
        "track_id": track_id,
        "category_id": 1,
        "category_name": "ship",
        "points": [
            {
                "frame_id": idx,
                "fused": {
                    "center_xy": [x, y],
                    "confidence": 0.95,
                    "source_modalities": ["rgb", "thermal"],
                    "component_scores": {"modal_offset_distance": 1.0},
                },
            }
            for idx, (x, y) in enumerate(centers)
        ],
    }


def test_simple_detector_scores_irregular_motion_higher(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "fused.jsonl"
    output_jsonl = tmp_path / "scores.jsonl"
    output_csv = tmp_path / "scores.csv"
    write_jsonl(
        input_jsonl,
        [
            trajectory("S1:1", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]),
            trajectory("S1:2", [(0, 0), (1, 0), (30, 0), (31, 10), (32, 30)]),
        ],
    )

    summary = score_fused_trajectories_simple(input_jsonl, output_jsonl, output_csv)

    assert summary["num_scores"] == 2
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    scores = {row["sample_id"]: row["score"] for row in rows}
    assert scores["S1:2"] > scores["S1:1"]
    assert output_csv.exists()


def test_simple_detector_outputs_frame_event_evidence_and_explanation(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "fused.jsonl"
    output_jsonl = tmp_path / "scores.jsonl"
    output_csv = tmp_path / "scores.csv"
    write_jsonl(
        input_jsonl,
        [
            trajectory("S1:1", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]),
            trajectory("S1:2", [(0, 0), (1, 0), (30, 0), (31, 10), (32, 30)]),
        ],
    )

    score_fused_trajectories_simple(input_jsonl, output_jsonl, output_csv)

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    irregular = next(row for row in rows if row["sample_id"] == "S1:2")
    frame_events = irregular["frame_event_scores"]
    event_scores = [row["score"] for row in frame_events]

    assert frame_events
    assert frame_events == sorted(frame_events, key=lambda row: row["frame"])
    assert irregular["event_score"] == max(event_scores)
    assert irregular["event_segments"]
    assert irregular["explanation_schema"]["schema_version"] == 1
    assert irregular["explanation_schema"]["evidence_source"] == "event_segments"
    assert irregular["explanation_schema"]["top_reason"] in {
        "speed_spike",
        "turn_irregularity",
        "low_confidence_ratio",
        "modal_offset_median",
    }
    assert any(row["dominant_reason"] == "speed_spike" for row in frame_events)
    assert all("component_scores" in row for row in frame_events)
