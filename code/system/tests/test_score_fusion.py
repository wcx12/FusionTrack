from __future__ import annotations

import json
from pathlib import Path

from fusiontrack.score_fusion import fuse_score_records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_score_fusion_combines_and_falls_back(tmp_path: Path) -> None:
    individual_jsonl = tmp_path / "individual.jsonl"
    group_jsonl = tmp_path / "group.jsonl"
    output_jsonl = tmp_path / "fused.jsonl"
    output_csv = tmp_path / "fused.csv"
    write_jsonl(
        individual_jsonl,
        [
            {
                "sample_id": "S1:1",
                "sequence": "S1",
                "track_id": "1",
                "source": "individual_simple",
                "score": 1.0,
                "event_score": 0.6,
                "frame_event_scores": [
                    {"frame": 11, "score": 0.6, "dominant_reason": "speed"},
                    {"frame": 12, "score": 0.5, "dominant_reason": "speed"},
                ],
                "component_scores": {"speed": 1.0},
            },
            {
                "sample_id": "S1:2",
                "sequence": "S1",
                "track_id": "2",
                "source": "individual_simple",
                "score": 4.0,
                "event_score": 0.4,
                "event_segments": [{"frame_start": 2, "frame_end": 2, "score": 0.4, "dominant_reason": "speed"}],
                "frame_event_scores": [{"frame": 2, "score": 0.4, "dominant_reason": "speed"}],
                "component_scores": {"speed": 4.0},
            },
        ],
    )
    write_jsonl(
        group_jsonl,
        [
            {
                "sample_id": "S1:2",
                "sequence": "S1",
                "track_id": "2",
                "source": "group",
                "score": 2.0,
                "event_score": 0.9,
                "event_segments": [{"frame_start": 3, "frame_end": 5, "score": 0.9, "dominant_reason": "leave"}],
                "frame_event_scores": [{"frame": 3, "score": 0.9, "dominant_reason": "leave"}],
                "component_scores": {"leave": 2.0},
            },
            {"sample_id": "S1:3", "sequence": "S1", "track_id": "3", "source": "group", "score": 5.0, "component_scores": {"leave": 5.0}},
        ],
    )

    summary = fuse_score_records(individual_jsonl, group_jsonl, output_jsonl, output_csv, alpha=0.5)

    assert summary["num_fused_scores"] == 3
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    by_id = {row["sample_id"]: row for row in rows}
    assert by_id["S1:1"]["metadata"]["used_sources"] == ["individual"]
    assert by_id["S1:1"]["event_segments"] == [
        {
            "frame_start": 11,
            "frame_end": 12,
            "score": 0.6,
            "dominant_reason": "speed",
            "num_frames": 2,
            "source": "individual",
        }
    ]
    assert by_id["S1:3"]["metadata"]["used_sources"] == ["group"]
    assert by_id["S1:2"]["metadata"]["used_sources"] == ["individual", "group"]
    assert "individual_speed" in by_id["S1:2"]["component_scores"]
    assert "group_leave" in by_id["S1:2"]["component_scores"]
    assert {"S_ind", "S_grp", "S_event", "S_fused"} <= set(by_id["S1:2"]["component_scores"])
    assert by_id["S1:2"]["event_score"] == 0.9
    assert [segment["source"] for segment in by_id["S1:2"]["event_segments"]] == ["individual", "group"]
    assert [item["frame"] for item in by_id["S1:2"]["frame_event_scores"]] == [2, 3]
    assert by_id["S1:2"]["frame_event_scores"][1]["source"] == "group"
    assert output_csv.exists()
