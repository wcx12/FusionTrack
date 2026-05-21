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
            {"sample_id": "S1:1", "sequence": "S1", "track_id": "1", "source": "individual_simple", "score": 1.0, "component_scores": {"speed": 1.0}},
            {"sample_id": "S1:2", "sequence": "S1", "track_id": "2", "source": "individual_simple", "score": 4.0, "component_scores": {"speed": 4.0}},
        ],
    )
    write_jsonl(
        group_jsonl,
        [
            {"sample_id": "S1:2", "sequence": "S1", "track_id": "2", "source": "group", "score": 2.0, "component_scores": {"leave": 2.0}},
            {"sample_id": "S1:3", "sequence": "S1", "track_id": "3", "source": "group", "score": 5.0, "component_scores": {"leave": 5.0}},
        ],
    )

    summary = fuse_score_records(individual_jsonl, group_jsonl, output_jsonl, output_csv, alpha=0.5)

    assert summary["num_fused_scores"] == 3
    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    by_id = {row["sample_id"]: row for row in rows}
    assert by_id["S1:1"]["metadata"]["used_sources"] == ["individual"]
    assert by_id["S1:3"]["metadata"]["used_sources"] == ["group"]
    assert by_id["S1:2"]["metadata"]["used_sources"] == ["individual", "group"]
    assert "individual_speed" in by_id["S1:2"]["component_scores"]
    assert "group_leave" in by_id["S1:2"]["component_scores"]
    assert output_csv.exists()
