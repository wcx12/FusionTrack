from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusiontrack.group_tracking import jaccard, track_groups


def test_jaccard_handles_empty_and_overlapping_sets() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_track_groups_continues_ids_and_records_split_events() -> None:
    tracked = track_groups(
        {
            1: [{"a", "b", "c"}],
            2: [{"a", "b"}, {"c"}],
        },
        threshold=0.3,
    )

    first_group_id = tracked["frames"][1][0]["group_id"]
    continued = next(
        group for group in tracked["frames"][2] if group["members"] == {"a", "b"}
    )
    assert continued["group_id"] == first_group_id
    assert tracked["events"] == [
        {
            "frame_id": 2,
            "event_type": "split",
            "source_group_ids": [first_group_id],
            "target_group_ids": [first_group_id, tracked["frames"][2][1]["group_id"]],
        }
    ]


def test_track_groups_records_merge_events() -> None:
    tracked = track_groups(
        {
            1: [{"a", "b"}, {"c"}],
            2: [{"a", "b", "c"}],
        },
        threshold=0.3,
    )

    source_ids = [group["group_id"] for group in tracked["frames"][1]]
    assert tracked["events"] == [
        {
            "frame_id": 2,
            "event_type": "merge",
            "source_group_ids": source_ids,
            "target_group_ids": [tracked["frames"][2][0]["group_id"]],
        }
    ]
