from __future__ import annotations

from typing import Iterable

from .group_graph import connected_components


def discover_frame_groups(
    states: list[dict],
    edges_by_frame: dict[int, set[tuple[str, str]]],
) -> dict[int, list[set[str]]]:
    track_ids_by_frame: dict[int, set[str]] = {}
    for state in states:
        track_ids_by_frame.setdefault(int(state["frame_id"]), set()).add(str(state["track_id"]))

    frame_groups: dict[int, list[set[str]]] = {}
    for frame_id, track_ids in sorted(track_ids_by_frame.items()):
        groups = connected_components(track_ids, edges_by_frame.get(frame_id, set()))
        frame_groups[frame_id] = sorted(groups, key=lambda group: sorted(group))
    return frame_groups


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def track_groups(frame_groups: dict[int, list[set[str]]], threshold: float = 0.3) -> dict:
    next_group_index = 1
    previous_records: list[dict] = []
    tracked_frames: dict[int, list[dict]] = {}
    events: list[dict] = []

    for frame_id in sorted(frame_groups):
        current_groups = [set(group) for group in frame_groups[frame_id]]
        matches_by_current: list[list[dict]] = []
        for members in current_groups:
            matches = [
                {**previous, "overlap": jaccard(members, previous["members"])}
                for previous in previous_records
                if jaccard(members, previous["members"]) >= threshold
            ]
            matches.sort(key=lambda record: (-record["overlap"], record["group_id"]))
            matches_by_current.append(matches)

        used_previous_ids: set[str] = set()
        current_records: list[dict] = []
        for members, matches in zip(current_groups, matches_by_current):
            chosen = next(
                (
                    match
                    for match in matches
                    if match["group_id"] not in used_previous_ids
                ),
                None,
            )
            if chosen is None:
                group_id = f"g{next_group_index}"
                next_group_index += 1
            else:
                group_id = chosen["group_id"]
                used_previous_ids.add(group_id)
            current_records.append({"group_id": group_id, "members": members})

        merge_events = _merge_events(frame_id, current_records, matches_by_current)
        split_events = _split_events(frame_id, current_records, matches_by_current)
        events.extend(merge_events)
        events.extend(split_events)

        tracked_frames[frame_id] = current_records
        previous_records = current_records

    return {"frames": tracked_frames, "events": events}


def _merge_events(
    frame_id: int,
    current_records: list[dict],
    matches_by_current: list[list[dict]],
) -> list[dict]:
    events: list[dict] = []
    for current, matches in zip(current_records, matches_by_current):
        source_ids = sorted({match["group_id"] for match in matches})
        if len(source_ids) > 1:
            events.append(
                {
                    "frame_id": frame_id,
                    "event_type": "merge",
                    "source_group_ids": source_ids,
                    "target_group_ids": [current["group_id"]],
                }
            )
    return events


def _split_events(
    frame_id: int,
    current_records: list[dict],
    matches_by_current: list[list[dict]],
) -> list[dict]:
    targets_by_source: dict[str, list[str]] = {}
    for current, matches in zip(current_records, matches_by_current):
        for match in matches:
            targets_by_source.setdefault(match["group_id"], []).append(current["group_id"])

    events: list[dict] = []
    for source_id in sorted(targets_by_source):
        target_ids = []
        for group_id in targets_by_source[source_id]:
            if group_id not in target_ids:
                target_ids.append(group_id)
        if len(target_ids) > 1:
            events.append(
                {
                    "frame_id": frame_id,
                    "event_type": "split",
                    "source_group_ids": [source_id],
                    "target_group_ids": target_ids,
                }
            )
    return events
