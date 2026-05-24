from __future__ import annotations

import math
from statistics import median
from typing import Iterable

from .group_graph import (
    build_spatial_edges,
    compute_relative_displacements,
    extract_object_states,
)
from .group_tracking import discover_frame_groups, jaccard, track_groups


COMPONENT_NAMES = (
    "leave",
    "motion",
    "neighbor",
    "count",
    "dispersion",
    "split_merge",
    "object_group",
    "group_event",
)


def score_group_windows(
    windows: Iterable[dict],
    k_neighbors: int = 3,
    rho_p: float = float("inf"),
    rho_v: float = float("inf"),
    eta: float = 0.5,
) -> list[dict]:
    rows: list[dict] = []
    for window in windows:
        rows.extend(_score_one_window(window, k_neighbors, rho_p, rho_v, eta))
    return rows


def _score_one_window(
    window: dict,
    k_neighbors: int,
    rho_p: float,
    rho_v: float,
    eta: float,
) -> list[dict]:
    states = compute_relative_displacements(extract_object_states(window))
    if not states:
        return []

    edges_by_frame = build_spatial_edges(states, k_neighbors, rho_p, rho_v)
    frame_groups = discover_frame_groups(states, edges_by_frame)
    tracked = track_groups(frame_groups)

    state_by_frame_track = {
        (int(state["frame_id"]), str(state["track_id"])): state for state in states
    }
    group_by_frame_id = {
        (frame_id, group["group_id"]): group
        for frame_id, groups in tracked["frames"].items()
        for group in groups
    }
    events_by_frame_group = _events_by_frame_group(tracked["events"])
    neighbors_by_frame_track = _neighbors_by_frame_track(edges_by_frame)

    distance_history_by_track: dict[str, list[float]] = {}
    size_history_by_group: dict[str, list[int]] = {}
    dispersion_history_by_group: dict[str, list[float]] = {}
    previous_neighbors_by_track: dict[str, set[str]] = {}
    frame_scores_by_track: dict[str, list[dict]] = {}

    for frame_id in sorted(tracked["frames"]):
        for group in tracked["frames"][frame_id]:
            group_id = group["group_id"]
            members = set(group["members"])
            member_states = [
                state_by_frame_track[(frame_id, track_id)]
                for track_id in sorted(members)
                if (frame_id, track_id) in state_by_frame_track
            ]
            if not member_states:
                continue

            center = _mean_center(member_states)
            dispersion = _dispersion(member_states, center)
            count_score = _history_change(
                len(member_states), size_history_by_group.get(group_id, [])
            )
            dispersion_score = _history_change(
                dispersion, dispersion_history_by_group.get(group_id, [])
            )
            split_merge_score = 1.0 if events_by_frame_group.get((frame_id, group_id)) else 0.0

            for state in member_states:
                track_id = str(state["track_id"])
                distance = math.dist(state["center_xy"], center)
                leave_score = _history_increase(
                    distance, distance_history_by_track.get(track_id, [])
                )
                motion_score = _motion_score(state, member_states)
                current_neighbors = neighbors_by_frame_track.get((frame_id, track_id), set())
                previous_neighbors = previous_neighbors_by_track.get(track_id)
                neighbor_score = (
                    0.0
                    if previous_neighbors is None
                    else 1.0 - jaccard(current_neighbors, previous_neighbors)
                )
                object_group = max(leave_score, motion_score, neighbor_score)
                group_event = max(count_score, dispersion_score, split_merge_score)
                component_scores = {
                    "leave": leave_score,
                    "motion": motion_score,
                    "neighbor": neighbor_score,
                    "count": count_score,
                    "dispersion": dispersion_score,
                    "split_merge": split_merge_score,
                    "object_group": object_group,
                    "group_event": group_event,
                }
                frame_scores_by_track.setdefault(track_id, []).append(
                    {
                        "frame_id": frame_id,
                        "state": state,
                        "group_id": group_id,
                        "score": max(object_group, eta * group_event),
                        "component_scores": component_scores,
                    }
                )
                distance_history_by_track.setdefault(track_id, []).append(distance)
                previous_neighbors_by_track[track_id] = set(current_neighbors)

            size_history_by_group.setdefault(group_id, []).append(len(member_states))
            dispersion_history_by_group.setdefault(group_id, []).append(dispersion)

    return [
        _aggregate_track_scores(track_id, frame_scores, window)
        for track_id, frame_scores in sorted(frame_scores_by_track.items())
    ]


def _events_by_frame_group(events: list[dict]) -> dict[tuple[int, str], list[dict]]:
    by_group: dict[tuple[int, str], list[dict]] = {}
    for event in events:
        frame_id = int(event["frame_id"])
        for group_id in event.get("source_group_ids", []) + event.get("target_group_ids", []):
            by_group.setdefault((frame_id, group_id), []).append(event)
    return by_group


def _neighbors_by_frame_track(
    edges_by_frame: dict[int, set[tuple[str, str]]],
) -> dict[tuple[int, str], set[str]]:
    neighbors: dict[tuple[int, str], set[str]] = {}
    for frame_id, edges in edges_by_frame.items():
        for left, right in edges:
            neighbors.setdefault((frame_id, left), set()).add(right)
            neighbors.setdefault((frame_id, right), set()).add(left)
    return neighbors


def _mean_center(states: list[dict]) -> list[float]:
    return [
        sum(state["center_xy"][0] for state in states) / len(states),
        sum(state["center_xy"][1] for state in states) / len(states),
    ]


def _dispersion(states: list[dict], center: list[float]) -> float:
    return sum(math.dist(state["center_xy"], center) for state in states) / len(states)


def _history_increase(current: float, history: list[float]) -> float:
    if not history:
        return 0.0
    baseline = median(history)
    if baseline <= 0.0:
        return max(0.0, current)
    return max(0.0, current / baseline - 1.0)


def _history_change(current: float, history: list[float]) -> float:
    if not history:
        return 0.0
    baseline = median(history)
    if baseline <= 0.0:
        return abs(current - baseline)
    return abs(current / baseline - 1.0)


def _motion_score(state: dict, group_states: list[dict]) -> float:
    velocity = state.get("rel_velocity") or [0.0, 0.0]
    others = [other for other in group_states if other["track_id"] != state["track_id"]]
    if not others:
        return 0.0
    mean_velocity = [
        sum((other.get("rel_velocity") or [0.0, 0.0])[0] for other in others) / len(others),
        sum((other.get("rel_velocity") or [0.0, 0.0])[1] for other in others) / len(others),
    ]
    velocity_norm = math.hypot(velocity[0], velocity[1])
    mean_norm = math.hypot(mean_velocity[0], mean_velocity[1])
    if velocity_norm == 0.0:
        return 0.0
    if mean_norm == 0.0:
        return 1.0
    cosine = (
        velocity[0] * mean_velocity[0] + velocity[1] * mean_velocity[1]
    ) / (velocity_norm * mean_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine


def _aggregate_track_scores(track_id: str, frame_scores: list[dict], window: dict) -> dict:
    best_frame = max(frame_scores, key=lambda item: item["score"])
    frame_start, frame_end = _frame_bounds(frame_scores, window)
    window_id = str(window.get("window_id", window.get("sample_id", "")))
    component_scores = {
        name: max(score["component_scores"][name] for score in frame_scores)
        for name in COMPONENT_NAMES
    }
    dominant_reason = max(
        COMPONENT_NAMES,
        key=lambda name: component_scores[name],
    )
    frame_event_scores = _frame_event_scores(frame_scores)
    event_segments = _event_segments_from_frame_scores(frame_event_scores)
    state = best_frame["state"]
    return {
        "sample_id": state.get("sample_id") or f"{state.get('sequence')}:{track_id}",
        "window_id": window_id,
        "sequence": state.get("sequence") or window.get("sequence"),
        "track_id": track_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "source": "fusiontrack_group_graph",
        "score": max(score["score"] for score in frame_scores),
        "event_score": max((score["score"] for score in frame_event_scores), default=0.0),
        "event_segments": event_segments,
        "frame_event_scores": frame_event_scores,
        "component_scores": component_scores,
        "metadata": {
            "dominant_reason": dominant_reason,
            "num_frames": len(frame_scores),
            "window_id": window_id,
            "window_sample_id": window.get("sample_id"),
            "group_ids": sorted({score["group_id"] for score in frame_scores}),
        },
    }


def _frame_bounds(frame_scores: list[dict], window: dict) -> tuple[int, int]:
    frame_ids = [int(score["frame_id"]) for score in frame_scores if "frame_id" in score]
    default_start = min(frame_ids) if frame_ids else 0
    default_end = max(frame_ids) if frame_ids else default_start
    return (
        int(window.get("frame_start", default_start)),
        int(window.get("frame_end", default_end)),
    )


def _frame_event_scores(frame_scores: list[dict]) -> list[dict]:
    events = []
    for item in sorted(frame_scores, key=lambda score: int(score["frame_id"])):
        components = item["component_scores"]
        dominant_reason = max(COMPONENT_NAMES, key=lambda name: components[name])
        events.append(
            {
                "frame": int(item["frame_id"]),
                "score": float(item["score"]),
                "dominant_reason": dominant_reason,
                "component_scores": {
                    name: float(components[name])
                    for name in COMPONENT_NAMES
                },
            }
        )
    return events


def _event_segments_from_frame_scores(frame_event_scores: list[dict]) -> list[dict]:
    segments = []
    current: dict | None = None
    for item in frame_event_scores:
        score = float(item.get("score", 0.0) or 0.0)
        if score <= 0.0:
            if current is not None:
                segments.append(current)
                current = None
            continue
        frame = int(item["frame"])
        reason = str(item.get("dominant_reason", "group_event"))
        if current is None:
            current = {
                "frame_start": frame,
                "frame_end": frame,
                "score": score,
                "dominant_reason": reason,
            }
            continue
        if frame <= int(current["frame_end"]) + 1:
            current["frame_end"] = frame
            current["score"] = max(float(current["score"]), score)
            if score >= float(current["score"]):
                current["dominant_reason"] = reason
        else:
            segments.append(current)
            current = {
                "frame_start": frame,
                "frame_end": frame,
                "score": score,
                "dominant_reason": reason,
            }
    if current is not None:
        segments.append(current)
    return segments
