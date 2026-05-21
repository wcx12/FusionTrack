from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from fusiontrack.group_graph import extract_object_states


IDENTIFIER_COLUMNS = ("window_id", "sequence", "frame_start", "frame_end")
FEATURE_COLUMNS = (
    "num_objects",
    "num_frames",
    "mean_group_size",
    "max_group_size",
    "mean_dispersion",
    "max_dispersion",
    "mean_speed",
    "std_speed",
    "neighbor_churn",
)
OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS


def build_group_feature_row(window: dict) -> dict[str, Any]:
    states = extract_object_states(window)
    frame_start, frame_end = _window_frame_bounds(window)
    by_frame = _states_by_frame(states)
    by_track = _states_by_track(states)
    group_sizes = [len(frame_states) for frame_states in by_frame.values()]
    dispersions = [_frame_dispersion(frame_states) for frame_states in by_frame.values()]
    speeds = _track_speeds(by_track)

    row: dict[str, Any] = {
        "window_id": _window_id(window),
        "sequence": str(window.get("sequence", "")),
        "frame_start": int(window.get("frame_start", frame_start)),
        "frame_end": int(window.get("frame_end", frame_end)),
        "num_objects": int(len(by_track)),
        "num_frames": int(len(by_frame)),
        "mean_group_size": _mean(group_sizes),
        "max_group_size": _max(group_sizes),
        "mean_dispersion": _mean(dispersions),
        "max_dispersion": _max(dispersions),
        "mean_speed": _mean(speeds),
        "std_speed": _std(speeds),
        "neighbor_churn": _neighbor_churn(by_frame),
    }
    return {key: _finite_default(value) for key, value in row.items()}


def build_group_feature_table(
    windows: Iterable[dict],
    sort_by_window_id: bool = True,
) -> pd.DataFrame:
    rows = [build_group_feature_row(window) for window in windows]
    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if table.empty:
        return table
    for column in FEATURE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    for column in ("frame_start", "frame_end"):
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0).astype(int)
    if sort_by_window_id:
        table = table.sort_values("window_id", kind="stable")
    return table.reset_index(drop=True)


def _window_id(window: dict) -> str:
    value = window.get("window_id", window.get("sample_id", ""))
    return str(value) if value not in (None, "") else ""


def _window_frame_bounds(window: dict) -> tuple[int, int]:
    frame_ids: list[int] = []
    for obj in window.get("objects", []):
        for state in obj.get("states", []):
            if "frame_id" in state:
                frame_ids.append(int(state["frame_id"]))
    default_start = min(frame_ids) if frame_ids else 0
    default_end = max(frame_ids) if frame_ids else default_start
    return (
        int(window.get("frame_start", default_start)),
        int(window.get("frame_end", default_end)),
    )


def _states_by_frame(states: list[dict]) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = {}
    for state in states:
        by_frame.setdefault(int(state["frame_id"]), []).append(state)
    return dict(sorted(by_frame.items()))


def _states_by_track(states: list[dict]) -> dict[str, list[dict]]:
    by_track: dict[str, list[dict]] = {}
    for state in states:
        by_track.setdefault(str(state["track_id"]), []).append(state)
    for track_states in by_track.values():
        track_states.sort(key=lambda state: int(state["frame_id"]))
    return dict(sorted(by_track.items()))


def _frame_dispersion(frame_states: list[dict]) -> float:
    if not frame_states:
        return 0.0
    center = [
        sum(float(state["center_xy"][0]) for state in frame_states) / len(frame_states),
        sum(float(state["center_xy"][1]) for state in frame_states) / len(frame_states),
    ]
    return float(
        np.mean([math.dist(state["center_xy"], center) for state in frame_states])
    )


def _track_speeds(by_track: dict[str, list[dict]]) -> list[float]:
    speeds: list[float] = []
    for states in by_track.values():
        for previous, current in zip(states, states[1:]):
            delta_frames = max(int(current["frame_id"]) - int(previous["frame_id"]), 1)
            speeds.append(
                math.dist(previous["center_xy"], current["center_xy"])
                / float(delta_frames)
            )
    return speeds


def _neighbor_churn(by_frame: dict[int, list[dict]]) -> float:
    previous_neighbors_by_track: dict[str, str | None] = {}
    changes = 0
    comparisons = 0
    for _, frame_states in sorted(by_frame.items()):
        current_neighbors = _nearest_neighbors(frame_states)
        for track_id, neighbor_id in current_neighbors.items():
            if track_id in previous_neighbors_by_track:
                comparisons += 1
                if previous_neighbors_by_track[track_id] != neighbor_id:
                    changes += 1
            previous_neighbors_by_track[track_id] = neighbor_id
    if comparisons == 0:
        return 0.0
    return changes / float(comparisons)


def _nearest_neighbors(frame_states: list[dict]) -> dict[str, str | None]:
    neighbors: dict[str, str | None] = {}
    for state in sorted(frame_states, key=lambda item: str(item["track_id"])):
        distances = [
            (math.dist(state["center_xy"], other["center_xy"]), str(other["track_id"]))
            for other in frame_states
            if str(other["track_id"]) != str(state["track_id"])
        ]
        distances.sort(key=lambda item: (item[0], item[1]))
        neighbors[str(state["track_id"])] = distances[0][1] if distances else None
    return neighbors


def _mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max(values: list[float] | list[int]) -> float:
    return float(np.max(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _finite_default(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value
