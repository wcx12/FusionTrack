from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Any, Sequence

from .schemas import AnomalyLabel, build_sample_id

DEFAULT_GROUP_ANOMALIES = (
    "leave_group",
    "against_motion",
    "neighbor_replacement",
    "population_change",
    "dispersion_change",
    "split_merge",
)
MODALITIES = ("fused", "rgb", "thermal")


def inject_group_anomalies(
    windows: list[dict[str, Any]],
    anomaly_fraction: float,
    seed: int,
    anomaly_types: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], list[AnomalyLabel]]:
    if not 0 <= anomaly_fraction <= 1:
        raise ValueError("anomaly_fraction must be between 0 and 1")

    selected_types = tuple(anomaly_types or DEFAULT_GROUP_ANOMALIES)
    _validate_types(selected_types, DEFAULT_GROUP_ANOMALIES)

    injected = deepcopy(windows)
    candidates = [index for index, window in enumerate(injected) if window.get("objects")]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    target_count = _selection_count(len(candidates), anomaly_fraction)

    labels: list[AnomalyLabel] = []
    type_cursor = 0
    for window_index in candidates:
        if len(labels) >= target_count:
            break
        window = injected[window_index]
        objects = window["objects"]
        object_indices = list(range(len(objects)))
        rng.shuffle(object_indices)
        applied = False
        for object_index in object_indices:
            obj = objects[object_index]
            for offset in range(len(selected_types)):
                anomaly_type = selected_types[
                    (type_cursor + offset) % len(selected_types)
                ]
                if _apply_group_anomaly(window, object_index, anomaly_type, rng):
                    labels.append(_build_label(window, obj, anomaly_type, seed))
                    type_cursor = (type_cursor + offset + 1) % len(selected_types)
                    applied = True
                    break
            if applied:
                break

    return injected, labels


def _validate_types(types: Sequence[str], allowed: Sequence[str]) -> None:
    unknown = sorted(set(types) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported anomaly_types: {unknown}")


def _selection_count(candidate_count: int, fraction: float) -> int:
    if candidate_count == 0 or fraction <= 0:
        return 0
    return min(candidate_count, max(1, math.ceil(candidate_count * fraction)))


def _apply_group_anomaly(
    window: dict[str, Any],
    object_index: int,
    anomaly_type: str,
    rng: random.Random,
) -> bool:
    objects = window["objects"]
    obj = objects[object_index]
    states = obj.get("states", [])

    if anomaly_type == "leave_group":
        changed = _shift_states(states, dx=_signed(rng, 12.0), dy=_signed(rng, 8.0))
    elif anomaly_type == "against_motion":
        changed = _reverse_motion(states)
    elif anomaly_type == "neighbor_replacement":
        changed = _neighbor_replacement(objects, object_index)
    elif anomaly_type == "population_change":
        changed = _population_change(objects, object_index)
    elif anomaly_type == "dispersion_change":
        changed = _dispersion_change(window, obj)
    elif anomaly_type == "split_merge":
        changed = _split_merge(states)
    else:
        changed = False

    if changed and obj in objects:
        obj.setdefault("metadata", {})["group_anomaly_type"] = anomaly_type
    return changed


def _shift_states(states: Sequence[dict[str, Any]], dx: float, dy: float) -> bool:
    changed = False
    for state in states:
        for center in _centers(state):
            center[0] += dx
            center[1] += dy
            changed = True
    return changed


def _reverse_motion(states: list[dict[str, Any]]) -> bool:
    if len(states) < 2:
        return False
    before = _state_center_tuples(states)
    centers_by_state = [[center[:] for center in _centers(state)] for state in states]
    if not any(centers_by_state):
        return False
    reversed_centers = list(reversed(centers_by_state))
    for state, source_centers in zip(states, reversed_centers):
        for center, source_center in zip(_centers(state), source_centers):
            center[0] = source_center[0]
            center[1] = source_center[1]
    return before != _state_center_tuples(states)


def _neighbor_replacement(objects: list[dict[str, Any]], object_index: int) -> bool:
    obj = objects[object_index]
    if len(objects) < 2:
        return False

    neighbor = objects[0 if object_index != 0 else 1]
    if not _object_has_centers(obj) or not _object_has_centers(neighbor):
        return False
    before = _state_center_tuples(obj.get("states", []))
    for state, neighbor_state in zip(obj.get("states", []), neighbor.get("states", [])):
        neighbor_centers = _centers(neighbor_state)
        for center, neighbor_center in zip(_centers(state), neighbor_centers):
            center[0] = neighbor_center[0] + 1.5
            center[1] = neighbor_center[1] + 1.5
    return before != _state_center_tuples(obj.get("states", []))


def _population_change(objects: list[dict[str, Any]], object_index: int) -> bool:
    obj = objects[object_index]
    if not _object_has_centers(obj):
        return False

    duplicate = deepcopy(obj)
    duplicate["sample_id"] = f"{obj.get('sample_id', 'object')}:population_copy"
    duplicate["track_id"] = f"{obj.get('track_id', 'object')}:population_copy"
    duplicate.setdefault("metadata", {})["population_change"] = "synthetic_copy"
    objects.append(duplicate)
    return True


def _dispersion_change(window: dict[str, Any], obj: dict[str, Any]) -> bool:
    all_centers = [
        center
        for other in window.get("objects", [])
        for state in other.get("states", [])
        for center in _centers(state)
    ]
    if not all_centers or not _object_has_centers(obj):
        return False

    mean_x = sum(center[0] for center in all_centers) / len(all_centers)
    mean_y = sum(center[1] for center in all_centers) / len(all_centers)
    before = _state_center_tuples(obj.get("states", []))
    for state in obj.get("states", []):
        for center in _centers(state):
            dx = center[0] - mean_x
            dy = center[1] - mean_y
            center[0] += dx if dx else 5.0
            center[1] += dy if dy else 5.0
    return before != _state_center_tuples(obj.get("states", []))


def _split_merge(states: list[dict[str, Any]]) -> bool:
    if len(states) < 2:
        return False
    before = _state_center_tuples(states)
    if not before:
        return False
    midpoint = len(states) // 2
    _shift_states(states[:midpoint], dx=-5.0, dy=0.0)
    _shift_states(states[midpoint:], dx=5.0, dy=0.0)
    return before != _state_center_tuples(states)


def _centers(item: dict[str, Any]) -> list[list[float]]:
    return [center for modality in MODALITIES if (center := _center(item.get(modality))) is not None]


def _center(modality_value: Any) -> list[float] | None:
    if isinstance(modality_value, dict):
        center = modality_value.get("center_xy")
        if isinstance(center, list) and len(center) >= 2:
            return center
    return None


def _signed(rng: random.Random, magnitude: float) -> float:
    return magnitude if rng.random() >= 0.5 else -magnitude


def _object_has_centers(obj: dict[str, Any]) -> bool:
    return any(_centers(state) for state in obj.get("states", []))


def _state_center_tuples(states: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (center[0], center[1])
        for state in states
        for center in _centers(state)
    ]


def _build_label(
    window: dict[str, Any],
    obj: dict[str, Any],
    anomaly_type: str,
    seed: int,
) -> AnomalyLabel:
    states = obj.get("states", [])
    frames = [int(state["frame_id"]) for state in states if "frame_id" in state]
    sequence = str(window.get("sequence", ""))
    track_id = str(obj.get("track_id", ""))
    sample_id = str(obj.get("sample_id") or build_sample_id(sequence, track_id))
    return AnomalyLabel(
        sample_id=sample_id,
        sequence=sequence,
        track_id=track_id,
        frame_start=int(window.get("frame_start", min(frames) if frames else 0)),
        frame_end=int(window.get("frame_end", max(frames) if frames else 0)),
        label=1,
        anomaly_type=anomaly_type,
        injection_seed=seed,
        metadata={
            "source": "group_injection",
            "window_id": window.get("window_id"),
        },
    )
