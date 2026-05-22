from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable

from protocol.schemas import build_sample_id


MODALITY_ORDER = ("fused", "rgb", "thermal")


def extract_object_states(window: dict) -> list[dict]:
    states: list[dict] = []
    sequence = window.get("sequence")
    window_sample_id = window.get("sample_id")

    for obj in window.get("objects", []):
        track_id = obj.get("track_id")
        if track_id is None:
            continue
        track_id = str(track_id)
        obj_sequence = obj.get("sequence", sequence)
        sample_id = obj.get("sample_id") or build_sample_id(str(obj_sequence), track_id)
        category = (
            obj.get("category")
            or obj.get("category_name")
            or obj.get("category_id")
        )

        for raw_state in obj.get("states", []):
            if "frame_id" not in raw_state:
                continue
            center_info = _select_center(raw_state)
            if center_info is None:
                continue
            modality, center, bbox = center_info
            states.append(
                {
                    "frame_id": int(raw_state["frame_id"]),
                    "sample_id": str(sample_id),
                    "sequence": obj_sequence,
                    "track_id": track_id,
                    "category": category,
                    "center_xy": center,
                    "bbox": bbox,
                    "velocity": _vector2(raw_state.get("velocity"))
                    or _vector2(raw_state.get(modality, {}).get("velocity") if isinstance(raw_state.get(modality), dict) else None),
                    "source_modality": modality,
                    "raw_state": raw_state,
                }
            )

    return sorted(states, key=lambda state: (state["frame_id"], state["track_id"]))


def compute_relative_displacements(states: list[dict]) -> list[dict]:
    enriched = [dict(state) for state in states]
    by_track: dict[str, list[dict]] = {}
    for state in enriched:
        by_track.setdefault(str(state["track_id"]), []).append(state)

    for track_states in by_track.values():
        track_states.sort(key=lambda state: state["frame_id"])
        previous: dict | None = None
        for state in track_states:
            if previous is not None:
                dx = float(state["center_xy"][0]) - float(previous["center_xy"][0])
                dy = float(state["center_xy"][1]) - float(previous["center_xy"][1])
                state["velocity"] = [dx, dy]
            else:
                state["velocity"] = _vector2(state.get("velocity")) or [0.0, 0.0]
            state["dx"] = float(state["velocity"][0])
            state["dy"] = float(state["velocity"][1])
            previous = state

    by_frame: dict[int, list[dict]] = {}
    for state in enriched:
        by_frame.setdefault(int(state["frame_id"]), []).append(state)

    for frame_states in by_frame.values():
        median_dx = median([state["dx"] for state in frame_states]) if frame_states else 0.0
        median_dy = median([state["dy"] for state in frame_states]) if frame_states else 0.0
        for state in frame_states:
            rel_dx = float(state["dx"]) - float(median_dx)
            rel_dy = float(state["dy"]) - float(median_dy)
            state["rel_dx"] = rel_dx
            state["rel_dy"] = rel_dy
            state["rel_velocity"] = [rel_dx, rel_dy]

    return sorted(enriched, key=lambda state: (state["frame_id"], state["track_id"]))


def build_spatial_edges(
    states: list[dict],
    k_neighbors: int = 3,
    rho_p: float = float("inf"),
    rho_v: float = float("inf"),
) -> dict[int, set[tuple[str, str]]]:
    by_frame: dict[int, list[dict]] = {}
    for state in states:
        by_frame.setdefault(int(state["frame_id"]), []).append(state)

    edges_by_frame: dict[int, set[tuple[str, str]]] = {}
    for frame_id, frame_states in by_frame.items():
        edges: set[tuple[str, str]] = set()
        sorted_states = sorted(frame_states, key=lambda state: str(state["track_id"]))
        for state in sorted_states:
            distances = [
                (_position_distance(state, other), other)
                for other in sorted_states
                if other["track_id"] != state["track_id"]
            ]
            distances.sort(key=lambda item: (item[0], str(item[1]["track_id"])))
            for position_distance, other in distances[: max(0, int(k_neighbors))]:
                if position_distance > rho_p:
                    continue
                if _relative_velocity_distance(state, other) > rho_v:
                    continue
                edge = tuple(sorted((str(state["track_id"]), str(other["track_id"]))))
                edges.add(edge)
        edges_by_frame[frame_id] = edges
    return dict(sorted(edges_by_frame.items()))


def connected_components(track_ids: Iterable[str], edges: set[tuple[str, str]]) -> list[set[str]]:
    nodes = {str(track_id) for track_id in track_ids}
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        left = str(left)
        right = str(right)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
        nodes.update((left, right))

    components: list[set[str]] = []
    seen: set[str] = set()
    for node in sorted(nodes):
        if node in seen:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(sorted(adjacency.get(current, set()) - seen, reverse=True))
        components.append(component)
    return components


def _select_center(state: dict[str, Any]) -> tuple[str, list[float], Any] | None:
    for modality in MODALITY_ORDER:
        modality_state = state.get(modality)
        if not isinstance(modality_state, dict):
            continue
        center = _center(modality_state)
        if center is not None:
            return modality, center, modality_state.get("bbox") or state.get("bbox")
    return None


def _center(value: dict[str, Any]) -> list[float] | None:
    center = value.get("center_xy")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return [float(center[0]), float(center[1])]
    return None


def _vector2(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    return None


def _position_distance(left: dict, right: dict) -> float:
    return math.dist(left["center_xy"], right["center_xy"])


def _relative_velocity_distance(left: dict, right: dict) -> float:
    left_velocity = left.get("rel_velocity") or [0.0, 0.0]
    right_velocity = right.get("rel_velocity") or [0.0, 0.0]
    return math.dist(left_velocity, right_velocity)
