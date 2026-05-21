from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from mtf_ba.group_interface import GroupScoreRecord, GroupWindow, iter_group_window_jsonl, write_group_score_records_jsonl


EPSILON = 1e-6


def _center_from_state(state: dict[str, Any]) -> tuple[float, float] | None:
    fused = state.get("fused")
    if fused and fused.get("center_xy"):
        return float(fused["center_xy"][0]), float(fused["center_xy"][1])
    centers = []
    for modality in ("rgb", "thermal"):
        branch = state.get(modality)
        if branch and branch.get("center_xy"):
            centers.append((float(branch["center_xy"][0]), float(branch["center_xy"][1])))
    if not centers:
        return None
    return (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers),
    )


def _object_centers(obj: dict[str, Any]) -> list[tuple[float, float]]:
    centers = []
    for state in obj.get("states", []):
        center = _center_from_state(state)
        if center is not None:
            centers.append(center)
    return centers


def _displacement(centers: list[tuple[float, float]]) -> tuple[float, float]:
    if len(centers) < 2:
        return (0.0, 0.0)
    return (centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _median_pair(values: list[tuple[float, float]]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    return (median([value[0] for value in values]), median([value[1] for value in values]))


def _robust(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    vals = list(values.values())
    med = median(vals)
    mad = median([abs(value - med) for value in vals])
    if mad <= EPSILON:
        ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
        denom = max(len(ordered) - 1, 1)
        return {key: idx / denom for idx, (key, _) in enumerate(ordered)}
    return {key: max(0.0, (value - med) / (1.4826 * mad + EPSILON)) for key, value in values.items()}


def _window_to_dict(window: GroupWindow | dict[str, Any]) -> dict[str, Any]:
    return window.to_dict() if hasattr(window, "to_dict") else window


class LightweightGroupAnomalyDetector:
    def score_windows(
        self,
        windows: Iterable[GroupWindow | dict[str, Any]],
    ) -> Iterable[GroupScoreRecord]:
        for window_obj in windows:
            window = _window_to_dict(window_obj)
            objects = window.get("objects", [])
            centers_by_track = {
                str(obj["track_id"]): _object_centers(obj)
                for obj in objects
            }
            displacements = {
                track_id: _displacement(centers)
                for track_id, centers in centers_by_track.items()
            }
            start_centers = {
                track_id: centers[0]
                for track_id, centers in centers_by_track.items()
                if centers
            }
            end_centers = {
                track_id: centers[-1]
                for track_id, centers in centers_by_track.items()
                if centers
            }
            start_group = _median_pair(list(start_centers.values()))
            end_group = _median_pair(list(end_centers.values()))

            raw: dict[str, dict[str, float]] = {}
            for obj in objects:
                track_id = str(obj["track_id"])
                other_displacements = [
                    displacement
                    for other_id, displacement in displacements.items()
                    if other_id != track_id
                ]
                median_other = _median_pair(other_displacements)
                motion_inconsistency = _distance(displacements.get(track_id, (0.0, 0.0)), median_other)
                centers = centers_by_track.get(track_id, [])
                if len(centers) >= 2:
                    group_leaving = max(
                        0.0,
                        _distance(centers[-1], end_group) - _distance(centers[0], start_group),
                    )
                else:
                    group_leaving = 0.0
                modal_offsets = [
                    float(state.get("modal", {}).get("offset_distance"))
                    for state in obj.get("states", [])
                    if state.get("modal") and state.get("modal", {}).get("offset_distance") is not None
                ]
                raw[track_id] = {
                    "motion_inconsistency": motion_inconsistency,
                    "group_leaving": group_leaving,
                    "modal_inconsistency": median(modal_offsets) if modal_offsets else 0.0,
                }

            normalized = {
                name: _robust({track_id: components[name] for track_id, components in raw.items()})
                for name in ("motion_inconsistency", "group_leaving", "modal_inconsistency")
            }
            for obj in objects:
                track_id = str(obj["track_id"])
                components = {
                    name: float(normalized[name].get(track_id, 0.0))
                    for name in ("motion_inconsistency", "group_leaving", "modal_inconsistency")
                }
                score = (
                    0.45 * components["motion_inconsistency"]
                    + 0.35 * components["group_leaving"]
                    + 0.20 * components["modal_inconsistency"]
                )
                yield GroupScoreRecord(
                    sequence=str(window["sequence"]),
                    track_id=track_id,
                    window_id=str(window["window_id"]),
                    frame_start=int(window["frame_start"]),
                    frame_end=int(window["frame_end"]),
                    category_id=obj.get("category_id"),
                    category_name=obj.get("category_name"),
                    score=float(score),
                    component_scores=components,
                    metadata={"detector": "lightweight_group_baseline", "raw_components": raw.get(track_id, {})},
                )


def score_group_windows_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
) -> dict[str, Any]:
    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)
    windows = iter_group_window_jsonl(input_jsonl)
    records = list(LightweightGroupAnomalyDetector().score_windows(windows))
    count = write_group_score_records_jsonl(output_jsonl, records)
    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "num_window_scores": count,
    }
