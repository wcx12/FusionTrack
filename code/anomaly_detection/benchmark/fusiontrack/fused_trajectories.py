from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable


MODALITY_ORDER = ("rgb", "thermal", "modal")


def _center_xy(state: Any) -> list[float] | None:
    if not isinstance(state, dict):
        return None
    center = state.get("center_xy")
    if center is None or len(center) != 2:
        return None
    return [float(center[0]), float(center[1])]


def _modal_offset_distance(point: dict[str, Any], centers: list[list[float]]) -> float:
    modal_state = point.get("modal")
    if isinstance(modal_state, dict) and modal_state.get("modal_offset_distance") is not None:
        return float(modal_state["modal_offset_distance"])
    if point.get("modal_offset_distance") is not None:
        return float(point["modal_offset_distance"])
    if len(centers) < 2:
        return 0.0
    first, second = centers[0], centers[1]
    return math.dist(first, second)


def fuse_state(point: dict[str, Any], offset_scale: float = 25.0) -> dict[str, Any] | None:
    centers_by_modality = [
        (modality, center)
        for modality in MODALITY_ORDER
        if (center := _center_xy(point.get(modality))) is not None
    ]
    if not centers_by_modality:
        return None

    source_modalities = [modality for modality, _ in centers_by_modality]
    centers = [center for _, center in centers_by_modality]

    if len(centers) == 1:
        center_xy = centers[0]
        confidence = 0.55
        modal_offset = 0.0
    else:
        center_xy = [
            sum(center[0] for center in centers) / len(centers),
            sum(center[1] for center in centers) / len(centers),
        ]
        modal_offset = _modal_offset_distance(point, centers)
        safe_scale = offset_scale if offset_scale > 0.0 else 25.0
        confidence = 1.0 / (1.0 + modal_offset / safe_scale)

    weight = 1.0 / len(source_modalities)
    component_scores = {
        "modal_offset_distance": modal_offset,
        **{f"{modality}_weight": weight for modality in source_modalities},
    }

    return {
        "center_xy": center_xy,
        "confidence": confidence,
        "source_modalities": source_modalities,
        "component_scores": component_scores,
    }


def build_fused_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    fused_trajectory: dict[str, Any] = {}
    for key in ("sample_id", "sequence", "track_id", "category_id", "category_name"):
        if key in trajectory:
            fused_trajectory[key] = copy.deepcopy(trajectory[key])

    fused_points: list[dict[str, Any]] = []
    for point in trajectory.get("points", []):
        fused_point = copy.deepcopy(point)
        fused_point["fused"] = fuse_state(point)
        fused_points.append(fused_point)
    fused_trajectory["points"] = fused_points
    return fused_trajectory


def build_fused_trajectories(trajectories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [build_fused_trajectory(trajectory) for trajectory in trajectories]


def write_fused_trajectories_jsonl(
    path: str | Path,
    trajectories: Iterable[dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
