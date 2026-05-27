from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from fusiontrack.event_segments import event_segments_from_frame_scores
from fusiontrack.explanation_schema import build_explanation_schema


EPSILON = 1e-6
EVENT_THRESHOLD = 0.25
EVENT_MAX_GAP = 2
EVENT_MIN_LENGTH = 1


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _robust_normalize_component(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    vals = list(values.values())
    med = median(vals)
    abs_dev = [abs(value - med) for value in vals]
    mad = median(abs_dev)
    if mad <= EPSILON:
        ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
        denom = max(len(ordered) - 1, 1)
        return {sample_id: idx / denom for idx, (sample_id, _) in enumerate(ordered)}
    return {sample_id: max(0.0, (value - med) / (1.4826 * mad + EPSILON)) for sample_id, value in values.items()}


def _trajectory_components(trajectory: dict[str, Any]) -> dict[str, float]:
    points = sorted(
        [point for point in trajectory.get("points", []) if point.get("fused")],
        key=lambda item: item.get("frame_id", 0),
    )
    centers = [tuple(point["fused"]["center_xy"]) for point in points]
    confidences = [float(point["fused"].get("confidence", 0.0)) for point in points]
    offsets = [
        float(point["fused"].get("component_scores", {}).get("modal_offset_distance", 0.0))
        for point in points
    ]

    speeds: list[float] = []
    directions: list[float] = []
    for prev, curr in zip(centers, centers[1:]):
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        speeds.append(math.hypot(dx, dy))
        directions.append(math.atan2(dy, dx))

    turns = []
    for prev, curr in zip(directions, directions[1:]):
        delta = abs(curr - prev)
        turns.append(min(delta, 2 * math.pi - delta))

    speed_med = median(speeds) if speeds else 0.0
    speed_spike = max(0.0, _percentile(speeds, 0.95) - speed_med)
    turn_irregularity = _percentile(turns, 0.95)
    low_confidence_ratio = (
        sum(1 for value in confidences if value < 0.65) / len(confidences)
        if confidences
        else 0.0
    )
    modal_offset_median = median(offsets) if offsets else 0.0

    return {
        "speed_spike": float(speed_spike),
        "turn_irregularity": float(turn_irregularity),
        "low_confidence_ratio": float(low_confidence_ratio),
        "modal_offset_median": float(modal_offset_median),
    }


def _fused_points(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [point for point in trajectory.get("points", []) if point.get("fused")],
        key=lambda item: item.get("frame_id", 0),
    )


def _safe_center(point: dict[str, Any]) -> tuple[float, float] | None:
    center = point.get("fused", {}).get("center_xy")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    try:
        return float(center[0]), float(center[1])
    except (TypeError, ValueError):
        return None


def _normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    if max_value <= EPSILON:
        return [0.0 for _ in values]
    return [max(0.0, float(value) / max_value) for value in values]


def _dominant_component(components: dict[str, float]) -> str:
    if not components:
        return "trajectory_event"
    return max(components.items(), key=lambda item: (float(item[1]), item[0]))[0]


def _trajectory_frame_event_scores(
    trajectory: dict[str, Any],
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    points = _fused_points(trajectory)
    centers = [_safe_center(point) for point in points]
    speeds: list[float] = []
    for prev, curr in zip(centers, centers[1:]):
        if prev is None or curr is None:
            speeds.append(0.0)
        else:
            speeds.append(math.hypot(curr[0] - prev[0], curr[1] - prev[1]))

    directions: list[float | None] = []
    for prev, curr in zip(centers, centers[1:]):
        if prev is None or curr is None:
            directions.append(None)
        else:
            directions.append(math.atan2(curr[1] - prev[1], curr[0] - prev[0]))

    turns_by_point = [0.0 for _ in points]
    for index in range(2, len(points)):
        prev_direction = directions[index - 2] if index - 2 < len(directions) else None
        curr_direction = directions[index - 1] if index - 1 < len(directions) else None
        if prev_direction is None or curr_direction is None:
            continue
        delta = abs(curr_direction - prev_direction)
        turns_by_point[index] = min(delta, 2 * math.pi - delta) / math.pi

    speed_med = median(speeds) if speeds else 0.0
    speed_spikes = [0.0]
    speed_spikes.extend(max(0.0, speed - speed_med) for speed in speeds)
    speed_scores = _normalize_values(speed_spikes)
    modal_offsets = [
        float(point.get("fused", {}).get("component_scores", {}).get("modal_offset_distance", 0.0) or 0.0)
        for point in points
    ]
    modal_scores = _normalize_values(modal_offsets)

    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        try:
            frame = int(point.get("frame_id", index))
        except (TypeError, ValueError):
            continue
        confidence = float(point.get("fused", {}).get("confidence", 0.0) or 0.0)
        components = {
            "speed_spike": speed_scores[index] if index < len(speed_scores) else 0.0,
            "turn_irregularity": turns_by_point[index] if index < len(turns_by_point) else 0.0,
            "low_confidence_ratio": max(0.0, min(1.0, (0.65 - confidence) / 0.65)),
            "modal_offset_median": modal_scores[index] if index < len(modal_scores) else 0.0,
        }
        score = sum(float(weights[name]) * components[name] for name in weights)
        rows.append(
            {
                "frame": frame,
                "score": round(float(score), 6),
                "dominant_reason": _dominant_component(components),
                "component_scores": {name: round(float(value), 6) for name, value in components.items()},
                "source": "individual_simple_frame",
            }
        )
    return rows


def score_fused_trajectories_simple(
    fused_jsonl: str | Path,
    output_jsonl: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    fused_jsonl = Path(fused_jsonl)
    output_jsonl = Path(output_jsonl)
    output_csv = Path(output_csv)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    trajectories = _iter_jsonl(fused_jsonl)
    raw_components = {
        trajectory["sample_id"]: _trajectory_components(trajectory)
        for trajectory in trajectories
    }
    component_names = [
        "speed_spike",
        "turn_irregularity",
        "low_confidence_ratio",
        "modal_offset_median",
    ]
    normalized_by_component = {}
    for component_name in component_names:
        normalized_by_component[component_name] = _robust_normalize_component(
            {
                sample_id: components[component_name]
                for sample_id, components in raw_components.items()
            }
        )

    weights = {
        "speed_spike": 0.35,
        "turn_irregularity": 0.25,
        "low_confidence_ratio": 0.20,
        "modal_offset_median": 0.20,
    }
    records: list[dict[str, Any]] = []
    by_sample = {trajectory["sample_id"]: trajectory for trajectory in trajectories}
    for sample_id in sorted(raw_components):
        component_scores = {
            component_name: float(normalized_by_component[component_name][sample_id])
            for component_name in component_names
        }
        score = sum(weights[name] * component_scores[name] for name in component_names)
        trajectory = by_sample[sample_id]
        frame_event_scores = _trajectory_frame_event_scores(trajectory, weights)
        event_score = max((float(row["score"]) for row in frame_event_scores), default=0.0)
        event_segments = event_segments_from_frame_scores(
            frame_event_scores,
            threshold=EVENT_THRESHOLD,
            max_gap=EVENT_MAX_GAP,
            min_length=EVENT_MIN_LENGTH,
        )
        explanation_row = {
            "score": float(score),
            "event_score": float(event_score),
            "component_scores": component_scores,
            "frame_event_scores": frame_event_scores,
            "event_segments": event_segments,
        }
        records.append(
            {
                "sample_id": sample_id,
                "sequence": trajectory["sequence"],
                "track_id": trajectory["track_id"],
                "category_id": trajectory.get("category_id"),
                "category_name": trajectory.get("category_name"),
                "source": "individual_simple",
                "score": float(score),
                "event_score": float(event_score),
                "component_scores": component_scores,
                "frame_event_scores": frame_event_scores,
                "event_segments": event_segments,
                "explanation_schema": build_explanation_schema(
                    explanation_row,
                    threshold=EVENT_THRESHOLD,
                    max_gap=EVENT_MAX_GAP,
                    min_length=EVENT_MIN_LENGTH,
                ),
                "metadata": {
                    "detector": "simple_fused_trajectory",
                    "raw_components": raw_components[sample_id],
                },
            }
        )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "sequence",
            "track_id",
            "category_id",
            "category_name",
            "score",
            "event_score",
            *component_names,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "sequence": record["sequence"],
                    "track_id": record["track_id"],
                    "category_id": record["category_id"],
                    "category_name": record["category_name"],
                    "score": record["score"],
                    "event_score": record["event_score"],
                    **record["component_scores"],
                }
            )

    return {
        "input_jsonl": str(fused_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "num_scores": len(records),
    }
