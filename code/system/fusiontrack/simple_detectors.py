from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


EPSILON = 1e-6


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
        records.append(
            {
                "sample_id": sample_id,
                "sequence": trajectory["sequence"],
                "track_id": trajectory["track_id"],
                "category_id": trajectory.get("category_id"),
                "category_name": trajectory.get("category_name"),
                "source": "individual_simple",
                "score": float(score),
                "component_scores": component_scores,
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
                    **record["component_scores"],
                }
            )

    return {
        "input_jsonl": str(fused_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "num_scores": len(records),
    }
