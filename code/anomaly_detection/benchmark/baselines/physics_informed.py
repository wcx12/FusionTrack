from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from baselines.individual_features import extract_center_sequence


IDENTIFIER_COLUMNS = ("sample_id", "sequence", "track_id")
FEATURE_COLUMNS = (
    "duration_frames",
    "num_points",
    "path_length",
    "displacement",
    "mean_speed",
    "max_speed",
    "speed_variance",
    "stop_ratio",
    "mean_acceleration",
    "max_acceleration",
    "mean_jerk",
    "max_jerk",
    "mean_turn_angle",
    "max_turn_angle",
    "path_efficiency",
    "smoothness_residual",
)
OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS
PROFILE_FEATURES = (
    "mean_speed",
    "max_speed",
    "speed_variance",
    "stop_ratio",
    "mean_acceleration",
    "max_acceleration",
    "mean_jerk",
    "max_jerk",
    "mean_turn_angle",
    "max_turn_angle",
    "path_efficiency",
    "smoothness_residual",
)
COMPONENT_FEATURES = {
    "speed": ("mean_speed", "max_speed", "speed_variance", "stop_ratio"),
    "acceleration": ("mean_acceleration", "max_acceleration"),
    "jerk": ("mean_jerk", "max_jerk"),
    "turn": ("mean_turn_angle", "max_turn_angle"),
    "smoothness": ("path_efficiency", "smoothness_residual"),
}
SOURCE = "individual_physics:kinematic_prior"


def build_physics_feature_row(trajectory: dict) -> dict[str, Any]:
    sequence = extract_center_sequence(trajectory)
    frames = [frame_id for frame_id, _, _ in sequence]
    centers = [(x, y) for _, x, y in sequence]
    step_distances = _step_distances(sequence)
    speeds = _step_speeds(sequence)
    accelerations = _differences(speeds)
    jerks = _differences(accelerations)
    turn_angles = _turn_angles(centers)
    path_length = _sum(step_distances)
    displacement = _distance(centers[0], centers[-1]) if len(centers) >= 2 else 0.0
    path_efficiency = displacement / path_length if path_length > 0.0 else 1.0
    path_efficiency = max(0.0, min(1.0, path_efficiency))

    row: dict[str, Any] = {
        "sample_id": _sample_id(trajectory),
        "sequence": str(trajectory.get("sequence", "")),
        "track_id": str(trajectory.get("track_id", "")),
        "duration_frames": int(frames[-1] - frames[0] + 1) if frames else 0,
        "num_points": int(len(sequence)),
        "path_length": path_length,
        "displacement": displacement,
        "mean_speed": _mean(speeds),
        "max_speed": _max(speeds),
        "speed_variance": _variance(speeds),
        "stop_ratio": _stop_ratio(speeds),
        "mean_acceleration": _mean(accelerations),
        "max_acceleration": _max(accelerations),
        "mean_jerk": _mean(jerks),
        "max_jerk": _max(jerks),
        "mean_turn_angle": _mean(turn_angles),
        "max_turn_angle": _max(turn_angles),
        "path_efficiency": path_efficiency,
        "smoothness_residual": 1.0 - path_efficiency,
    }
    return {key: _finite_default(value) for key, value in row.items()}


def build_physics_feature_table(
    trajectories: Iterable[dict],
    sort_by_sample_id: bool = True,
) -> pd.DataFrame:
    rows = [build_physics_feature_row(trajectory) for trajectory in trajectories]
    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if table.empty:
        return table
    for column in FEATURE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    if sort_by_sample_id:
        table = table.sort_values("sample_id", kind="stable")
    return table.reset_index(drop=True)


def fit_physics_profile(train_trajectories: Iterable[dict], eps: float = 1e-3) -> dict[str, Any]:
    table = build_physics_feature_table(train_trajectories)
    if table.empty:
        raise ValueError("train_trajectories must contain at least one trajectory")

    features = table.loc[:, PROFILE_FEATURES].astype(float)
    medians = features.median(axis=0)
    mad = (features - medians).abs().median(axis=0)
    fallback_scales = features.abs().median(axis=0).clip(lower=1.0) * eps
    scales = (1.4826 * mad).where(mad > 0.0, fallback_scales).clip(lower=eps)
    return {
        "profile": "median_mad",
        "feature_columns": list(PROFILE_FEATURES),
        "median": medians.to_dict(),
        "scale": scales.to_dict(),
        "eps": float(eps),
    }


def score_physics_profile(model: dict[str, Any], score_trajectories: Iterable[dict]) -> list[dict]:
    table = build_physics_feature_table(score_trajectories, sort_by_sample_id=False)
    results: list[dict] = []
    for _, row in table.iterrows():
        residuals = {
            feature: _robust_residual(float(row[feature]), model, feature)
            for feature in model["feature_columns"]
        }
        component_scores = {
            component: _mean([residuals[feature] for feature in features])
            for component, features in COMPONENT_FEATURES.items()
        }
        score = _mean(list(component_scores.values()))
        results.append(
            {
                "sample_id": str(row["sample_id"]),
                "sequence": str(row["sequence"]),
                "track_id": str(row["track_id"]),
                "source": SOURCE,
                "score": _finite_float(score),
                "component_scores": {
                    key: _finite_float(value) for key, value in component_scores.items()
                },
                "metadata": {
                    "feature_columns": list(model["feature_columns"]),
                    "profile": model["profile"],
                },
            }
        )
    return results


def run_physics_informed_baseline(
    train_trajectories: Iterable[dict],
    score_trajectories: Iterable[dict],
) -> list[dict]:
    model = fit_physics_profile(train_trajectories)
    return score_physics_profile(model, score_trajectories)


def _sample_id(trajectory: dict) -> str:
    sample_id = trajectory.get("sample_id")
    if sample_id not in (None, ""):
        return str(sample_id)
    sequence = str(trajectory.get("sequence", ""))
    track_id = str(trajectory.get("track_id", ""))
    return f"{sequence}:{track_id}" if sequence or track_id else ""


def _step_distances(sequence: list[tuple[int, float, float]]) -> list[float]:
    return [
        _distance((x0, y0), (x1, y1))
        for (_, x0, y0), (_, x1, y1) in zip(sequence, sequence[1:])
    ]


def _step_speeds(sequence: list[tuple[int, float, float]]) -> list[float]:
    speeds: list[float] = []
    for (frame0, x0, y0), (frame1, x1, y1) in zip(sequence, sequence[1:]):
        delta_frames = max(frame1 - frame0, 1)
        speeds.append(_distance((x0, y0), (x1, y1)) / float(delta_frames))
    return speeds


def _differences(values: list[float]) -> list[float]:
    return [abs(next_value - value) for value, next_value in zip(values, values[1:])]


def _turn_angles(centers: list[tuple[float, float]]) -> list[float]:
    angles: list[float] = []
    for start, middle, end in zip(centers, centers[1:], centers[2:]):
        vector_a = (middle[0] - start[0], middle[1] - start[1])
        vector_b = (end[0] - middle[0], end[1] - middle[1])
        norm_a = math.hypot(*vector_a)
        norm_b = math.hypot(*vector_b)
        if norm_a == 0.0 or norm_b == 0.0:
            continue
        cosine = (
            (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1])
            / (norm_a * norm_b)
        )
        angles.append(math.acos(max(-1.0, min(1.0, cosine))))
    return angles


def _robust_residual(value: float, model: dict[str, Any], feature: str) -> float:
    median = float(model["median"][feature])
    scale = float(model["scale"][feature])
    return abs(value - median) / scale


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(math.hypot(second[0] - first[0], second[1] - first[1]))


def _sum(values: list[float]) -> float:
    return float(np.sum(values)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _variance(values: list[float]) -> float:
    return float(np.var(values)) if values else 0.0


def _stop_ratio(speeds: list[float], threshold: float = 1e-6) -> float:
    if not speeds:
        return 0.0
    return float(np.mean([speed <= threshold for speed in speeds]))


def _finite_default(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value


def _finite_float(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0
