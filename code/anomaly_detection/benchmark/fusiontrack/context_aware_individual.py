from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baselines.individual_features import (
    FEATURE_COLUMNS as INDIVIDUAL_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    build_handcrafted_feature_row,
)
from fusiontrack.group_graph import extract_object_states


CONTEXT_FEATURE_COLUMNS = (
    "context_num_neighbors",
    "context_nearest_distance_mean",
    "context_neighbor_distance_mean",
    "context_group_dispersion_mean",
    "context_isolation_ratio",
    "context_relative_speed_mean",
)
FEATURE_COLUMNS = INDIVIDUAL_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS
OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS
NEAR_DISTANCE_THRESHOLD = 10.0
NO_NEIGHBOR_DISTANCE = NEAR_DISTANCE_THRESHOLD * 2.0


def build_context_feature_table(windows: Iterable[dict]) -> pd.DataFrame:
    accumulators: dict[tuple[str, str, str], dict[str, Any]] = {}

    for window in windows:
        states = _states_with_velocity(extract_object_states(window))
        by_frame: dict[int, list[dict]] = {}
        for state in states:
            by_frame.setdefault(int(state["frame_id"]), []).append(state)

        for frame_states in by_frame.values():
            centroid = _centroid(frame_states)
            group_velocity = _mean_velocity(frame_states)
            dispersion = _mean(
                [
                    _distance(tuple(state["center_xy"]), centroid)
                    for state in frame_states
                ]
            )
            for state in frame_states:
                key = _state_key(state)
                accumulator = accumulators.setdefault(
                    key,
                    {
                        "sample_id": key[0],
                        "sequence": key[1],
                        "track_id": key[2],
                        "num_neighbors": [],
                        "nearest_distances": [],
                        "neighbor_distances": [],
                        "dispersions": [],
                        "is_isolated": [],
                        "relative_speeds": [],
                    },
                )
                distances = [
                    _distance(tuple(state["center_xy"]), tuple(other["center_xy"]))
                    for other in frame_states
                    if str(other["track_id"]) != str(state["track_id"])
                ]
                finite_distances = [distance for distance in distances if math.isfinite(distance)]
                near_count = sum(
                    1 for distance in finite_distances if distance <= NEAR_DISTANCE_THRESHOLD
                )
                nearest_distance = (
                    min(finite_distances) if finite_distances else NO_NEIGHBOR_DISTANCE
                )
                neighbor_distance = _mean(finite_distances) if finite_distances else NO_NEIGHBOR_DISTANCE
                velocity = _vector2(state.get("velocity")) or (0.0, 0.0)
                relative_speed = _distance(velocity, group_velocity)

                accumulator["num_neighbors"].append(float(near_count))
                accumulator["nearest_distances"].append(nearest_distance)
                accumulator["neighbor_distances"].append(neighbor_distance)
                accumulator["dispersions"].append(dispersion)
                accumulator["is_isolated"].append(
                    1.0
                    if not finite_distances or nearest_distance > NEAR_DISTANCE_THRESHOLD
                    else 0.0
                )
                accumulator["relative_speeds"].append(relative_speed)

    rows = []
    for accumulator in accumulators.values():
        rows.append(
            {
                "sample_id": accumulator["sample_id"],
                "sequence": str(accumulator["sequence"]),
                "track_id": str(accumulator["track_id"]),
                "context_num_neighbors": _mean(accumulator["num_neighbors"]),
                "context_nearest_distance_mean": _mean(accumulator["nearest_distances"]),
                "context_neighbor_distance_mean": _mean(accumulator["neighbor_distances"]),
                "context_group_dispersion_mean": _mean(accumulator["dispersions"]),
                "context_isolation_ratio": _mean(accumulator["is_isolated"]),
                "context_relative_speed_mean": _mean(accumulator["relative_speeds"]),
            }
        )

    table = pd.DataFrame(rows, columns=IDENTIFIER_COLUMNS + CONTEXT_FEATURE_COLUMNS)
    if table.empty:
        return table
    for column in CONTEXT_FEATURE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    return table.sort_values("sample_id", kind="stable").reset_index(drop=True)


def build_context_aware_feature_table(
    trajectories: Iterable[dict],
    windows: Iterable[dict],
) -> pd.DataFrame:
    individual_rows = [
        build_handcrafted_feature_row(trajectory) for trajectory in trajectories
    ]
    individual_table = pd.DataFrame(
        individual_rows,
        columns=IDENTIFIER_COLUMNS + INDIVIDUAL_FEATURE_COLUMNS,
    )
    if individual_table.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    context_table = build_context_feature_table(windows)
    context_by_sample_id = {
        str(row["sample_id"]): row for row in context_table.to_dict("records")
    }
    context_by_sequence_track = {
        (str(row["sequence"]), str(row["track_id"])): row
        for row in context_table.to_dict("records")
    }

    rows: list[dict[str, Any]] = []
    for individual_row in individual_table.to_dict("records"):
        row = dict(individual_row)
        context_row = context_by_sample_id.get(str(row["sample_id"]))
        if context_row is None:
            context_row = context_by_sequence_track.get(
                (str(row["sequence"]), str(row["track_id"]))
            )
        for column in CONTEXT_FEATURE_COLUMNS:
            row[column] = _finite_float(context_row.get(column, 0.0) if context_row else 0.0)
        rows.append(row)

    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for column in FEATURE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    return table.reset_index(drop=True)


def run_context_aware_fusiontrack_baseline(
    train_trajectories: Iterable[dict],
    score_trajectories: Iterable[dict],
    train_windows: Iterable[dict],
    score_windows: Iterable[dict],
    n_neighbors: int = 1,
) -> list[dict[str, Any]]:
    train_features = build_context_aware_feature_table(train_trajectories, train_windows)
    score_features = build_context_aware_feature_table(score_trajectories, score_windows)
    model = _fit_nearest_profile(train_features, n_neighbors=n_neighbors)
    scores = _score_nearest_profile(model, score_features)

    rows: list[dict[str, Any]] = []
    for (_, feature_row), score in zip(score_features.iterrows(), scores):
        rows.append(
            {
                "sample_id": str(feature_row["sample_id"]),
                "sequence": str(feature_row["sequence"]),
                "track_id": str(feature_row["track_id"]),
                "source": "fusiontrack_individual:context_aware",
                "score": float(score),
                "component_scores": {"context_aware_distance": float(score)},
                "metadata": {
                    "method": "context_aware_nearest_feature",
                    "feature_columns": list(FEATURE_COLUMNS),
                    "individual_feature_columns": list(INDIVIDUAL_FEATURE_COLUMNS),
                    "context_feature_columns": list(CONTEXT_FEATURE_COLUMNS),
                    "n_neighbors": max(1, int(n_neighbors)),
                },
            }
        )
    return rows


def _fit_nearest_profile(feature_df: pd.DataFrame, n_neighbors: int) -> Pipeline:
    features = _feature_matrix(feature_df)
    if len(features) == 0:
        raise ValueError("Cannot fit context-aware nearest profile with no feature rows")
    neighbors = max(1, min(int(n_neighbors), len(features)))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("nearest_neighbors", NearestNeighbors(n_neighbors=neighbors)),
        ]
    )
    return model.fit(features)


def _score_nearest_profile(model: Pipeline, feature_df: pd.DataFrame) -> list[float]:
    features = _feature_matrix(feature_df)
    if len(features) == 0:
        return []
    scaled_features = model.named_steps["scaler"].transform(features)
    distances, _ = model.named_steps["nearest_neighbors"].kneighbors(scaled_features)
    raw_scores = np.mean(distances, axis=1)
    return [float(score) if np.isfinite(score) else 0.0 for score in raw_scores]


def _feature_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    matrix = feature_df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _states_with_velocity(states: list[dict]) -> list[dict]:
    enriched = [dict(state) for state in states]
    by_track: dict[tuple[str, str], list[dict]] = {}
    for state in enriched:
        by_track.setdefault(
            (str(state.get("sequence", "")), str(state["track_id"])),
            [],
        ).append(state)

    for track_states in by_track.values():
        track_states.sort(key=lambda state: int(state["frame_id"]))
        previous: dict | None = None
        for state in track_states:
            if previous is None:
                state["velocity"] = _vector2(state.get("velocity")) or (0.0, 0.0)
            else:
                state["velocity"] = (
                    float(state["center_xy"][0]) - float(previous["center_xy"][0]),
                    float(state["center_xy"][1]) - float(previous["center_xy"][1]),
                )
            previous = state
    return enriched


def _state_key(state: dict) -> tuple[str, str, str]:
    return (
        str(state["sample_id"]),
        str(state.get("sequence", "")),
        str(state["track_id"]),
    )


def _centroid(states: list[dict]) -> tuple[float, float]:
    if not states:
        return 0.0, 0.0
    return (
        _mean([float(state["center_xy"][0]) for state in states]),
        _mean([float(state["center_xy"][1]) for state in states]),
    )


def _mean_velocity(states: list[dict]) -> tuple[float, float]:
    velocities = [_vector2(state.get("velocity")) or (0.0, 0.0) for state in states]
    if not velocities:
        return 0.0, 0.0
    return _mean([velocity[0] for velocity in velocities]), _mean(
        [velocity[1] for velocity in velocities]
    )


def _vector2(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x = float(value[0])
            y = float(value[1])
        except (TypeError, ValueError):
            return None
        if math.isfinite(x) and math.isfinite(y):
            return x, y
    return None


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(math.hypot(second[0] - first[0], second[1] - first[1]))


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0
