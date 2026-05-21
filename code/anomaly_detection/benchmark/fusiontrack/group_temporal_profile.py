from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baselines.group_prediction import run_prediction_baseline
from baselines.group_features import FEATURE_COLUMNS, build_group_feature_table
from fusiontrack.group_scoring import score_group_windows
from protocol.schemas import build_sample_id


def fit_group_temporal_knn(
    train_windows: Iterable[dict],
    n_neighbors: int = 3,
) -> dict[str, Any]:
    feature_df = build_group_feature_table(train_windows)
    features = _feature_matrix(feature_df)
    if len(features) == 0:
        raise ValueError("Cannot fit group temporal KNN with no training windows")

    k = max(1, min(int(n_neighbors), len(features)))
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("neighbors", NearestNeighbors(n_neighbors=k, metric="euclidean")),
        ]
    )
    pipeline.fit(features)
    return {
        "pipeline": pipeline,
        "n_neighbors": k,
        "feature_columns": list(FEATURE_COLUMNS),
    }


def score_group_temporal_knn(
    model: dict[str, Any],
    score_windows: Iterable[dict],
) -> tuple[list[float], pd.DataFrame]:
    windows = list(score_windows)
    feature_df = build_group_feature_table(windows, sort_by_window_id=False)
    features = _feature_matrix(feature_df)
    if len(features) == 0:
        return [], feature_df

    pipeline = model["pipeline"]
    distances, _ = pipeline.named_steps["neighbors"].kneighbors(
        pipeline.named_steps["scaler"].transform(features)
    )
    scores = np.mean(np.asarray(distances, dtype=float), axis=1)
    return [
        float(score) if np.isfinite(score) else 0.0
        for score in scores
    ], feature_df


def run_group_temporal_knn(
    train_windows: Iterable[dict],
    score_windows: Iterable[dict],
    n_neighbors: int = 3,
) -> list[dict[str, Any]]:
    model = fit_group_temporal_knn(train_windows, n_neighbors=n_neighbors)
    score_windows = list(score_windows)
    scores, feature_df = score_group_temporal_knn(model, score_windows)

    rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(score_windows):
        window_id = str(window.get("window_id", window.get("sample_id", "")))
        score = float(scores[window_index]) if window_index < len(scores) else 0.0
        frame_start, frame_end = _window_frame_bounds(window)
        feature_row = (
            feature_df.iloc[window_index].to_dict()
            if window_index < len(feature_df)
            else {}
        )
        component_scores = _component_scores(score, feature_row)
        for obj in _window_objects(window):
            sequence = str(window.get("sequence", ""))
            track_id = str(obj["track_id"])
            rows.append(
                {
                    "sample_id": _sample_id(obj, sequence, track_id),
                    "window_id": window_id,
                    "sequence": sequence,
                    "track_id": track_id,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "source": "fusiontrack_group_temporal_knn",
                    "score": score,
                    "component_scores": dict(component_scores),
                    "metadata": {
                        "method": "fusiontrack_group_temporal_knn",
                        "n_neighbors": int(model["n_neighbors"]),
                        "feature_columns": list(model["feature_columns"]),
                        "window_id": window_id,
                    },
                }
            )
    return rows


def run_group_hybrid_fusiontrack(
    train_windows: Iterable[dict],
    score_windows: Iterable[dict],
    n_neighbors: int = 3,
    k_neighbors: int = 3,
    rho_p: float = 80.0,
    rho_v: float = 20.0,
    eta: float = 0.5,
    prediction_weight: float = 0.6,
    graph_weight: float = 0.2,
    temporal_weight: float = 0.2,
    invert_graph_rank: bool = True,
    invert_temporal_rank: bool = True,
) -> list[dict[str, Any]]:
    train_windows = list(train_windows)
    score_windows = list(score_windows)

    prediction_rows = run_prediction_baseline(score_windows)
    graph_by_key = _rows_by_key(
        score_group_windows(
            score_windows,
            k_neighbors=k_neighbors,
            rho_p=rho_p,
            rho_v=rho_v,
            eta=eta,
        )
    )
    temporal_by_key = _rows_by_key(
        run_group_temporal_knn(
            train_windows,
            score_windows,
            n_neighbors=n_neighbors,
        )
    )

    keys = [_row_key(row) for row in prediction_rows]
    prediction_rank = _rank01([row.get("score", 0.0) for row in prediction_rows])
    graph_rank = _rank01(
        [
            graph_by_key.get(key, {}).get("score", 0.0)
            for key in keys
        ],
        inverse=invert_graph_rank,
    )
    temporal_rank = _rank01(
        [
            temporal_by_key.get(key, {}).get("score", 0.0)
            for key in keys
        ],
        inverse=invert_temporal_rank,
    )

    weights = np.asarray(
        [prediction_weight, graph_weight, temporal_weight],
        dtype=float,
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.asarray([0.6, 0.2, 0.2], dtype=float)
    weights = weights / float(weights.sum())

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(prediction_rows):
        key = keys[index]
        graph_row = graph_by_key.get(key, {})
        temporal_row = temporal_by_key.get(key, {})
        component_scores = {
            "prediction_residual_rank": float(prediction_rank[index]),
            "graph_rank": float(graph_rank[index]),
            "temporal_profile_rank": float(temporal_rank[index]),
        }
        score = (
            float(weights[0]) * component_scores["prediction_residual_rank"]
            + float(weights[1]) * component_scores["graph_rank"]
            + float(weights[2]) * component_scores["temporal_profile_rank"]
        )
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "window_id": str(row["window_id"]),
                "sequence": str(row.get("sequence", "")),
                "track_id": str(row.get("track_id", "")),
                "frame_start": row.get("frame_start"),
                "frame_end": row.get("frame_end"),
                "source": "fusiontrack_group_hybrid",
                "score": float(score) if np.isfinite(score) else 0.0,
                "component_scores": component_scores,
                "metadata": {
                    "method": "fusiontrack_group_hybrid",
                    "window_id": str(row["window_id"]),
                    "n_neighbors": int(n_neighbors),
                    "k_neighbors": int(k_neighbors),
                    "rho_p": float(rho_p),
                    "rho_v": float(rho_v),
                    "eta": float(eta),
                    "rank_transforms": {
                        "prediction_residual_rank": "rank",
                        "graph_rank": "inverse_rank" if invert_graph_rank else "rank",
                        "temporal_profile_rank": (
                            "inverse_rank" if invert_temporal_rank else "rank"
                        ),
                    },
                    "weights": {
                        "prediction_residual_rank": float(weights[0]),
                        "graph_rank": float(weights[1]),
                        "temporal_profile_rank": float(weights[2]),
                    },
                    "raw_scores": {
                        "prediction_residual": float(row.get("score", 0.0) or 0.0),
                        "graph": float(graph_row.get("score", 0.0) or 0.0),
                        "temporal_profile": float(temporal_row.get("score", 0.0) or 0.0),
                    },
                },
            }
        )
    return rows


def _feature_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    matrix = feature_df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _component_scores(score: float, feature_row: dict[str, Any]) -> dict[str, float]:
    return {
        "temporal_profile_distance": float(score),
        "mean_speed": _float_feature(feature_row, "mean_speed"),
        "mean_dispersion": _float_feature(feature_row, "mean_dispersion"),
        "neighbor_churn": _float_feature(feature_row, "neighbor_churn"),
    }


def _float_feature(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


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


def _window_objects(window: dict) -> list[dict]:
    objects = [
        obj
        for obj in window.get("objects", [])
        if obj.get("track_id") not in (None, "")
    ]
    return sorted(objects, key=lambda obj: str(obj["track_id"]))


def _sample_id(obj: dict, sequence: str, track_id: str) -> str:
    sample_id = obj.get("sample_id")
    if sample_id not in (None, ""):
        return str(sample_id)
    return build_sample_id(sequence, track_id)


def _rows_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {_row_key(row): row for row in rows}


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("sample_id", "")), str(row.get("window_id", ""))


def _rank01(values: Iterable[float], inverse: bool = False) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return []
    array = np.where(np.isfinite(array), array, 0.0)
    if float(np.max(array) - np.min(array)) <= 0.0:
        return [0.0 for _ in array]
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, num=len(array))
    if inverse:
        ranks = 1.0 - ranks
    return [float(value) for value in ranks]
