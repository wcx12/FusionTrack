from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baselines.individual_classical import (
    fit_classical_detector,
    score_classical_detector,
)
from baselines.individual_features import (
    FEATURE_COLUMNS,
    build_handcrafted_feature_table,
)


def fit_nearest_feature_profile(
    feature_df: pd.DataFrame,
    n_neighbors: int = 1,
) -> Pipeline:
    features = _feature_matrix(feature_df)
    if len(features) == 0:
        raise ValueError("Cannot fit nearest feature profile with no feature rows")
    neighbors = max(1, min(int(n_neighbors), len(features)))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("nearest_neighbors", NearestNeighbors(n_neighbors=neighbors)),
        ]
    )
    return model.fit(features)


def score_nearest_feature_profile(
    model: Pipeline,
    feature_df: pd.DataFrame,
) -> list[float]:
    features = _feature_matrix(feature_df)
    if len(features) == 0:
        return []
    scaled_features = model.named_steps["scaler"].transform(features)
    distances, _ = model.named_steps["nearest_neighbors"].kneighbors(scaled_features)
    raw_scores = np.mean(distances, axis=1)
    return [float(score) if np.isfinite(score) else 0.0 for score in raw_scores]


def run_individual_fusiontrack_baseline(
    train_trajectories: Iterable[dict],
    score_trajectories: Iterable[dict],
    n_neighbors: int = 1,
) -> list[dict[str, Any]]:
    train_features = build_handcrafted_feature_table(train_trajectories)
    score_features = build_handcrafted_feature_table(score_trajectories)
    model = fit_nearest_feature_profile(train_features, n_neighbors=n_neighbors)
    scores = score_nearest_feature_profile(model, score_features)

    rows: list[dict[str, Any]] = []
    for (_, feature_row), score in zip(score_features.iterrows(), scores):
        rows.append(
            {
                "sample_id": str(feature_row["sample_id"]),
                "sequence": str(feature_row["sequence"]),
                "track_id": str(feature_row["track_id"]),
                "source": "fusiontrack_individual:nearest_feature",
                "score": float(score),
                "component_scores": {"nearest_feature_distance": float(score)},
                "metadata": {
                    "method": "nearest_feature",
                    "n_neighbors": max(1, int(n_neighbors)),
                    "feature_columns": list(FEATURE_COLUMNS),
                },
            }
        )
    return rows


def run_individual_fusiontrack_ensemble(
    train_trajectories: Iterable[dict],
    score_trajectories: Iterable[dict],
    n_neighbors: int = 1,
    seed: int = 42,
    contamination: float = 0.05,
    nearest_weight: float = 0.4,
    lof_weight: float = 0.35,
    iforest_weight: float = 0.25,
) -> list[dict[str, Any]]:
    train_features = build_handcrafted_feature_table(train_trajectories)
    score_features = build_handcrafted_feature_table(score_trajectories)
    sample_ids = score_features["sample_id"].astype(str).tolist()

    nearest_model = fit_nearest_feature_profile(
        train_features,
        n_neighbors=n_neighbors,
    )
    nearest_scores = score_nearest_feature_profile(nearest_model, score_features)
    nearest_rank = _rank01(nearest_scores)

    lof_scores = _classical_component_scores(
        train_features,
        score_features,
        method="lof",
        seed=seed,
        contamination=contamination,
    )
    iforest_scores = _classical_component_scores(
        train_features,
        score_features,
        method="isolation_forest",
        seed=seed,
        contamination=contamination,
    )
    lof_rank = _rank01([lof_scores.get(sample_id, 0.0) for sample_id in sample_ids])
    iforest_rank = _rank01([iforest_scores.get(sample_id, 0.0) for sample_id in sample_ids])

    weights = np.asarray(
        [nearest_weight, lof_weight, iforest_weight],
        dtype=float,
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.asarray([0.4, 0.35, 0.25], dtype=float)
    weights = weights / float(weights.sum())

    rows: list[dict[str, Any]] = []
    for index, (_, feature_row) in enumerate(score_features.iterrows()):
        component_scores = {
            "nearest_feature_rank": float(nearest_rank[index]),
            "lof_novelty_rank": float(lof_rank[index]),
            "isolation_forest_rank": float(iforest_rank[index]),
        }
        score = (
            float(weights[0]) * component_scores["nearest_feature_rank"]
            + float(weights[1]) * component_scores["lof_novelty_rank"]
            + float(weights[2]) * component_scores["isolation_forest_rank"]
        )
        rows.append(
            {
                "sample_id": str(feature_row["sample_id"]),
                "sequence": str(feature_row["sequence"]),
                "track_id": str(feature_row["track_id"]),
                "source": "fusiontrack_individual:ensemble",
                "score": float(score) if np.isfinite(score) else 0.0,
                "component_scores": component_scores,
                "metadata": {
                    "method": "fusiontrack_individual_ensemble",
                    "n_neighbors": max(1, int(n_neighbors)),
                    "seed": int(seed),
                    "contamination": float(contamination),
                    "feature_columns": list(FEATURE_COLUMNS),
                    "weights": {
                        "nearest_feature_rank": float(weights[0]),
                        "lof_novelty_rank": float(weights[1]),
                        "isolation_forest_rank": float(weights[2]),
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


def _classical_component_scores(
    train_features: pd.DataFrame,
    score_features: pd.DataFrame,
    method: str,
    seed: int,
    contamination: float,
) -> dict[str, float]:
    if train_features.empty or score_features.empty:
        return {}
    if method == "lof" and len(train_features) < 2:
        return {str(row["sample_id"]): 0.0 for _, row in score_features.iterrows()}
    model = fit_classical_detector(
        train_features,
        method=method,
        seed=seed,
        contamination=contamination,
    )
    return score_classical_detector(model, score_features, method=method)


def _rank01(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return []
    array = np.where(np.isfinite(array), array, 0.0)
    if float(np.max(array) - np.min(array)) <= 0.0:
        return [0.0 for _ in array]
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, num=len(array))
    return [float(value) for value in ranks]
