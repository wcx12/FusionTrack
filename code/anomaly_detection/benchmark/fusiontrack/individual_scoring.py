from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def _feature_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    matrix = feature_df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
