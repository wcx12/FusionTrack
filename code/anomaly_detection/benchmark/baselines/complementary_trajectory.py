from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baselines.individual_features import build_handcrafted_feature_table


ROUTE_FEATURES = ("duration_frames", "num_points", "path_length", "displacement")
SPEED_FEATURES = (
    "mean_speed",
    "max_speed",
    "std_speed",
    "mean_acceleration",
    "max_acceleration",
)
SHAPE_FEATURES = (
    "mean_turn_angle",
    "max_turn_angle",
    "bbox_area_mean",
    "bbox_area_std",
)
MODAL_FEATURES = ("modal_offset_mean", "modal_offset_max")
FEATURE_SUBSPACES = {
    "route": ROUTE_FEATURES,
    "speed": SPEED_FEATURES,
    "shape": SHAPE_FEATURES,
    "modal": MODAL_FEATURES,
}


def run_complementary_baseline(
    train_trajectories: Iterable[dict],
    score_trajectories: Iterable[dict],
    seed: int = 42,
    contamination: float = 0.05,
    n_neighbors: int = 1,
) -> list[dict[str, Any]]:
    train_features = build_handcrafted_feature_table(train_trajectories)
    score_features = build_handcrafted_feature_table(score_trajectories)
    if score_features.empty:
        return []

    contamination = _bounded_contamination(contamination)
    n_neighbors = int(n_neighbors)
    _validate_training_rows(train_features, n_neighbors)

    raw_components = {
        "route": _route_scores(train_features, score_features, n_neighbors),
        "speed": _speed_scores(train_features, score_features, contamination, seed),
        "shape": _shape_scores(train_features, score_features, contamination, n_neighbors),
        "modal": _modal_scores(train_features, score_features),
    }
    ranked_components = {
        name: _rank_normalize(scores) for name, scores in raw_components.items()
    }
    fused_scores = _mean_rank_scores(ranked_components)

    rows: list[dict[str, Any]] = []
    for row_index, feature_row in score_features.reset_index(drop=True).iterrows():
        component_scores = {
            name: float(scores[row_index]) for name, scores in ranked_components.items()
        }
        rows.append(
            {
                "sample_id": str(feature_row["sample_id"]),
                "sequence": str(feature_row["sequence"]),
                "track_id": str(feature_row["track_id"]),
                "source": "individual_complementary:cetrajad_style",
                "score": float(fused_scores[row_index]),
                "component_scores": component_scores,
                "metadata": {
                    "seed": int(seed),
                    "contamination": float(contamination),
                    "n_neighbors": int(n_neighbors),
                    "feature_columns": {
                        name: list(columns)
                        for name, columns in FEATURE_SUBSPACES.items()
                    },
                },
            }
        )
    return rows


def _route_scores(
    train_features: pd.DataFrame,
    score_features: pd.DataFrame,
    n_neighbors: int,
) -> np.ndarray:
    train_matrix = _feature_matrix(train_features, ROUTE_FEATURES)
    score_matrix = _feature_matrix(score_features, ROUTE_FEATURES)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("detector", NearestNeighbors(n_neighbors=n_neighbors)),
        ]
    )
    model.fit(train_matrix)
    distances, _ = model.named_steps["detector"].kneighbors(
        model.named_steps["scaler"].transform(score_matrix)
    )
    return _finite_array(np.mean(distances, axis=1))


def _speed_scores(
    train_features: pd.DataFrame,
    score_features: pd.DataFrame,
    contamination: float,
    seed: int,
) -> np.ndarray:
    train_matrix = _feature_matrix(train_features, SPEED_FEATURES)
    score_matrix = _feature_matrix(score_features, SPEED_FEATURES)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "detector",
                IsolationForest(
                    contamination=contamination,
                    random_state=int(seed),
                    n_estimators=100,
                ),
            ),
        ]
    )
    model.fit(train_matrix)
    return _finite_array(-np.asarray(model.decision_function(score_matrix), dtype=float))


def _shape_scores(
    train_features: pd.DataFrame,
    score_features: pd.DataFrame,
    contamination: float,
    n_neighbors: int,
) -> np.ndarray:
    train_matrix = _feature_matrix(train_features, SHAPE_FEATURES)
    score_matrix = _feature_matrix(score_features, SHAPE_FEATURES)
    shape_neighbors = max(1, min(n_neighbors, len(train_matrix) - 1))
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "detector",
                LocalOutlierFactor(
                    n_neighbors=shape_neighbors,
                    contamination=contamination,
                    novelty=True,
                ),
            ),
        ]
    )
    model.fit(train_matrix)
    return _finite_array(-np.asarray(model.decision_function(score_matrix), dtype=float))


def _modal_scores(
    train_features: pd.DataFrame,
    score_features: pd.DataFrame,
) -> np.ndarray:
    train_matrix = _feature_matrix(train_features, MODAL_FEATURES)
    score_matrix = _feature_matrix(score_features, MODAL_FEATURES)
    if train_matrix.empty or score_matrix.empty:
        return np.zeros(len(score_features), dtype=float)
    if float(np.abs(train_matrix.to_numpy()).sum() + np.abs(score_matrix.to_numpy()).sum()) == 0.0:
        return np.zeros(len(score_features), dtype=float)

    median = train_matrix.median(axis=0)
    mad = (train_matrix - median).abs().median(axis=0)
    usable = mad > 0.0
    if not bool(usable.any()):
        return np.zeros(len(score_features), dtype=float)

    z_scores = ((score_matrix.loc[:, usable] - median.loc[usable]).abs() / mad.loc[usable])
    return _finite_array(z_scores.mean(axis=1).to_numpy(dtype=float))


def _validate_training_rows(train_features: pd.DataFrame, n_neighbors: int) -> None:
    train_count = len(train_features)
    if train_count == 0:
        raise ValueError("individual_complementary requires at least 1 training row")
    if n_neighbors < 1:
        raise ValueError("individual_complementary n_neighbors must be at least 1")
    if train_count < n_neighbors:
        raise ValueError(
            "route detector requires at least n_neighbors training rows "
            f"(got {train_count}, n_neighbors={n_neighbors})"
        )
    if train_count < 2:
        raise ValueError(
            "shape detector requires at least 2 training rows for LocalOutlierFactor"
        )


def _feature_matrix(feature_df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=columns)
    return (
        feature_df.reindex(columns=columns, fill_value=0.0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    scores = _finite_array(scores)
    if len(scores) == 0:
        return scores
    if len(scores) == 1 or np.allclose(scores, scores[0]):
        return np.zeros(len(scores), dtype=float)
    ranks = pd.Series(scores).rank(method="average", ascending=True).to_numpy(dtype=float)
    return (ranks - 1.0) / float(len(scores) - 1)


def _mean_rank_scores(components: dict[str, np.ndarray]) -> np.ndarray:
    present = [scores for scores in components.values() if len(scores) > 0]
    if not present:
        return np.zeros(0, dtype=float)
    return _finite_array(np.mean(np.vstack(present), axis=0))


def _finite_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _bounded_contamination(contamination: float) -> float:
    return min(max(float(contamination), 1e-6), 0.5)
