from __future__ import annotations

from typing import Any, Iterable, Sequence

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
    calibration_columns: Sequence[str] = (),
    calibration_bins: int = 4,
    calibration_global_weight: float = 0.7,
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

    base_scores = (
        float(weights[0]) * np.asarray(nearest_rank, dtype=float)
        + float(weights[1]) * np.asarray(lof_rank, dtype=float)
        + float(weights[2]) * np.asarray(iforest_rank, dtype=float)
    )
    calibration_config = _calibration_metadata(
        calibration_columns,
        calibration_bins,
        calibration_global_weight,
    )
    if calibration_config["enabled"]:
        final_scores = _feature_stratified_rank01(
            base_scores,
            score_features,
            columns=tuple(calibration_config["columns"]),
            bins=int(calibration_config["bins"]),
            global_weight=float(calibration_config["global_weight"]),
        )
    else:
        final_scores = [
            float(score) if np.isfinite(score) else 0.0
            for score in base_scores
        ]

    rows: list[dict[str, Any]] = []
    for index, (_, feature_row) in enumerate(score_features.iterrows()):
        component_scores = {
            "nearest_feature_rank": float(nearest_rank[index]),
            "lof_novelty_rank": float(lof_rank[index]),
            "isolation_forest_rank": float(iforest_rank[index]),
        }
        if calibration_config["enabled"]:
            component_scores["uncalibrated_ensemble_rank"] = float(base_scores[index])
        score = final_scores[index]
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
                    "calibration": dict(calibration_config),
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


def _calibration_metadata(
    columns: Sequence[str],
    bins: int,
    global_weight: float,
) -> dict[str, Any]:
    valid_columns = [
        str(column)
        for column in columns
        if str(column) in FEATURE_COLUMNS
    ]
    clean_bins = max(2, int(bins))
    clean_global_weight = min(1.0, max(0.0, float(global_weight)))
    return {
        "enabled": bool(valid_columns),
        "columns": valid_columns,
        "bins": clean_bins,
        "global_weight": clean_global_weight,
    }


def _feature_stratified_rank01(
    values: Iterable[float],
    feature_df: pd.DataFrame,
    columns: Sequence[str],
    bins: int = 4,
    global_weight: float = 0.7,
) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return []
    global_rank = np.asarray(_rank01(array), dtype=float)
    valid_columns = [
        str(column)
        for column in columns
        if str(column) in feature_df.columns and str(column) in FEATURE_COLUMNS
    ]
    if not valid_columns:
        return [float(value) for value in global_rank]

    local_ranks: list[np.ndarray] = []
    for column in valid_columns:
        strata = _quantile_strata(feature_df[column], bins=bins)
        local_ranks.append(_rank_within_strata(array, strata, fallback=global_rank))
    if not local_ranks:
        return [float(value) for value in global_rank]

    clean_global_weight = min(1.0, max(0.0, float(global_weight)))
    local_rank = np.mean(np.vstack(local_ranks), axis=0)
    blended = clean_global_weight * global_rank + (1.0 - clean_global_weight) * local_rank
    blended = np.clip(np.where(np.isfinite(blended), blended, 0.0), 0.0, 1.0)
    return [float(value) for value in blended]


def _quantile_strata(values: Iterable[float], bins: int) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return np.asarray([], dtype=int)
    array = np.where(np.isfinite(array), array, 0.0)
    if float(np.max(array) - np.min(array)) <= 0.0:
        return np.zeros(len(array), dtype=int)

    max_bins = max(1, min(int(bins), len(array)))
    quantiles = np.linspace(0.0, 1.0, num=max_bins + 1)
    edges = np.unique(np.quantile(array, quantiles))
    if len(edges) <= 2:
        return np.zeros(len(array), dtype=int)
    return np.digitize(array, edges[1:-1], right=True).astype(int)


def _rank_within_strata(
    values: np.ndarray,
    strata: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    local = np.asarray(fallback, dtype=float).copy()
    for stratum in np.unique(strata):
        mask = strata == stratum
        if int(np.sum(mask)) < 2:
            continue
        local[mask] = np.asarray(_rank01(values[mask]), dtype=float)
    return np.clip(np.where(np.isfinite(local), local, 0.0), 0.0, 1.0)


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
