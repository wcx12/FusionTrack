from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from baselines.individual_features import (
    FEATURE_COLUMNS,
    build_handcrafted_feature_table,
)


CLASSICAL_METHODS = ("isolation_forest", "lof", "one_class_svm")


def fit_classical_detector(
    feature_df: pd.DataFrame,
    method: str,
    seed: int = 42,
    contamination: float = 0.05,
) -> object:
    _validate_method(method)
    features = _feature_matrix(feature_df)
    if len(features) == 0:
        raise ValueError("Cannot fit a classical detector with no feature rows")

    contamination = _bounded_contamination(contamination)
    if method == "isolation_forest":
        detector = IsolationForest(
            contamination=contamination,
            random_state=seed,
            n_estimators=100,
        )
    elif method == "lof":
        detector = LocalOutlierFactor(
            n_neighbors=max(1, min(20, len(features) - 1)),
            contamination=contamination,
            novelty=True,
        )
    else:
        detector = OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("detector", detector),
        ]
    )
    return model.fit(features)


def score_classical_detector(
    model: object,
    feature_df: pd.DataFrame,
    method: str,
) -> dict[str, float]:
    _validate_method(method)
    features = _feature_matrix(feature_df)
    if len(features) == 0:
        return {}
    raw_scores = -np.asarray(model.decision_function(features), dtype=float)
    sample_ids = feature_df["sample_id"].astype(str).tolist()
    return {
        sample_id: float(score) if np.isfinite(score) else 0.0
        for sample_id, score in zip(sample_ids, raw_scores)
    }


def run_classical_baseline(
    train_trajectories: Iterable[dict],
    score_trajectories: Iterable[dict],
    method: str,
    seed: int = 42,
    contamination: float = 0.05,
) -> list[dict[str, Any]]:
    _validate_method(method)
    train_features = build_handcrafted_feature_table(train_trajectories)
    score_features = build_handcrafted_feature_table(score_trajectories)
    model = fit_classical_detector(
        train_features,
        method=method,
        seed=seed,
        contamination=contamination,
    )
    scores = score_classical_detector(model, score_features, method=method)

    rows: list[dict[str, Any]] = []
    for _, feature_row in score_features.iterrows():
        sample_id = str(feature_row["sample_id"])
        score = float(scores[sample_id])
        rows.append(
            {
                "sample_id": sample_id,
                "sequence": str(feature_row["sequence"]),
                "track_id": str(feature_row["track_id"]),
                "source": f"individual_classical:{method}",
                "score": score,
                "component_scores": {method: score},
                "metadata": {
                    "method": method,
                    "seed": int(seed),
                    "contamination": float(contamination),
                    "feature_columns": list(FEATURE_COLUMNS),
                },
            }
        )
    return rows


def _validate_method(method: str) -> None:
    if method not in CLASSICAL_METHODS:
        raise ValueError(
            f"Unknown classical detector '{method}'. Expected one of {CLASSICAL_METHODS}."
        )


def _feature_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    matrix = feature_df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _bounded_contamination(contamination: float) -> float:
    return min(max(float(contamination), 1e-6), 0.5)
