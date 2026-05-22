from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from baselines.group_features import FEATURE_COLUMNS, build_group_feature_table
from protocol.schemas import build_sample_id


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
        if len(features) < 2:
            raise ValueError("LOF requires at least 2 training feature rows")
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
) -> list[float]:
    _validate_method(method)
    features = _feature_matrix(feature_df)
    if len(features) == 0:
        return []
    raw_scores = -np.asarray(model.decision_function(features), dtype=float)
    return [float(score) if np.isfinite(score) else 0.0 for score in raw_scores]


def run_classical_baseline(
    train_windows: Iterable[dict],
    score_windows: Iterable[dict],
    method: str,
    seed: int = 42,
    contamination: float = 0.05,
) -> list[dict[str, Any]]:
    _validate_method(method)
    train_features = build_group_feature_table(train_windows)
    score_windows = list(score_windows)
    score_features = build_group_feature_table(score_windows, sort_by_window_id=False)
    model = fit_classical_detector(
        train_features,
        method=method,
        seed=seed,
        contamination=contamination,
    )
    scores = score_classical_detector(model, score_features, method=method)

    rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(score_windows):
        window_id = str(window.get("window_id", window.get("sample_id", "")))
        score = float(scores[window_index]) if window_index < len(scores) else 0.0
        frame_start, frame_end = _window_frame_bounds(window)
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
                    "source": f"group_classical:{method}",
                    "score": score,
                    "component_scores": {method: score},
                    "metadata": {
                        "method": method,
                        "seed": int(seed),
                        "contamination": float(contamination),
                        "feature_columns": list(FEATURE_COLUMNS),
                        "window_id": window_id,
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


def _window_track_ids(window: dict) -> list[str]:
    track_ids = [
        str(obj["track_id"])
        for obj in window.get("objects", [])
        if obj.get("track_id") not in (None, "")
    ]
    return sorted(dict.fromkeys(track_ids))


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
