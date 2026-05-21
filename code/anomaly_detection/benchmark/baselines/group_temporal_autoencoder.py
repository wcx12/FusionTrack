from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fusiontrack.group_graph import extract_object_states
from protocol.schemas import build_sample_id


FEATURE_COLUMNS = (
    "num_objects",
    "edge_density",
    "mean_degree",
    "degree_std",
    "mean_pair_distance",
    "dispersion",
    "mean_speed",
    "neighbor_churn",
)


def fit_temporal_graph_autoencoder(
    train_windows: Iterable[dict],
    n_components: int = 3,
) -> dict[str, Any]:
    signatures = build_temporal_graph_signature_table(train_windows)
    if signatures.empty:
        raise ValueError(
            "Cannot fit temporal graph autoencoder with no training windows or graph signatures"
        )

    features = _feature_matrix(signatures)
    component_count = _clamp_n_components(n_components, len(features), len(FEATURE_COLUMNS))
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=component_count)),
        ]
    )
    pipeline.fit(features)
    return {
        "pipeline": pipeline,
        "n_components": component_count,
        "feature_columns": list(FEATURE_COLUMNS),
    }


def score_temporal_graph_autoencoder(
    model: dict[str, Any],
    score_windows: Iterable[dict],
) -> list[dict[str, Any]]:
    score_windows = list(score_windows)
    rows: list[dict[str, Any]] = []
    for window in score_windows:
        signatures = build_temporal_graph_signature_table([window])
        reconstruction_error = _reconstruction_error(model, signatures)
        temporal_churn = _mean_column(signatures, "neighbor_churn")
        dispersion = _mean_column(signatures, "dispersion")
        frame_start, frame_end = _window_frame_bounds(window)
        sequence = str(window.get("sequence", ""))
        window_id = str(window.get("window_id", window.get("sample_id", "")))
        component_scores = {
            "reconstruction_error": reconstruction_error,
            "temporal_churn": temporal_churn,
            "dispersion": dispersion,
        }
        for obj in _window_objects(window):
            track_id = str(obj["track_id"])
            rows.append(
                {
                    "sample_id": _sample_id(obj, sequence, track_id),
                    "window_id": window_id,
                    "sequence": sequence,
                    "track_id": track_id,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "source": "group_temporal_autoencoder:pca_reconstruction",
                    "score": reconstruction_error,
                    "component_scores": dict(component_scores),
                    "metadata": {
                        "n_components": int(model["n_components"]),
                        "window_id": window_id,
                        "feature_columns": list(model["feature_columns"]),
                    },
                }
            )
    return rows


def run_temporal_graph_autoencoder(
    train_windows: Iterable[dict],
    score_windows: Iterable[dict],
    n_components: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    np.random.seed(seed)
    model = fit_temporal_graph_autoencoder(train_windows, n_components=n_components)
    return score_temporal_graph_autoencoder(model, score_windows)


def build_temporal_graph_signature_table(windows: Iterable[dict]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        rows.extend(_window_signature_rows(window, window_index))
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=("window_id", "frame_id", *FEATURE_COLUMNS))
    for column in FEATURE_COLUMNS:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    return table.reset_index(drop=True)


def _window_signature_rows(window: dict, window_index: int) -> list[dict[str, Any]]:
    states = extract_object_states(window)
    if not states:
        return []

    by_frame: dict[int, list[dict]] = {}
    by_track: dict[str, list[dict]] = {}
    for state in states:
        by_frame.setdefault(int(state["frame_id"]), []).append(state)
        by_track.setdefault(str(state["track_id"]), []).append(state)
    for track_states in by_track.values():
        track_states.sort(key=lambda state: int(state["frame_id"]))

    speed_by_frame = _speed_by_frame(by_track)
    previous_neighbors: dict[str, str | None] = {}
    rows: list[dict[str, Any]] = []
    for frame_id, frame_states in sorted(by_frame.items()):
        frame_states = sorted(frame_states, key=lambda state: str(state["track_id"]))
        neighbors = _nearest_neighbors(frame_states)
        pair_distances = _pair_distances(frame_states)
        edges = _undirected_nearest_edges(neighbors)
        num_objects = len(frame_states)
        possible_edges = num_objects * (num_objects - 1) / 2.0
        degrees = _degrees(frame_states, edges)
        churn = _neighbor_churn(previous_neighbors, neighbors)
        previous_neighbors = neighbors
        rows.append(
            {
                "window_id": str(
                    window.get("window_id", window.get("sample_id", f"window_{window_index}"))
                ),
                "frame_id": int(frame_id),
                "num_objects": float(num_objects),
                "edge_density": len(edges) / possible_edges if possible_edges else 0.0,
                "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
                "degree_std": float(np.std(degrees)) if degrees else 0.0,
                "mean_pair_distance": float(np.mean(pair_distances)) if pair_distances else 0.0,
                "dispersion": _frame_dispersion(frame_states),
                "mean_speed": float(np.mean(speed_by_frame.get(frame_id, [0.0]))),
                "neighbor_churn": churn,
            }
        )
    return [{key: _finite_default(value) for key, value in row.items()} for row in rows]


def _feature_matrix(signatures: pd.DataFrame) -> pd.DataFrame:
    if signatures.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return signatures.reindex(columns=FEATURE_COLUMNS, fill_value=0.0).apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)


def _clamp_n_components(requested: int, n_samples: int, n_features: int) -> int:
    return max(1, min(int(requested), int(n_samples), int(n_features)))


def _reconstruction_error(model: dict[str, Any], signatures: pd.DataFrame) -> float:
    if signatures.empty:
        return 0.0
    features = _feature_matrix(signatures)
    pipeline: Pipeline = model["pipeline"]
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    pca: PCA = pipeline.named_steps["pca"]
    scaled = scaler.transform(features)
    reconstructed = pca.inverse_transform(pca.transform(scaled))
    errors = np.mean((scaled - reconstructed) ** 2, axis=1)
    return _finite_default(float(np.mean(errors)))


def _speed_by_frame(by_track: dict[str, list[dict]]) -> dict[int, list[float]]:
    speeds: dict[int, list[float]] = {}
    for states in by_track.values():
        for previous, current in zip(states, states[1:]):
            delta_frames = max(int(current["frame_id"]) - int(previous["frame_id"]), 1)
            speed = math.dist(previous["center_xy"], current["center_xy"]) / float(delta_frames)
            speeds.setdefault(int(current["frame_id"]), []).append(speed)
    return speeds


def _nearest_neighbors(frame_states: list[dict]) -> dict[str, str | None]:
    neighbors: dict[str, str | None] = {}
    for state in frame_states:
        distances = [
            (math.dist(state["center_xy"], other["center_xy"]), str(other["track_id"]))
            for other in frame_states
            if str(other["track_id"]) != str(state["track_id"])
        ]
        distances.sort(key=lambda item: (item[0], item[1]))
        neighbors[str(state["track_id"])] = distances[0][1] if distances else None
    return neighbors


def _undirected_nearest_edges(neighbors: dict[str, str | None]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for track_id, neighbor_id in neighbors.items():
        if neighbor_id is None:
            continue
        edges.add(tuple(sorted((track_id, neighbor_id))))
    return edges


def _degrees(frame_states: list[dict], edges: set[tuple[str, str]]) -> list[float]:
    degree_by_track = {str(state["track_id"]): 0.0 for state in frame_states}
    for left, right in edges:
        degree_by_track[left] = degree_by_track.get(left, 0.0) + 1.0
        degree_by_track[right] = degree_by_track.get(right, 0.0) + 1.0
    return list(degree_by_track.values())


def _pair_distances(frame_states: list[dict]) -> list[float]:
    distances: list[float] = []
    for index, state in enumerate(frame_states):
        for other in frame_states[index + 1 :]:
            distances.append(math.dist(state["center_xy"], other["center_xy"]))
    return distances


def _frame_dispersion(frame_states: list[dict]) -> float:
    if not frame_states:
        return 0.0
    center = [
        sum(float(state["center_xy"][0]) for state in frame_states) / len(frame_states),
        sum(float(state["center_xy"][1]) for state in frame_states) / len(frame_states),
    ]
    return float(np.mean([math.dist(state["center_xy"], center) for state in frame_states]))


def _neighbor_churn(
    previous_neighbors: dict[str, str | None],
    current_neighbors: dict[str, str | None],
) -> float:
    comparisons = 0
    changes = 0
    for track_id, neighbor_id in current_neighbors.items():
        if track_id not in previous_neighbors:
            continue
        comparisons += 1
        if previous_neighbors[track_id] != neighbor_id:
            changes += 1
    return changes / float(comparisons) if comparisons else 0.0


def _mean_column(signatures: pd.DataFrame, column: str) -> float:
    if signatures.empty or column not in signatures:
        return 0.0
    return _finite_default(float(pd.to_numeric(signatures[column], errors="coerce").fillna(0.0).mean()))


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
    return sorted(
        [obj for obj in window.get("objects", []) if obj.get("track_id") not in (None, "")],
        key=lambda obj: str(obj["track_id"]),
    )


def _sample_id(obj: dict, sequence: str, track_id: str) -> str:
    sample_id = obj.get("sample_id")
    if sample_id not in (None, ""):
        return str(sample_id)
    return build_sample_id(sequence, track_id)


def _finite_default(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value
