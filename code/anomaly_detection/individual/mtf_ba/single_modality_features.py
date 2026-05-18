from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mtf_ba.trajectory_jsonl import iter_trajectory_jsonl

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):  # type: ignore[misc]
        return iterable


EPSILON = 1e-8


@dataclass
class FeatureBuildConfig:
    route_step_size: float = 10.0
    shape_time_step: float = 24.0
    min_points_per_modality: int = 3
    shape_min_total_length: float = 1.0
    shape_min_nonzero_steps: int = 2
    shape_min_variance: float = 1e-8


def _extract_modality_points(
    trajectory: dict[str, Any],
    modality: str,
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for point in trajectory["points"]:
        state = point.get(modality)
        if state is None:
            continue
        center_xy = state.get("center_xy")
        if center_xy is None:
            continue
        frame_id = point.get("frame_id")
        if frame_id is None:
            continue
        points.append(
            {
                "frame_id": float(frame_id),
                "x": float(center_xy[0]),
                "y": float(center_xy[1]),
            }
        )
    return points


def _relative_coordinates(points: list[dict[str, float]]) -> pd.DataFrame:
    data = pd.DataFrame(points)
    data["x_rel"] = data["x"] - data["x"].iloc[0]
    data["y_rel"] = data["y"] - data["y"].iloc[0]
    return data


def build_route_feature(
    trajectory: dict[str, Any],
    modality: str,
    step_size: float,
    min_points: int,
) -> pd.DataFrame | None:
    points = _extract_modality_points(trajectory, modality)
    if len(points) < min_points:
        return None

    data = _relative_coordinates(points)
    dx = data["x_rel"].diff().fillna(0.0)
    dy = data["y_rel"].diff().fillna(0.0)
    data["step_dist"] = np.sqrt(dx**2 + dy**2)
    data["cumulative_dist"] = data["step_dist"].cumsum()

    max_dist = float(data["cumulative_dist"].iloc[-1])
    if max_dist <= EPSILON:
        return pd.DataFrame(
            {
                "latitude": data["x_rel"].to_numpy(dtype=float),
                "longitude": data["y_rel"].to_numpy(dtype=float),
            }
        )

    desired_dists = np.arange(0.0, max_dist + step_size, step_size, dtype=float)
    if desired_dists.size == 0 or desired_dists[-1] < max_dist:
        desired_dists = np.append(desired_dists, max_dist)

    interp_x = np.interp(desired_dists, data["cumulative_dist"], data["x_rel"])
    interp_y = np.interp(desired_dists, data["cumulative_dist"], data["y_rel"])

    return pd.DataFrame(
        {
            "latitude": interp_x,
            "longitude": interp_y,
        }
    )


def build_speed_feature(
    trajectory: dict[str, Any],
    modality: str,
    min_points: int,
) -> pd.DataFrame | None:
    points = _extract_modality_points(trajectory, modality)
    if len(points) < min_points:
        return None

    data = pd.DataFrame(points)
    dx = data["x"].diff().fillna(0.0)
    dy = data["y"].diff().fillna(0.0)
    delta_frame = data["frame_id"].diff().fillna(1.0).clip(lower=1.0)
    speed = np.sqrt(dx**2 + dy**2) / delta_frame

    return pd.DataFrame({"speed": speed.to_numpy(dtype=float)})


def _drop_consecutive_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    keep = (data["x_rel"] != data["x_rel"].shift()) | (
        data["y_rel"] != data["y_rel"].shift()
    )
    keep.iloc[0] = True
    return data.loc[keep].reset_index(drop=True)


def _resample_shape(data: pd.DataFrame, new_time_step: float) -> pd.DataFrame:
    total_duration = float(data["timestamp_nor"].iloc[-1])
    if total_duration < new_time_step or len(data) <= 1:
        return pd.DataFrame(
            {
                "timestamp_nor": [0.0, new_time_step],
                "x_nor_re": [0.0, float(data["x_nor"].iloc[-1])],
                "y_nor_re": [0.0, float(data["y_nor"].iloc[-1])],
            }
        )

    new_timestamps = np.arange(0.0, total_duration, new_time_step, dtype=float)
    if new_timestamps.size == 0 or new_timestamps[-1] < total_duration:
        new_timestamps = np.append(new_timestamps, total_duration)

    return pd.DataFrame(
        {
            "timestamp_nor": new_timestamps,
            "x_nor_re": np.interp(new_timestamps, data["timestamp_nor"], data["x_nor"]),
            "y_nor_re": np.interp(new_timestamps, data["timestamp_nor"], data["y_nor"]),
        }
    )


def build_shape_feature(
    trajectory: dict[str, Any],
    modality: str,
    new_time_step: float,
    min_points: int,
    min_total_length: float = 1.0,
    min_nonzero_steps: int = 2,
    min_variance: float = 1e-8,
) -> pd.DataFrame | None:
    """
    Build a baseline-style shape feature for one modality.

    Why this branch needs extra filtering:

    Shape is intended to represent motion geometry, not absolute position. For
    truly static or almost-static tracks, the geometry signal is effectively
    absent. If we force those tracks through the normalization + resampling +
    StandardScaler + PCA pipeline, the resampled curve often collapses to a
    constant sequence. In that case PCA receives zero-variance input and emits
    warnings such as:

      RuntimeWarning: invalid value encountered in divide

    To avoid producing meaningless shape features, we explicitly treat these
    tracks as "shape not available" and return None.

    Filtering logic:
    1. require enough visible points for this modality
    2. remove consecutive duplicate positions
    3. require enough total path length
    4. require enough genuinely moving steps
    5. require enough variance after resampling, before PCA
    """
    points = _extract_modality_points(trajectory, modality)
    if len(points) < min_points:
        return None

    data = _relative_coordinates(points)
    data = _drop_consecutive_duplicates(data)
    if len(data) < 2:
        return None

    dx = data["x_rel"].diff().fillna(0.0)
    dy = data["y_rel"].diff().fillna(0.0)
    displacement = np.sqrt(dx**2 + dy**2)
    total_length = float(displacement.sum())
    # A very small total path length means the object is effectively static in
    # this modality, so there is no reliable shape information to model.
    if total_length <= max(EPSILON, min_total_length):
        return None

    # Count how many steps contain real movement. This prevents tracks with one
    # tiny numerical wiggle from being treated as meaningful shape trajectories.
    nonzero_steps = int((displacement > EPSILON).sum())
    if nonzero_steps < min_nonzero_steps:
        return None

    data["x_nor"] = data["x_rel"] / total_length
    data["y_nor"] = data["y_rel"] / total_length
    data["timestamp"] = np.arange(len(data), dtype=float)
    data["timestamp_nor"] = data["timestamp"] / total_length

    resampled = _resample_shape(data, new_time_step=new_time_step)
    resampled_xy = resampled[["x_nor_re", "y_nor_re"]].to_numpy(dtype=float)

    # Final safety check before StandardScaler/PCA:
    # if the resampled sequence has near-zero variance, PCA would receive an
    # almost constant matrix and produce unstable explained-variance statistics.
    if float(np.var(resampled_xy)) <= min_variance:
        return None

    scaled = StandardScaler().fit_transform(resampled_xy)
    if float(np.var(scaled)) <= min_variance:
        return None

    projected = PCA(n_components=2).fit_transform(scaled)

    return pd.DataFrame(
        {
            "delta_x": projected[:, 0],
            "delta_y": projected[:, 1],
        }
    )


def build_single_modality_feature_sets(
    jsonl_path: str | Path,
    config: FeatureBuildConfig | None = None,
    show_progress: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    config = config or FeatureBuildConfig()
    feature_sets: dict[str, dict[str, pd.DataFrame]] = {
        "route_rgb": {},
        "speed_rgb": {},
        "shape_rgb": {},
        "route_thermal": {},
        "speed_thermal": {},
        "shape_thermal": {},
    }

    trajectories = iter_trajectory_jsonl(jsonl_path)
    for trajectory in tqdm(
        trajectories,
        desc="Building single-modality features",
        unit="trajectory",
        disable=not show_progress,
    ):
        sample_id = trajectory["sample_id"]

        route_rgb = build_route_feature(
            trajectory,
            modality="rgb",
            step_size=config.route_step_size,
            min_points=config.min_points_per_modality,
        )
        if route_rgb is not None:
            feature_sets["route_rgb"][sample_id] = route_rgb

        speed_rgb = build_speed_feature(
            trajectory,
            modality="rgb",
            min_points=config.min_points_per_modality,
        )
        if speed_rgb is not None:
            feature_sets["speed_rgb"][sample_id] = speed_rgb

        shape_rgb = build_shape_feature(
            trajectory,
            modality="rgb",
            new_time_step=config.shape_time_step,
            min_points=config.min_points_per_modality,
            min_total_length=config.shape_min_total_length,
            min_nonzero_steps=config.shape_min_nonzero_steps,
            min_variance=config.shape_min_variance,
        )
        if shape_rgb is not None:
            feature_sets["shape_rgb"][sample_id] = shape_rgb

        route_thermal = build_route_feature(
            trajectory,
            modality="thermal",
            step_size=config.route_step_size,
            min_points=config.min_points_per_modality,
        )
        if route_thermal is not None:
            feature_sets["route_thermal"][sample_id] = route_thermal

        speed_thermal = build_speed_feature(
            trajectory,
            modality="thermal",
            min_points=config.min_points_per_modality,
        )
        if speed_thermal is not None:
            feature_sets["speed_thermal"][sample_id] = speed_thermal

        shape_thermal = build_shape_feature(
            trajectory,
            modality="thermal",
            new_time_step=config.shape_time_step,
            min_points=config.min_points_per_modality,
            min_total_length=config.shape_min_total_length,
            min_nonzero_steps=config.shape_min_nonzero_steps,
            min_variance=config.shape_min_variance,
        )
        if shape_thermal is not None:
            feature_sets["shape_thermal"][sample_id] = shape_thermal

    return feature_sets


def save_feature_sets(
    feature_sets: dict[str, dict[str, pd.DataFrame]],
    output_dir: str | Path,
    split: str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    for feature_name, feature_dict in feature_sets.items():
        path = output_dir / f"{feature_name}_{split}.pkl"
        with path.open("wb") as f:
            pickle.dump(feature_dict, f)
        paths[feature_name] = str(path)
    return paths
