from __future__ import annotations

import pickle
from pathlib import Path
import sys

import pandas as pd


_INDIVIDUAL_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "individual"
if _INDIVIDUAL_PACKAGE_ROOT.is_dir():
    _individual_package_root = str(_INDIVIDUAL_PACKAGE_ROOT)
    if _individual_package_root not in sys.path:
        sys.path.insert(0, _individual_package_root)

from mtf_ba.single_modality_features import (
    FeatureBuildConfig,
    build_route_feature,
    build_shape_feature,
    build_speed_feature,
)
from mtf_ba.trajectory_jsonl import iter_trajectory_jsonl


def build_fused_feature_sets(
    jsonl_path: Path | str,
    config: FeatureBuildConfig | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    config = config or FeatureBuildConfig()
    feature_sets: dict[str, dict[str, pd.DataFrame]] = {
        "route_fused": {},
        "speed_fused": {},
        "shape_fused": {},
    }

    for trajectory in iter_trajectory_jsonl(jsonl_path):
        sample_id = trajectory["sample_id"]

        route = build_route_feature(
            trajectory,
            modality="fused",
            step_size=config.route_step_size,
            min_points=config.min_points_per_modality,
        )
        if route is not None:
            feature_sets["route_fused"][sample_id] = route

        speed = build_speed_feature(
            trajectory,
            modality="fused",
            min_points=config.min_points_per_modality,
        )
        if speed is not None:
            feature_sets["speed_fused"][sample_id] = speed

        shape = build_shape_feature(
            trajectory,
            modality="fused",
            new_time_step=config.shape_time_step,
            min_points=config.min_points_per_modality,
            min_total_length=config.shape_min_total_length,
            min_nonzero_steps=config.shape_min_nonzero_steps,
            min_variance=config.shape_min_variance,
        )
        if shape is not None:
            feature_sets["shape_fused"][sample_id] = shape

    return feature_sets


def save_fused_feature_sets(
    feature_sets: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path | str,
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
