#!/usr/bin/env python3
"""
Export baseline-style single-modality features from object-centric trajectory JSONL.

Outputs six pickle files per split:
  - route_rgb_<split>.pkl
  - speed_rgb_<split>.pkl
  - shape_rgb_<split>.pkl
  - route_thermal_<split>.pkl
  - speed_thermal_<split>.pkl
  - shape_thermal_<split>.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtf_ba.single_modality_features import (
    FeatureBuildConfig,
    build_single_modality_feature_sets,
    save_feature_sets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export single-modality baseline-style trajectory features."
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        required=True,
        help="Path to individual_trajectories_<split>.jsonl.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Split name used in output filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_features",
        help="Directory for exported feature pickle files.",
    )
    parser.add_argument(
        "--route-step-size",
        type=float,
        default=10.0,
        help="Distance step used for route interpolation.",
    )
    parser.add_argument(
        "--shape-time-step",
        type=float,
        default=24.0,
        help="Normalized time step used for shape resampling.",
    )
    parser.add_argument(
        "--min-points-per-modality",
        type=int,
        default=3,
        help="Minimum visible points required to build one modality feature set.",
    )
    parser.add_argument(
        "--shape-min-total-length",
        type=float,
        default=1.0,
        help="Minimum total path length required before shape is considered valid.",
    )
    parser.add_argument(
        "--shape-min-nonzero-steps",
        type=int,
        default=2,
        help="Minimum number of non-zero motion steps required for shape.",
    )
    parser.add_argument(
        "--shape-min-variance",
        type=float,
        default=1e-8,
        help="Minimum variance required before PCA in the shape branch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FeatureBuildConfig(
        route_step_size=args.route_step_size,
        shape_time_step=args.shape_time_step,
        min_points_per_modality=args.min_points_per_modality,
        shape_min_total_length=args.shape_min_total_length,
        shape_min_nonzero_steps=args.shape_min_nonzero_steps,
        shape_min_variance=args.shape_min_variance,
    )

    feature_sets = build_single_modality_feature_sets(
        jsonl_path=args.jsonl_path,
        config=config,
        show_progress=True,
    )
    output_paths = save_feature_sets(
        feature_sets=feature_sets,
        output_dir=args.output_dir,
        split=args.split,
    )

    summary = {
        "jsonl_path": str(args.jsonl_path.resolve()),
        "split": args.split,
        "output_dir": str(args.output_dir.resolve()),
        "route_step_size": args.route_step_size,
        "shape_time_step": args.shape_time_step,
        "min_points_per_modality": args.min_points_per_modality,
        "shape_min_total_length": args.shape_min_total_length,
        "shape_min_nonzero_steps": args.shape_min_nonzero_steps,
        "shape_min_variance": args.shape_min_variance,
        "num_samples": {name: len(items) for name, items in feature_sets.items()},
        "outputs": output_paths,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
