#!/usr/bin/env python3
"""
Run baseline-style scoring for all six trained detectors on one split.

Example:
  python score_all_single_modality_detectors.py --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FEATURES = [
    "route_rgb",
    "speed_rgb",
    "shape_rgb",
    "route_thermal",
    "speed_thermal",
    "shape_thermal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score all six trained detectors on one split."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Which split feature files to score.",
    )
    parser.add_argument(
        "--split-feature-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_features_split",
        help="Directory containing train/val split feature files.",
    )
    parser.add_argument(
        "--test-feature-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_features",
        help="Directory containing original test feature files.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_models",
        help="Directory containing trained detector subdirectories.",
    )
    parser.add_argument(
        "--score-output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_scores",
        help="Base directory for embeddings/loss/final score exports.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument(
        "--min-length",
        type=int,
        default=None,
        help=(
            "Minimum sequence length for scoring. "
            "If omitted, a feature-aware default is used "
            "(10 for route/speed, 2 for shape)."
        ),
    )
    parser.add_argument("--max-length", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    from mtf_ba.feature_scoring import score_feature_detector

    args = parse_args()
    summaries: list[dict] = []

    feature_dir = (
        args.test_feature_dir if args.split == "test" else args.split_feature_dir
    )

    for feature_name in FEATURES:
        print(f"\n=== Scoring {feature_name} on {args.split} ===")
        feature_pkl = feature_dir / f"{feature_name}_{args.split}.pkl"
        model_dir = args.model_output_dir / feature_name
        output_dir = args.score_output_dir / feature_name / args.split

        summary = score_feature_detector(
            feature_name=feature_name,
            feature_pkl=feature_pkl,
            model_dir=model_dir,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cuda_device=args.cuda_device,
            k_neighbors=args.k_neighbors,
            min_length=args.min_length,
            max_length=args.max_length,
        )
        summaries.append(summary)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
