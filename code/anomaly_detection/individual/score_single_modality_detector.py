#!/usr/bin/env python3
"""
Score one trained detector exactly in the baseline style:

1. load trained autoencoder checkpoint
2. export per-trajectory embeddings
3. export per-trajectory reconstruction losses
4. compute baseline-style neighborhood-adjusted final scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one trained detector on one feature pickle."
    )
    parser.add_argument(
        "--feature-name",
        required=True,
        choices=[
            "route_rgb",
            "speed_rgb",
            "shape_rgb",
            "route_thermal",
            "speed_thermal",
            "shape_thermal",
        ],
    )
    parser.add_argument(
        "--feature-pkl",
        type=Path,
        required=True,
        help="Path to the feature pickle to score.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing checkpoint + train_summary.json + normalization_stats.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where embeddings/loss/final score files will be written.",
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
    summary = score_feature_detector(
        feature_name=args.feature_name,
        feature_pkl=args.feature_pkl,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cuda_device=args.cuda_device,
        k_neighbors=args.k_neighbors,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
