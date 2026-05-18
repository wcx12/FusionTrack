#!/usr/bin/env python3
"""
Train one LSTM autoencoder detector on one exported feature family.

Example:
  python train_single_modality_detector.py --feature-name route_rgb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one detector for one single-modality feature set."
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
        help="Which exported feature family to train on.",
    )
    parser.add_argument(
        "--train-pkl",
        type=Path,
        default=None,
        help="Optional explicit train pickle path.",
    )
    parser.add_argument(
        "--val-pkl",
        type=Path,
        default=None,
        help="Optional explicit val pickle path.",
    )
    parser.add_argument(
        "--test-pkl",
        type=Path,
        default=None,
        help="Optional explicit test pickle path.",
    )
    parser.add_argument(
        "--split-feature-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_features_split",
        help="Directory containing split train/val feature pickles.",
    )
    parser.add_argument(
        "--test-feature-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_features",
        help="Directory containing test feature pickles.",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_models",
        help="Base directory for trained detector checkpoints.",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Stop training if validation loss does not improve enough for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss drop required to reset early-stopping patience.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--min-length",
        type=int,
        default=None,
        help=(
            "Minimum sequence length after feature export. "
            "If omitted, a feature-aware default is used "
            "(10 for route/speed, 2 for shape)."
        ),
    )
    parser.add_argument("--max-length", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_path(explicit: Path | None, directory: Path, filename: str) -> Path:
    return explicit if explicit is not None else directory / filename


def main() -> None:
    from mtf_ba.feature_training import TrainConfig, train_feature_detector

    args = parse_args()
    feature_name = args.feature_name

    train_pkl = resolve_path(
        args.train_pkl,
        args.split_feature_dir,
        f"{feature_name}_train.pkl",
    )
    val_pkl = resolve_path(
        args.val_pkl,
        args.split_feature_dir,
        f"{feature_name}_val.pkl",
    )
    test_pkl = resolve_path(
        args.test_pkl,
        args.test_feature_dir,
        f"{feature_name}_test.pkl",
    )

    config = TrainConfig(
        feature_name=feature_name,
        train_pkl=train_pkl,
        val_pkl=val_pkl,
        test_pkl=test_pkl,
        output_dir=args.model_output_dir / feature_name,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        seed=args.seed,
        normalize=not args.no_normalize,
        min_length=args.min_length,
        max_length=args.max_length,
        num_workers=args.num_workers,
        cuda_device=args.cuda_device,
    )
    summary = train_feature_detector(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
