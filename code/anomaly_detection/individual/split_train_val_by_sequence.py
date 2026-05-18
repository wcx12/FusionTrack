#!/usr/bin/env python3
"""
Split the exported training trajectories into train/val by sequence, and
optionally split already-exported feature pickle files with the same IDs.

Why split by sequence instead of by sample_id:
- avoids scene leakage between train and validation
- keeps evaluation closer to "new sequence" generalization
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

from mtf_ba.trajectory_jsonl import iter_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split training trajectories/features into train/val by sequence."
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("outputs")
        / "vt_tiny_mot_individual"
        / "individual_trajectories_train.jsonl",
        help="Path to the exported training trajectory JSONL.",
    )
    parser.add_argument(
        "--trajectory-output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_individual_split",
        help="Directory for split train/val trajectory JSONL files.",
    )
    parser.add_argument(
        "--feature-input-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_features",
        help="Directory containing full training feature pickle files.",
    )
    parser.add_argument(
        "--feature-output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_features_split",
        help="Directory for split train/val feature pickle files.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Target ratio of validation trajectories, selected at sequence granularity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sequence shuffling.",
    )
    parser.add_argument(
        "--split-features",
        action="store_true",
        help="Also split feature pickle files using the resulting sample_id split.",
    )
    return parser.parse_args()


def load_train_trajectories(path: Path) -> list[dict[str, Any]]:
    return list(iter_trajectory_jsonl(path))


def choose_val_sequences(
    trajectories: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> set[str]:
    sequence_counts: dict[str, int] = {}
    for trajectory in trajectories:
        sequence = trajectory["sequence"]
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1

    sequences = list(sequence_counts.keys())
    rng = random.Random(seed)
    rng.shuffle(sequences)

    target_val = len(trajectories) * val_ratio
    selected: set[str] = set()
    current = 0
    for sequence in sequences:
        if current >= target_val and selected:
            break
        selected.add(sequence)
        current += sequence_counts[sequence]
    return selected


def split_trajectories(
    trajectories: list[dict[str, Any]],
    val_sequences: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_items: list[dict[str, Any]] = []
    val_items: list[dict[str, Any]] = []

    for trajectory in trajectories:
        if trajectory["sequence"] in val_sequences:
            val_items.append(trajectory)
        else:
            train_items.append(trajectory)
    return train_items, val_items


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def split_feature_pickles(
    feature_input_dir: Path,
    feature_output_dir: Path,
    train_ids: set[str],
    val_ids: set[str],
) -> dict[str, dict[str, int]]:
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}

    for feature_path in sorted(feature_input_dir.glob("*_train.pkl")):
        with feature_path.open("rb") as f:
            feature_dict: dict[str, Any] = pickle.load(f)

        train_dict = {k: v for k, v in feature_dict.items() if k in train_ids}
        val_dict = {k: v for k, v in feature_dict.items() if k in val_ids}

        stem = feature_path.stem[:-6] if feature_path.stem.endswith("_train") else feature_path.stem
        train_out = feature_output_dir / f"{stem}_train.pkl"
        val_out = feature_output_dir / f"{stem}_val.pkl"

        with train_out.open("wb") as f:
            pickle.dump(train_dict, f)
        with val_out.open("wb") as f:
            pickle.dump(val_dict, f)

        stats[stem] = {
            "train_samples": len(train_dict),
            "val_samples": len(val_dict),
        }

    return stats


def main() -> None:
    args = parse_args()
    trajectories = load_train_trajectories(args.train_jsonl)
    val_sequences = choose_val_sequences(
        trajectories=trajectories,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_items, val_items = split_trajectories(
        trajectories=trajectories,
        val_sequences=val_sequences,
    )

    trajectory_output_dir = args.trajectory_output_dir.resolve()
    train_jsonl = trajectory_output_dir / "individual_trajectories_train.jsonl"
    val_jsonl = trajectory_output_dir / "individual_trajectories_val.jsonl"
    summary_json = trajectory_output_dir / "train_val_split_summary.json"

    write_jsonl(train_jsonl, train_items)
    write_jsonl(val_jsonl, val_items)

    train_ids = {item["sample_id"] for item in train_items}
    val_ids = {item["sample_id"] for item in val_items}

    feature_stats: dict[str, dict[str, int]] = {}
    if args.split_features:
        feature_stats = split_feature_pickles(
            feature_input_dir=args.feature_input_dir.resolve(),
            feature_output_dir=args.feature_output_dir.resolve(),
            train_ids=train_ids,
            val_ids=val_ids,
        )

    summary = {
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "num_train_trajectories": len(train_items),
        "num_val_trajectories": len(val_items),
        "num_train_sequences": len({item["sequence"] for item in train_items}),
        "num_val_sequences": len(val_sequences),
        "val_ratio_requested": args.val_ratio,
        "seed": args.seed,
        "val_sequences": sorted(val_sequences),
        "feature_stats": feature_stats,
    }
    save_summary(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
