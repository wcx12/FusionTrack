#!/usr/bin/env python3
"""
Export object-centric VT-Tiny-MOT trajectories keyed by sample_id.

Input:
  outputs/vt_tiny_mot_trajectories/observations_<split>.csv

Outputs:
  individual_trajectories_<split>.jsonl
  individual_trajectories_summary_<split>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtf_ba.individual_trajectories import load_object_trajectories

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export object-centric trajectories from observations CSV."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("outputs")
        / "vt_tiny_mot_trajectories"
        / "observations_train.csv",
        help="Path to observations_<split>.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_individual",
        help="Directory for object-centric trajectory exports.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name used in output file naming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_object_trajectories(args.csv_path, show_progress=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"individual_trajectories_{args.split}.jsonl"
    summary_path = output_dir / f"individual_trajectories_summary_{args.split}.json"

    total_points = 0
    total_rgb_visible = 0
    total_thermal_visible = 0

    with jsonl_path.open("w", encoding="utf-8") as f:
        for trajectory in tqdm(
            trajectories,
            desc="Writing trajectories",
            unit="trajectory",
        ):
            total_points += trajectory["num_points"]
            total_rgb_visible += trajectory["visible_rgb_frames"]
            total_thermal_visible += trajectory["visible_thermal_frames"]
            f.write(json.dumps(trajectory, ensure_ascii=False))
            f.write("\n")

    summary = {
        "csv_path": str(args.csv_path.resolve()),
        "output_jsonl": str(jsonl_path),
        "split": args.split,
        "num_trajectories": len(trajectories),
        "num_points": total_points,
        "visible_rgb_frames": total_rgb_visible,
        "visible_thermal_frames": total_thermal_visible,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
