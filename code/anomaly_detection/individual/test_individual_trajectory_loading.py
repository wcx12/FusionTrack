#!/usr/bin/env python3
"""
Minimal test/demo for loading object-centric trajectories from observations CSV.

This script shows how per-frame observations are grouped into
`(sequence, track_id)` trajectories and how to inspect all trajectory points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtf_ba.individual_trajectories import load_object_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo loading object-centric trajectories from observations CSV."
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
        "--limit-trajectories",
        type=int,
        default=3,
        help="How many trajectories to print.",
    )
    parser.add_argument(
        "--limit-points",
        type=int,
        default=5,
        help="How many points to print per trajectory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_object_trajectories(args.csv_path)

    print(f"csv_path={args.csv_path.resolve()}")
    print(f"num_trajectories={len(trajectories)}")
    print()

    for trajectory in trajectories[: args.limit_trajectories]:
        print("=" * 80)
        print(f"sample_id={trajectory['sample_id']}")
        print(f"sequence={trajectory['sequence']}")
        print(f"track_id={trajectory['track_id']}")
        print(f"category={trajectory['category_name']} ({trajectory['category_id']})")
        print(f"fps={trajectory['fps']}")
        print(f"num_points={trajectory['num_points']}")
        print(f"visible_rgb_frames={trajectory['visible_rgb_frames']}")
        print(f"visible_thermal_frames={trajectory['visible_thermal_frames']}")
        print("points_preview=")

        preview = trajectory["points"][: args.limit_points]
        for point in preview:
            print(
                json.dumps(
                    point,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        print()

    if trajectories:
        total_points = sum(item["num_points"] for item in trajectories)
        avg_points = total_points / len(trajectories)
        print("=" * 80)
        print(f"total_points={total_points}")
        print(f"avg_points_per_trajectory={avg_points:.2f}")


if __name__ == "__main__":
    main()
