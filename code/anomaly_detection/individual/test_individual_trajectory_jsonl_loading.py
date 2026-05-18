#!/usr/bin/env python3
"""
Simple test/demo for loading exported object-centric trajectory JSONL files.

Focus:
1. confirm the JSONL file can be read
2. inspect trajectory-level metadata
3. verify both RGB and thermal modalities are present in trajectory points
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtf_ba.trajectory_jsonl import iter_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test loading exported individual trajectory JSONL files."
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path("outputs")
        / "vt_tiny_mot_individual"
        / "individual_trajectories_train.jsonl",
        help="Path to individual_trajectories_<split>.jsonl.",
    )
    parser.add_argument(
        "--limit-trajectories",
        type=int,
        default=3,
        help="How many trajectories to preview.",
    )
    parser.add_argument(
        "--limit-points",
        type=int,
        default=3,
        help="How many points to preview per trajectory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    num_trajectories = 0
    total_points = 0
    total_rgb_visible = 0
    total_thermal_visible = 0

    print(f"jsonl_path={args.jsonl_path.resolve()}")
    print()

    for trajectory in iter_trajectory_jsonl(args.jsonl_path):
        num_trajectories += 1
        total_points += trajectory["num_points"]
        total_rgb_visible += trajectory["visible_rgb_frames"]
        total_thermal_visible += trajectory["visible_thermal_frames"]

        if num_trajectories <= args.limit_trajectories:
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

            for point in trajectory["points"][: args.limit_points]:
                compact_point = {
                    "frame_id": point["frame_id"],
                    "rgb_present": point["rgb"] is not None,
                    "thermal_present": point["thermal"] is not None,
                    "rgb_center_xy": (
                        None if point["rgb"] is None else point["rgb"]["center_xy"]
                    ),
                    "thermal_center_xy": (
                        None
                        if point["thermal"] is None
                        else point["thermal"]["center_xy"]
                    ),
                    "modal_offset_distance": point["modal"]["offset_distance"],
                }
                print(json.dumps(compact_point, ensure_ascii=False, indent=2))
            print()

    print("=" * 80)
    print(f"num_trajectories={num_trajectories}")
    print(f"total_points={total_points}")
    print(f"total_rgb_visible={total_rgb_visible}")
    print(f"total_thermal_visible={total_thermal_visible}")
    if num_trajectories:
        print(f"avg_points_per_trajectory={total_points / num_trajectories:.2f}")


if __name__ == "__main__":
    main()
