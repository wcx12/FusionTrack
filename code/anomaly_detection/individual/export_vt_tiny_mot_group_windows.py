#!/usr/bin/env python3
"""
Export scene/window samples reserved for future group anomaly detection.

Input:
  outputs/vt_tiny_mot_trajectories/observations_<split>.csv

Outputs:
  group_windows_<split>.jsonl
  group_windows_summary_<split>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtf_ba.group_interface import (
    GroupWindowConfig,
    iter_group_windows,
    write_group_windows_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export group/window samples from observations CSV."
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
        default=Path("outputs") / "vt_tiny_mot_group",
        help="Directory for group-window exports.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name used in output file naming.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("sequence", "window"),
        default="window",
        help="Use full sequences or fixed-length windows as group samples.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Number of frames per group window when sample-mode=window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Sliding-window stride in frames when sample-mode=window.",
    )
    parser.add_argument(
        "--min-visible-frames",
        type=int,
        default=2,
        help="Keep objects visible in at least this many frames in either modality.",
    )
    parser.add_argument(
        "--require-both-modalities",
        action="store_true",
        help="Keep only objects visible in both RGB and thermal inside the sample.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Optional sequence name filter for debugging one sequence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GroupWindowConfig(
        sample_mode=args.sample_mode,
        window_size=args.window_size,
        stride=args.stride,
        min_visible_frames=args.min_visible_frames,
        require_both_modalities=args.require_both_modalities,
    )
    jsonl_path = output_dir / f"group_windows_{args.split}.jsonl"
    summary_path = output_dir / f"group_windows_summary_{args.split}.json"

    windows = iter_group_windows(
        csv_path=args.csv_path,
        config=config,
        sequence=args.sequence,
    )
    stats = write_group_windows_jsonl(jsonl_path, windows)

    summary = {
        "csv_path": str(args.csv_path.resolve()),
        "output_jsonl": str(jsonl_path),
        "split": args.split,
        "sample_mode": args.sample_mode,
        "window_size": args.window_size,
        "stride": args.stride,
        "min_visible_frames": args.min_visible_frames,
        "require_both_modalities": args.require_both_modalities,
        "sequence": args.sequence,
        **stats,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
