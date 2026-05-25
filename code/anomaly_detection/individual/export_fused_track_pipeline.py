#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtf_ba.fused_track_pipeline import (
    FusedTrackPipelineConfig,
    TrackQualityConfig,
    run_fused_track_pipeline,
)
from mtf_ba.group_interface import GroupWindowConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the reusable FusionTrack trajectory package from an observations CSV. "
            "Outputs individual trajectories, fused trajectories, group windows, summary, and manifest."
        )
    )
    parser.add_argument("--csv-path", type=Path, required=True, help="Input observations_<split>.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for pipeline artifacts.")
    parser.add_argument("--split", default="train", help="Split name used in output file names.")
    parser.add_argument("--offset-scale", type=float, default=25.0, help="Cross-modal offset scale for confidence.")
    parser.add_argument(
        "--sample-mode",
        choices=("sequence", "window"),
        default="window",
        help="Group sample mode.",
    )
    parser.add_argument("--window-size", type=int, default=16, help="Group window size.")
    parser.add_argument("--stride", type=int, default=8, help="Group window stride.")
    parser.add_argument(
        "--min-visible-frames",
        type=int,
        default=2,
        help="Minimum visible frames for each object in a group sample.",
    )
    parser.add_argument(
        "--require-both-modalities",
        action="store_true",
        help="Keep group objects only when both RGB and thermal are visible.",
    )
    parser.add_argument(
        "--min-track-points",
        type=int,
        default=1,
        help="Minimum points required to keep a fused trajectory.",
    )
    parser.add_argument(
        "--min-track-visible-frames",
        type=int,
        default=1,
        help="Minimum frames with either RGB or thermal visible.",
    )
    parser.add_argument(
        "--max-track-frame-gap",
        type=int,
        default=None,
        help="Maximum allowed gap between adjacent frames; unset disables this filter.",
    )
    parser.add_argument(
        "--min-track-fused-ratio",
        type=float,
        default=0.0,
        help="Minimum ratio of points that can produce a fused state.",
    )
    parser.add_argument(
        "--keep-filtered-tracks",
        action="store_true",
        help="Keep filtered tracks in outputs while still recording quality drop reasons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_fused_track_pipeline(
        args.csv_path,
        args.output_dir,
        FusedTrackPipelineConfig(
            split=args.split,
            offset_scale=args.offset_scale,
            group=GroupWindowConfig(
                sample_mode=args.sample_mode,
                window_size=args.window_size,
                stride=args.stride,
                min_visible_frames=args.min_visible_frames,
                require_both_modalities=args.require_both_modalities,
            ),
            quality=TrackQualityConfig(
                min_points=args.min_track_points,
                min_visible_any_frames=args.min_track_visible_frames,
                max_frame_gap=args.max_track_frame_gap,
                min_fused_ratio=args.min_track_fused_ratio,
                keep_filtered=args.keep_filtered_tracks,
            ),
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
