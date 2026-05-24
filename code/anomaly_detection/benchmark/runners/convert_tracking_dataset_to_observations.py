from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from dataset_adapters.tracking_observations import (  # noqa: E402
    convert_mot_roots_to_observations,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert M3OT or MOT-family gt/gt.txt annotations into the "
            "FusionTrack observations_<split>.csv format."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name recorded in the output CSV, e.g. M3OT or MOT17.",
    )
    parser.add_argument(
        "--profile",
        default="motchallenge",
        choices=("motchallenge", "m3ot", "dancetrack", "sportsmot"),
        help="Column/profile defaults for the source annotation files.",
    )
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=None,
        help="Single-modality MOT-style root. Equivalent to --rgb-root.",
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=None,
        help="RGB/visible MOT-style root containing sequence/gt/gt.txt files.",
    )
    parser.add_argument(
        "--thermal-root",
        type=Path,
        default=None,
        help="Thermal/IR MOT-style root containing sequence/gt/gt.txt files.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter; keeps gt.txt files whose path contains this split.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS override when seqinfo.ini is absent or unreliable.",
    )
    parser.add_argument(
        "--frame-digits",
        type=int,
        default=None,
        help="Optional frame filename width override. Defaults to 8 for DanceTrack and 6 otherwise.",
    )
    parser.add_argument(
        "--keep-category-id",
        action="append",
        type=int,
        default=None,
        help="Category ID to keep. Repeat for multiple IDs. Defaults to class 1 for supported profiles.",
    )
    parser.add_argument(
        "--include-all-categories",
        action="store_true",
        help="Disable default category filtering.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=None,
        help="Optional sequence name filter. Repeat to keep multiple sequences.",
    )
    parser.add_argument(
        "--sequence-name-source",
        choices=("directory", "seqinfo", "relative"),
        default="directory",
        help=(
            "How to derive sequence IDs. directory avoids seqinfo.ini name collisions; "
            "seqinfo preserves source metadata; relative keeps split/path context."
        ),
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Keep rows whose MOT confidence/valid flag is <= 0.",
    )
    parser.add_argument(
        "--require-paired-modalities",
        action="store_true",
        help="For RGB/thermal conversion, keep only rows present in both modalities.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing an empty observations CSV. Intended only for debugging filters.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output observations_<split>.csv path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for conversion summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgb_root = args.rgb_root or args.mot_root
    summary = convert_mot_roots_to_observations(
        output_csv=args.output_csv,
        dataset=args.dataset,
        rgb_root=rgb_root,
        thermal_root=args.thermal_root,
        profile=args.profile,
        split=args.split,
        fps=args.fps,
        frame_digits=args.frame_digits,
        keep_category_ids=args.keep_category_id,
        use_default_category_filter=not args.include_all_categories,
        include_ignored=args.include_ignored,
        sequences=args.sequence,
        sequence_name_source=args.sequence_name_source,
        require_paired_modalities=args.require_paired_modalities,
        allow_empty=args.allow_empty,
    )
    if args.summary_json is not None:
        write_summary_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
