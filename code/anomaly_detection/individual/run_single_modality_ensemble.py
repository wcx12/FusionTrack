#!/usr/bin/env python3
"""
Run baseline-style ensemble over the six single-modality detector scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ensemble over route/speed/shape rgb/thermal detector scores."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Which split to ensemble.",
    )
    parser.add_argument(
        "--score-root-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_scores",
        help="Directory containing per-detector scoring outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_ensemble",
        help="Directory where ensemble outputs will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Top-k size used in Jaccard ranking comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    from mtf_ba.ensemble_scoring import run_ensemble

    args = parse_args()
    summary = run_ensemble(
        score_root_dir=args.score_root_dir,
        split=args.split,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
