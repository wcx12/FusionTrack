from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from baselines.group_classical import CLASSICAL_METHODS, run_classical_baseline
from baselines.group_prediction import run_prediction_baseline
from evaluation.io import load_jsonl, write_jsonl


BASELINES = ("classical", "prediction")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run group-level anomaly baselines on group window JSONL files."
    )
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help="Group window JSONL to score. Also used for classical training if --train-jsonl is omitted.",
    )
    parser.add_argument(
        "--score-jsonl",
        type=Path,
        help="Group window JSONL to score for classical baselines. Overrides --input-jsonl.",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        help="Training group window JSONL for classical baselines.",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--method", choices=CLASSICAL_METHODS, default="isolation_forest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_jsonl is None and args.score_jsonl is None:
        raise SystemExit("--input-jsonl or --score-jsonl is required")

    score_path = args.score_jsonl or args.input_jsonl
    score_windows = load_jsonl(score_path)
    if args.baseline == "prediction":
        rows = run_prediction_baseline(score_windows)
    else:
        train_path = args.train_jsonl or args.input_jsonl or score_path
        train_windows = load_jsonl(train_path)
        rows = run_classical_baseline(
            train_windows,
            score_windows,
            method=args.method,
            seed=args.seed,
            contamination=args.contamination,
        )
    write_jsonl(args.output_jsonl, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
