from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from baselines.individual_classical import CLASSICAL_METHODS, run_classical_baseline
from evaluation.io import load_jsonl, write_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run individual-level classical anomaly baselines on trajectory JSONL files."
    )
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--score-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=CLASSICAL_METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    train_trajectories = load_jsonl(args.train_jsonl)
    score_trajectories = load_jsonl(args.score_jsonl)
    rows = run_classical_baseline(
        train_trajectories,
        score_trajectories,
        method=args.method,
        seed=args.seed,
        contamination=args.contamination,
    )
    write_jsonl(args.output_jsonl, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
