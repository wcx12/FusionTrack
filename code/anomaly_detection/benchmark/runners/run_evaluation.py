from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.reporting import evaluate_score_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly score files against injected-label JSONL/CSV files."
    )
    parser.add_argument("--label-file", required=True, type=Path)
    parser.add_argument("--score-file", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--key-fields",
        nargs="+",
        default=["sample_id"],
        help="Fields used to align labels and scores; defaults to sample_id.",
    )
    parser.add_argument(
        "--require-unique-keys",
        action="store_true",
        help="Fail when label or score rows contain duplicate alignment keys.",
    )
    parser.add_argument(
        "--require-score-key-match",
        action="store_true",
        help="Fail when score keys do not exactly match label keys.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = evaluate_score_file(
        score_path=args.score_file,
        label_path=args.label_file,
        key_fields=tuple(args.key_fields),
        k=args.k,
        require_unique_keys=args.require_unique_keys,
        require_score_key_match=args.require_score_key_match,
    )
    metrics.update(
        {
            "method": args.method,
            "source": str(args.score_file),
            "split": args.split,
            "seed": int(args.seed),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
