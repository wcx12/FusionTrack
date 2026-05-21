from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from external_sources.lmtad_adapters import convert_lmtad_scores_to_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert official LM-TAD score CSV/TSV/JSON/JSONL output to "
            "benchmark score JSONL."
        )
    )
    parser.add_argument("--lmtad-scores", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--score-column")
    parser.add_argument("--id-column")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = convert_lmtad_scores_to_jsonl(
        scores_path=args.lmtad_scores,
        output_jsonl=args.output_jsonl,
        manifest_json=args.manifest_json,
        score_column=args.score_column,
        id_column=args.id_column,
    )
    print(
        json.dumps(
            {
                "lmtad_scores": str(args.lmtad_scores),
                "manifest_json": str(args.manifest_json) if args.manifest_json else None,
                "output_jsonl": str(args.output_jsonl),
                "num_scores": len(rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
