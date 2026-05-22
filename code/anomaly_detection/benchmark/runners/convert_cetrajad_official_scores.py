from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from external_sources.cetrajad_adapters import convert_cetrajad_scores_to_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CETrajAD official/external score CSV, JSON, or JSONL into "
            "benchmark score JSONL. Common ID columns include sample_id, "
            "trajectory_id, and track_id; common score columns include score "
            "and anomaly_score."
        )
    )
    parser.add_argument("--cetrajad-scores", required=True, type=Path)
    parser.add_argument("--sidecar-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--score-column")
    parser.add_argument("--id-column")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = convert_cetrajad_scores_to_jsonl(
        cetrajad_scores=args.cetrajad_scores,
        sidecar_json=args.sidecar_json,
        output_jsonl=args.output_jsonl,
        score_column=args.score_column,
        id_column=args.id_column,
    )
    print(
        json.dumps(
            {
                "cetrajad_scores": str(args.cetrajad_scores),
                "sidecar_json": str(args.sidecar_json),
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
