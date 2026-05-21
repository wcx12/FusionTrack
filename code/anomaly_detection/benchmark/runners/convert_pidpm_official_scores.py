from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from external_sources.official_adapters import convert_pidpm_scores_to_jsonl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Pi-DPM official anomaly_scores.csv to benchmark score JSONL."
    )
    parser.add_argument("--pidpm-scores-csv", required=True, type=Path)
    parser.add_argument("--sidecar-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = convert_pidpm_scores_to_jsonl(
        pidpm_scores_csv=args.pidpm_scores_csv,
        sidecar_json=args.sidecar_json,
        output_jsonl=args.output_jsonl,
    )
    print(
        json.dumps(
            {
                "pidpm_scores_csv": str(args.pidpm_scores_csv),
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
