from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.io import load_jsonl
from external_sources.official_adapters import write_pidpm_trajectory_csv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert benchmark trajectory JSONL to Pi-DPM official flattened CSV input."
    )
    parser.add_argument("--trajectory-jsonl", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--sidecar-json", required=True, type=Path)
    parser.add_argument("--max-points", default=32, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sidecar_rows = write_pidpm_trajectory_csv(
        load_jsonl(args.trajectory_jsonl),
        output_csv=args.output_csv,
        sidecar_json=args.sidecar_json,
        max_points=args.max_points,
    )
    print(
        json.dumps(
            {
                "trajectory_jsonl": str(args.trajectory_jsonl),
                "output_csv": str(args.output_csv),
                "sidecar_json": str(args.sidecar_json),
                "num_trajectories": len(sidecar_rows),
                "max_points": int(args.max_points),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
