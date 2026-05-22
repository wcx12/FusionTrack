from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.tables import export_report_tables


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper-ready Markdown, LaTeX, and ranked CSV tables from benchmark summary.csv."
    )
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--metric", default="auprc")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_report_tables(
        summary_csv=args.summary_csv,
        output_dir=args.output_dir,
        metric=args.metric,
    )
    print(
        json.dumps(
            {key: str(path) for key, path in manifest.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
