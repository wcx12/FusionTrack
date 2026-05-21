from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.reporting import summarize_metric_files


ALIGNMENT_FIELDS = (
    "num_duplicate_label_keys",
    "num_duplicate_score_keys",
    "num_missing_score_keys",
    "num_extra_score_keys",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit metrics alignment fields and merge metric JSON files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Metric JSON files or directories containing metric JSON files.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--allow-alignment-issues",
        action="store_true",
        help="Merge metrics even when alignment audit fields are non-zero.",
    )
    return parser.parse_args(argv)


def collect_metric_files(inputs: Sequence[Path]) -> list[Path]:
    metric_files: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            metric_files.extend(
                sorted(path for path in input_path.rglob("*.json") if path.is_file())
            )
        elif input_path.is_file():
            metric_files.append(input_path)
        else:
            raise ValueError(f"Metrics input does not exist: {input_path}")
    if not metric_files:
        raise ValueError("No metric JSON files found")
    return metric_files


def audit_alignment(rows: Sequence[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for index, row in enumerate(rows, start=1):
        source = str(row.get("source") or row.get("method") or f"row {index}")
        for field in ALIGNMENT_FIELDS:
            value = int(row.get(field, 0))
            if value != 0:
                issues.append(f"{source}: {field}={value}")
    return issues


def merge_metric_files(
    metric_files: Sequence[Path],
    output_csv: Path | None = None,
    output_json: Path | None = None,
    allow_alignment_issues: bool = False,
) -> list[dict[str, Any]]:
    rows = summarize_metric_files(metric_files)
    issues = audit_alignment(rows)
    if issues and not allow_alignment_issues:
        raise ValueError("Alignment audit failed: " + "; ".join(issues))

    if output_csv is not None:
        summarize_metric_files(metric_files, output_csv=output_csv)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metric_files = collect_metric_files(args.inputs)
        rows = merge_metric_files(
            metric_files,
            output_csv=args.output_csv,
            output_json=args.output_json,
            allow_alignment_issues=args.allow_alignment_issues,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
