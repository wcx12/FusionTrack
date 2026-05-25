from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from dataset_adapters.real_labels import normalize_real_label_rows
from evaluation.io import load_label_rows, write_jsonl
from evaluation.schema import validate_label_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize external real anomaly labels into FusionTrack label JSONL schema."
    )
    parser.add_argument("--level", required=True, choices=("individual", "group"))
    parser.add_argument("--input-labels", required=True, type=Path)
    parser.add_argument("--output-labels", required=True, type=Path)
    parser.add_argument(
        "--allow-duplicate-keys",
        action="store_true",
        help="Allow duplicate label keys; by default real labels must be unique.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_label_rows(args.input_labels)
    normalized = normalize_real_label_rows(rows, level=args.level)
    key_fields = ("sample_id", "window_id") if args.level == "group" else ("sample_id",)
    validate_label_rows(
        normalized,
        key_fields=key_fields,
        require_unique_keys=not bool(args.allow_duplicate_keys),
    )
    write_jsonl(args.output_labels, normalized)
    summary = {
        "level": args.level,
        "input_labels": str(args.input_labels),
        "output_labels": str(args.output_labels),
        "key_fields": list(key_fields),
        "num_labels": len(normalized),
        "num_positive": sum(1 for row in normalized if int(row.get("label", 0) or 0) == 1),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
