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
from external_sources.lmtad_adapters import write_lmtad_official_inputs


DESCRIPTION = """Convert benchmark trajectory JSONL to LM-TAD official intermediate input files.

The inspected official LMTAD checkout is bound to Porto and Pattern-of-Life
dataset loaders. This runner writes a manifest, sequence JSONL, and vocab for an
external LMTAD checkout; the checkout still needs a custom dataset loader that
reads those files before running the official train/eval scripts.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trajectory-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--grid-size", default=25.0, type=float)
    parser.add_argument("--sequence-filename", default="lmtad_sequences.jsonl")
    parser.add_argument("--manifest-filename", default="manifest.json")
    parser.add_argument("--vocab-filename", default="vocab.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = write_lmtad_official_inputs(
        load_jsonl(args.trajectory_jsonl),
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        sequence_filename=args.sequence_filename,
        manifest_filename=args.manifest_filename,
        vocab_filename=args.vocab_filename,
    )
    print(
        json.dumps(
            {
                "trajectory_jsonl": str(args.trajectory_jsonl),
                "output_dir": str(args.output_dir),
                "manifest_json": str(args.output_dir / args.manifest_filename),
                "sequence_jsonl": str(args.output_dir / args.sequence_filename),
                "vocab_json": str(args.output_dir / args.vocab_filename),
                "num_trajectories": len(manifest["trajectories"]),
                "grid_size": float(args.grid_size),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
