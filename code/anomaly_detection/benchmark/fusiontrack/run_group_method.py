from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__:
    from .group_scoring import score_group_windows
else:  # pragma: no cover - direct script execution
    _BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
    _benchmark_root = str(_BENCHMARK_ROOT)
    if _benchmark_root not in sys.path:
        sys.path.insert(0, _benchmark_root)
    from fusiontrack.group_scoring import score_group_windows


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score FusionTrack group graph windows.")
    parser.add_argument("input_jsonl", type=Path, help="Input group window JSONL file.")
    parser.add_argument("output_jsonl", type=Path, help="Output score JSONL file.")
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--rho-p", type=float, default=float("inf"))
    parser.add_argument("--rho-v", type=float, default=float("inf"))
    parser.add_argument("--eta", type=float, default=0.5)
    args = parser.parse_args(argv)

    rows = score_group_windows(
        iter_jsonl(args.input_jsonl),
        k_neighbors=args.k_neighbors,
        rho_p=args.rho_p,
        rho_v=args.rho_v,
        eta=args.eta,
    )
    write_jsonl(args.output_jsonl, rows)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
