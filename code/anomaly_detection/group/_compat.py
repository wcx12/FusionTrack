from __future__ import annotations

from pathlib import Path
import sys


def ensure_benchmark_on_path() -> Path:
    benchmark_root = Path(__file__).resolve().parents[1] / "benchmark"
    benchmark_root_str = str(benchmark_root)
    if benchmark_root_str not in sys.path:
        sys.path.insert(0, benchmark_root_str)
    return benchmark_root
