from __future__ import annotations

from pathlib import Path
import sys


if __package__:
    from ._compat import ensure_benchmark_on_path
else:  # pragma: no cover - direct script execution
    group_root = Path(__file__).resolve().parent
    group_root_str = str(group_root)
    if group_root_str not in sys.path:
        sys.path.insert(0, group_root_str)
    from _compat import ensure_benchmark_on_path

ensure_benchmark_on_path()

from fusiontrack.run_group_method import main as _benchmark_main


def main(argv: list[str] | None = None) -> int:
    return _benchmark_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
