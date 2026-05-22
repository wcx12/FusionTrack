from __future__ import annotations

from ._compat import ensure_benchmark_on_path

ensure_benchmark_on_path()

from fusiontrack.group_scoring import COMPONENT_NAMES, score_group_windows

__all__ = [
    "COMPONENT_NAMES",
    "score_group_windows",
]
