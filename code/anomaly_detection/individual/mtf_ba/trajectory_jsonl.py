from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def iter_trajectory_jsonl(jsonl_path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield one object-centric trajectory record per JSONL line."""
    jsonl_path = Path(jsonl_path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_trajectory_jsonl(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load all object-centric trajectories from a JSONL file."""
    return list(iter_trajectory_jsonl(jsonl_path))
