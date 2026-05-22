from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


_LABEL_INT_FIELDS = ("label", "frame_start", "frame_end", "injection_seed")
_SCORE_INT_FIELDS = ("label", "frame_start", "frame_end", "injection_seed")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_label_rows(path: Path) -> list[dict[str, Any]]:
    rows = _load_rows(path)
    return [_coerce_int_fields(row, _LABEL_INT_FIELDS) for row in rows]


def load_score_rows(path: Path) -> list[dict[str, Any]]:
    rows = _load_rows(path)
    converted = [_coerce_int_fields(row, _SCORE_INT_FIELDS) for row in rows]
    for row in converted:
        if "score" in row and row["score"] not in (None, ""):
            row["score"] = float(row["score"])
    return converted


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported file extension for {path}; expected .jsonl or .csv")


def _coerce_int_fields(row: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    converted = dict(row)
    for field in fields:
        value = converted.get(field)
        if value not in (None, ""):
            converted[field] = int(value)
    return converted
