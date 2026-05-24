from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any


OPTIONAL_INT_FIELDS = ("frame_start", "frame_end", "injection_seed")


def validate_label_rows(
    rows: Sequence[dict[str, Any]],
    key_fields: Sequence[str] = ("sample_id",),
    require_unique_keys: bool = False,
) -> list[dict[str, Any]]:
    validated = list(rows)
    _validate_key_fields(validated, key_fields, row_kind="label")
    for row_index, row in enumerate(validated, start=1):
        label = row.get("label")
        if "label" not in row or not _is_binary_label(label):
            raise ValueError(f"label row {row_index} must contain a binary label")
        _validate_optional_int_fields(row, row_index, row_kind="label")
        _validate_frame_range(row, row_index, row_kind="label")
    if require_unique_keys:
        _raise_on_duplicate_keys(validated, key_fields, row_kind="label")
    return validated


def validate_score_rows(
    rows: Sequence[dict[str, Any]],
    key_fields: Sequence[str] = ("sample_id",),
    require_unique_keys: bool = False,
) -> list[dict[str, Any]]:
    validated = list(rows)
    _validate_key_fields(validated, key_fields, row_kind="score")
    for row_index, row in enumerate(validated, start=1):
        if "score" not in row:
            raise ValueError(f"score row {row_index} is missing required field 'score'")
        if not _is_finite_score(row.get("score")):
            raise ValueError(f"score row {row_index} must contain a finite numeric score")
        _validate_optional_int_fields(row, row_index, row_kind="score")
        _validate_frame_range(row, row_index, row_kind="score")
    if require_unique_keys:
        _raise_on_duplicate_keys(validated, key_fields, row_kind="score")
    return validated


def _validate_key_fields(
    rows: Sequence[dict[str, Any]],
    key_fields: Sequence[str],
    row_kind: str,
) -> None:
    if not key_fields:
        raise ValueError("key_fields must contain at least one field")
    for row_index, row in enumerate(rows, start=1):
        for field in key_fields:
            if field not in row or row[field] in (None, ""):
                raise ValueError(
                    f"{row_kind} row {row_index} is missing required key field '{field}'"
                )


def _validate_optional_int_fields(
    row: dict[str, Any],
    row_index: int,
    row_kind: str,
) -> None:
    for field in OPTIONAL_INT_FIELDS:
        value = row.get(field)
        if value in (None, ""):
            continue
        try:
            int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{row_kind} row {row_index} field '{field}' must be an integer"
            ) from exc


def _validate_frame_range(row: dict[str, Any], row_index: int, row_kind: str) -> None:
    start = row.get("frame_start")
    end = row.get("frame_end")
    if start in (None, "") or end in (None, ""):
        return
    if int(end) < int(start):
        raise ValueError(
            f"{row_kind} row {row_index} has frame_end before frame_start"
        )


def _is_binary_label(value: Any) -> bool:
    if value in (None, ""):
        return False
    try:
        return int(value) in (0, 1) and float(value) == int(value)
    except (TypeError, ValueError):
        return False


def _is_finite_score(value: Any) -> bool:
    if value in (None, ""):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _raise_on_duplicate_keys(
    rows: Sequence[dict[str, Any]],
    key_fields: Sequence[str],
    row_kind: str,
) -> None:
    counts: Counter[tuple[Any, ...]] = Counter(
        tuple(row[field] for field in key_fields) for row in rows
    )
    duplicates = [key for key, count in counts.items() if count > 1]
    if not duplicates:
        return
    preview = ", ".join(repr(key) for key in duplicates[:5])
    if len(duplicates) > 5:
        preview += ", ..."
    raise ValueError(
        f"Duplicate {row_kind} keys for key_fields {tuple(key_fields)}: {preview}"
    )
