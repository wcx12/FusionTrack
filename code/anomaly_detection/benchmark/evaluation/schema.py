from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any


OPTIONAL_INT_FIELDS = ("frame_start", "frame_end", "injection_seed")
SCHEMA_DIAGNOSTICS_VERSION = 1


def schema_diagnostics(
    label_rows: Sequence[dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
    key_fields: Sequence[str] = ("sample_id",),
) -> dict[str, Any]:
    label_rows = list(label_rows)
    score_rows = list(score_rows)
    label_counts = _key_counts(label_rows, key_fields)
    score_counts = _key_counts(score_rows, key_fields)
    label_keys = set(label_counts)
    score_keys = set(score_counts)
    alignment = {
        "num_missing_score_keys": len(label_keys - score_keys),
        "num_extra_score_keys": len(score_keys - label_keys),
    }
    warnings = _diagnostic_warnings(
        label_counts=label_counts,
        score_counts=score_counts,
        alignment=alignment,
        label_coverage=_field_coverage(label_rows),
        score_coverage=_field_coverage(score_rows),
    )
    return {
        "schema_diagnostics_version": SCHEMA_DIAGNOSTICS_VERSION,
        "status": "ok" if not warnings else "warning",
        "key_fields": list(key_fields),
        "label": {
            "num_rows": len(label_rows),
            "num_unique_keys": len(label_keys),
            "num_duplicate_keys": _num_duplicate_keys(label_counts),
            "field_coverage": _field_coverage(label_rows),
        },
        "score": {
            "num_rows": len(score_rows),
            "num_unique_keys": len(score_keys),
            "num_duplicate_keys": _num_duplicate_keys(score_counts),
            "field_coverage": _field_coverage(score_rows),
        },
        "alignment": alignment,
        "warnings": warnings,
    }


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


def _key_counts(
    rows: Sequence[dict[str, Any]],
    key_fields: Sequence[str],
) -> Counter[tuple[Any, ...]]:
    return Counter(tuple(row.get(field) for field in key_fields) for row in rows)


def _num_duplicate_keys(counts: Counter[tuple[Any, ...]]) -> int:
    return sum(1 for count in counts.values() if count > 1)


def _field_coverage(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = sorted({field for row in rows for field in row})
    total = len(rows)
    coverage: dict[str, dict[str, int]] = {}
    for field in fields:
        present = sum(1 for row in rows if field in row and row[field] not in (None, ""))
        coverage[field] = {"present": present, "missing": total - present}
    return coverage


def _has_partial_coverage(coverage: dict[str, dict[str, int]]) -> bool:
    return any(
        int(values.get("present", 0)) > 0 and int(values.get("missing", 0)) > 0
        for values in coverage.values()
    )


def _diagnostic_warnings(
    label_counts: Counter[tuple[Any, ...]],
    score_counts: Counter[tuple[Any, ...]],
    alignment: dict[str, int],
    label_coverage: dict[str, dict[str, int]],
    score_coverage: dict[str, dict[str, int]],
) -> list[str]:
    warnings: list[str] = []
    if alignment["num_missing_score_keys"] > 0:
        warnings.append("missing_score_keys")
    if alignment["num_extra_score_keys"] > 0:
        warnings.append("extra_score_keys")
    if _num_duplicate_keys(label_counts) > 0:
        warnings.append("duplicate_label_keys")
    if _num_duplicate_keys(score_counts) > 0:
        warnings.append("duplicate_score_keys")
    if _has_partial_coverage(label_coverage):
        warnings.append("partial_label_field_coverage")
    if _has_partial_coverage(score_coverage):
        warnings.append("partial_score_field_coverage")
    return warnings
