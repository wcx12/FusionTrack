from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.schema import validate_label_rows, validate_score_rows


def test_validate_score_rows_rejects_missing_and_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="score row 1.*missing required field 'score'"):
        validate_score_rows([{"sample_id": "a"}])

    with pytest.raises(ValueError, match="score row 1.*finite numeric score"):
        validate_score_rows([{"sample_id": "a", "score": "nan"}])


def test_validate_label_rows_rejects_bad_binary_label_and_frame_range() -> None:
    with pytest.raises(ValueError, match="label row 1.*binary label"):
        validate_label_rows([{"sample_id": "a", "label": 2}])

    with pytest.raises(ValueError, match="label row 1.*frame_end.*before frame_start"):
        validate_label_rows([{"sample_id": "a", "label": 1, "frame_start": 8, "frame_end": 3}])


def test_schema_validation_uses_task_key_fields_and_duplicate_policy() -> None:
    rows = [
        {"sample_id": "seq:0", "window_id": "0-15", "label": 1},
        {"sample_id": "seq:0", "window_id": "0-15", "label": 0},
    ]

    validate_label_rows(rows, key_fields=("sample_id", "window_id"))
    with pytest.raises(ValueError, match="Duplicate label keys"):
        validate_label_rows(
            rows,
            key_fields=("sample_id", "window_id"),
            require_unique_keys=True,
        )

    with pytest.raises(ValueError, match="missing required key field 'window_id'"):
        validate_score_rows(
            [{"sample_id": "seq:0", "score": 0.5}],
            key_fields=("sample_id", "window_id"),
        )
