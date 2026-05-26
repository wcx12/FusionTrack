from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.schema import schema_diagnostics, validate_label_rows, validate_score_rows


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


def test_schema_diagnostics_reports_field_coverage_and_alignment_warnings() -> None:
    labels = [
        {"sample_id": "a", "label": 1, "frame_start": 1, "frame_end": 5},
        {"sample_id": "b", "label": 0},
    ]
    scores = [
        {"sample_id": "a", "score": 0.7, "frame_start": 1},
        {"sample_id": "a", "score": 0.9, "source": "rerank"},
        {"sample_id": "extra", "score": 0.2},
    ]

    diagnostics = schema_diagnostics(labels, scores, key_fields=("sample_id",))

    assert diagnostics["schema_diagnostics_version"] == 1
    assert diagnostics["key_fields"] == ["sample_id"]
    assert diagnostics["label"]["num_rows"] == 2
    assert diagnostics["score"]["num_rows"] == 3
    assert diagnostics["score"]["num_duplicate_keys"] == 1
    assert diagnostics["label"]["field_coverage"]["frame_start"] == {
        "present": 1,
        "missing": 1,
    }
    assert diagnostics["score"]["field_coverage"]["source"] == {
        "present": 1,
        "missing": 2,
    }
    assert diagnostics["alignment"]["num_missing_score_keys"] == 1
    assert diagnostics["alignment"]["num_extra_score_keys"] == 1
    assert diagnostics["status"] == "warning"
    assert diagnostics["warnings"] == [
        "missing_score_keys",
        "extra_score_keys",
        "duplicate_score_keys",
        "partial_label_field_coverage",
        "partial_score_field_coverage",
    ]
