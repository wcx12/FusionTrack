from pathlib import Path
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import (
    alignment_report,
    align_scores_with_labels,
    best_f1_threshold,
    evaluate_binary_scores,
    event_f1,
    precision_recall_at_k,
)


def test_precision_recall_at_k_counts_top_ranked_hits() -> None:
    result = precision_recall_at_k(
        y_true=[1, 0, 1, 0],
        y_score=[0.9, 0.8, 0.4, 0.1],
        k=2,
    )

    assert result == {
        "precision_at_k": 0.5,
        "recall_at_k": 0.5,
        "hits_at_k": 1,
        "k": 2,
    }


def test_best_f1_threshold_finds_separating_score() -> None:
    result = best_f1_threshold(
        y_true=[0, 1, 1, 0],
        y_score=[0.1, 0.8, 0.7, 0.2],
    )

    assert result["threshold"] == pytest.approx(0.7)
    assert result["f1"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_evaluate_binary_scores_returns_core_metrics() -> None:
    result = evaluate_binary_scores(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.4, 0.35, 0.8],
        k=2,
    )

    assert result["num_positive"] == 2
    assert result["num_total"] == 4
    assert result["auroc"] == pytest.approx(0.75)
    assert 0.0 <= result["auprc"] <= 1.0
    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(0.5)
    assert 0.0 <= result["f1"] <= 1.0
    assert result["threshold"] in {0.35, 0.4, 0.8}


def test_align_scores_with_labels_collapses_duplicates_and_demotes_missing_scores() -> None:
    label_rows = [
        {"sample_id": "a", "label": 0},
        {"sample_id": "a", "label": 1},
        {"sample_id": "b", "label": 0},
        {"sample_id": "c", "label": 1},
    ]
    score_rows = [
        {"sample_id": "a", "score": 0.2},
        {"sample_id": "a", "score": 0.9},
        {"sample_id": "b", "score": 0.4},
        {"sample_id": "extra", "score": 1.0},
    ]

    y_true, y_score = align_scores_with_labels(score_rows, label_rows)

    assert y_true == [1, 0, 1]
    assert y_score[:2] == [0.9, 0.4]
    assert math.isinf(y_score[2])
    assert y_score[2] < 0


def test_alignment_report_counts_duplicate_missing_and_extra_keys() -> None:
    label_rows = [
        {"sample_id": "a", "label": 0},
        {"sample_id": "a", "label": 1},
        {"sample_id": "b", "label": 0},
        {"sample_id": "c", "label": 1},
    ]
    score_rows = [
        {"sample_id": "a", "score": 0.2},
        {"sample_id": "a", "score": 0.9},
        {"sample_id": "b", "score": 0.4},
        {"sample_id": "extra", "score": 1.0},
    ]

    report = alignment_report(score_rows, label_rows)

    assert report == {
        "num_label_rows": 4,
        "num_score_rows": 4,
        "num_unique_label_keys": 3,
        "num_unique_score_keys": 3,
        "num_duplicate_label_keys": 1,
        "num_duplicate_score_keys": 1,
        "num_missing_score_keys": 1,
        "num_extra_score_keys": 1,
    }


def test_align_scores_with_labels_can_reject_duplicate_keys() -> None:
    label_rows = [{"sample_id": "a", "label": 1}]
    duplicate_score_rows = [
        {"sample_id": "a", "score": 0.2},
        {"sample_id": "a", "score": 0.9},
    ]

    with pytest.raises(ValueError, match="Duplicate score keys"):
        align_scores_with_labels(
            duplicate_score_rows,
            label_rows,
            require_unique_score_keys=True,
        )

    with pytest.raises(ValueError, match="Duplicate label keys"):
        align_scores_with_labels(
            [{"sample_id": "a", "score": 0.2}],
            [{"sample_id": "a", "label": 0}, {"sample_id": "a", "label": 1}],
            require_unique_label_keys=True,
        )


def test_event_f1_greedily_matches_events_by_track_iou() -> None:
    true_events = [
        {"sequence": "seq_1", "track_id": "1", "frame_start": 10, "frame_end": 20},
        {"sequence": "seq_1", "track_id": "2", "frame_start": 30, "frame_end": 40},
    ]
    pred_events = [
        {"sequence": "seq_1", "track_id": "1", "frame_start": 12, "frame_end": 18},
        {"sequence": "seq_1", "track_id": "1", "frame_start": 50, "frame_end": 60},
        {"sequence": "seq_2", "track_id": "2", "frame_start": 30, "frame_end": 40},
    ]

    result = event_f1(pred_events, true_events, iou_threshold=0.5)

    assert result["event_precision"] == pytest.approx(1 / 3)
    assert result["event_recall"] == pytest.approx(1 / 2)
    assert result["event_f1"] == pytest.approx(0.4)
