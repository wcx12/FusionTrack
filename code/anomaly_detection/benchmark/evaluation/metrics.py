from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any


def alignment_report(
    score_rows: Iterable[dict[str, Any]],
    label_rows: Iterable[dict[str, Any]],
    key_fields: Sequence[str] = ("sample_id",),
) -> dict[str, int]:
    label_counts, num_label_rows = _key_counts(
        label_rows,
        key_fields,
        row_kind="label",
    )
    score_counts, num_score_rows = _key_counts(
        score_rows,
        key_fields,
        row_kind="score",
    )
    label_keys = set(label_counts)
    score_keys = set(score_counts)
    return {
        "num_label_rows": num_label_rows,
        "num_score_rows": num_score_rows,
        "num_unique_label_keys": len(label_keys),
        "num_unique_score_keys": len(score_keys),
        "num_duplicate_label_keys": _num_duplicate_keys(label_counts),
        "num_duplicate_score_keys": _num_duplicate_keys(score_counts),
        "num_missing_score_keys": len(label_keys - score_keys),
        "num_extra_score_keys": len(score_keys - label_keys),
    }


def align_scores_with_labels(
    score_rows: Iterable[dict[str, Any]],
    label_rows: Iterable[dict[str, Any]],
    key_fields: Sequence[str] = ("sample_id",),
    require_unique_label_keys: bool = False,
    require_unique_score_keys: bool = False,
) -> tuple[list[int], list[float]]:
    label_rows = list(label_rows)
    score_rows = list(score_rows)
    label_counts, _ = _key_counts(label_rows, key_fields, row_kind="label")
    score_counts, _ = _key_counts(score_rows, key_fields, row_kind="score")
    if require_unique_label_keys:
        _raise_on_duplicate_keys(label_counts, key_fields, row_kind="label")
    if require_unique_score_keys:
        _raise_on_duplicate_keys(score_counts, key_fields, row_kind="score")

    label_by_key: dict[tuple[Any, ...], int] = {}
    ordered_keys: list[tuple[Any, ...]] = []
    for row_index, row in enumerate(label_rows, start=1):
        key = _row_key(row, key_fields, row_kind="label", row_index=row_index)
        if key not in label_by_key:
            ordered_keys.append(key)
            label_by_key[key] = int(row.get("label", 0))
        else:
            label_by_key[key] = max(label_by_key[key], int(row.get("label", 0)))

    score_by_key: dict[tuple[Any, ...], float] = {}
    for row_index, row in enumerate(score_rows, start=1):
        key = _row_key(row, key_fields, row_kind="score", row_index=row_index)
        score = float(row["score"])
        score_by_key[key] = max(score_by_key.get(key, float("-inf")), score)

    y_true = [label_by_key[key] for key in ordered_keys]
    y_score = [score_by_key.get(key, float("-inf")) for key in ordered_keys]
    return y_true, y_score


def precision_recall_at_k(
    y_true: Sequence[int],
    y_score: Sequence[float],
    k: int,
) -> dict[str, float | int]:
    _validate_same_length(y_true, y_score)
    if k < 0:
        raise ValueError("k must be non-negative")

    cutoff = min(k, len(y_true))
    ranked_indices = sorted(range(len(y_score)), key=lambda index: y_score[index], reverse=True)
    top_indices = ranked_indices[:cutoff]
    hits = sum(1 for index in top_indices if int(y_true[index]) > 0)
    total_positive = sum(1 for value in y_true if int(value) > 0)

    precision = hits / cutoff if cutoff else 0.0
    recall = hits / total_positive if total_positive else 0.0
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hits_at_k": hits,
        "k": cutoff,
    }


def best_f1_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> dict[str, float | None]:
    _validate_same_length(y_true, y_score)
    if not y_true:
        return {"threshold": None, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    best = {"threshold": None, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in sorted(set(float(score) for score in y_score), reverse=True):
        predicted = [score >= threshold for score in y_score]
        true_positive = sum(
            1 for prediction, label in zip(predicted, y_true) if prediction and int(label) > 0
        )
        false_positive = sum(
            1 for prediction, label in zip(predicted, y_true) if prediction and int(label) <= 0
        )
        false_negative = sum(
            1 for prediction, label in zip(predicted, y_true) if not prediction and int(label) > 0
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if f1 > float(best["f1"]):
            best = {
                "threshold": threshold,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
    return best


def evaluate_binary_scores(
    y_true: Sequence[int],
    y_score: Sequence[float],
    k: int | None = None,
) -> dict[str, float | int | None]:
    _validate_same_length(y_true, y_score)
    labels = [1 if int(value) > 0 else 0 for value in y_true]
    scores = _finite_scores(y_score)
    num_positive = sum(labels)
    num_total = len(labels)

    auroc = None
    if len(set(labels)) == 2:
        auroc = _roc_auc_score(labels, scores)
    auprc = _average_precision_score(labels, scores)

    rank_k = k if k is not None else num_total
    top_k = precision_recall_at_k(labels, scores, rank_k)
    f1_result = best_f1_threshold(labels, scores)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision_at_k": top_k["precision_at_k"],
        "recall_at_k": top_k["recall_at_k"],
        "f1": f1_result["f1"],
        "threshold": f1_result["threshold"],
        "num_positive": num_positive,
        "num_total": num_total,
    }


def event_f1(
    pred_events: Iterable[dict[str, Any]],
    true_events: Iterable[dict[str, Any]],
    iou_threshold: float = 0.1,
) -> dict[str, float]:
    predictions = list(pred_events)
    truths = list(true_events)
    candidates: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for true_index, truth in enumerate(truths):
            if _event_identity(prediction) != _event_identity(truth):
                continue
            iou = _frame_iou(prediction, truth)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, true_index))

    matched_predictions: set[int] = set()
    matched_truths: set[int] = set()
    for _, pred_index, true_index in sorted(candidates, reverse=True):
        if pred_index in matched_predictions or true_index in matched_truths:
            continue
        matched_predictions.add(pred_index)
        matched_truths.add(true_index)

    true_positive = len(matched_predictions)
    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(truths) if truths else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
    }


def _row_key(
    row: dict[str, Any],
    key_fields: Sequence[str],
    row_kind: str = "row",
    row_index: int | None = None,
) -> tuple[Any, ...]:
    for field in key_fields:
        if field not in row:
            prefix = f"{row_kind} row"
            if row_index is not None:
                prefix += f" {row_index}"
            raise ValueError(f"{prefix} is missing required key field '{field}'")
    return tuple(row[field] for field in key_fields)


def _key_counts(
    rows: Iterable[dict[str, Any]],
    key_fields: Sequence[str],
    row_kind: str,
) -> tuple[Counter[tuple[Any, ...]], int]:
    counts: Counter[tuple[Any, ...]] = Counter()
    num_rows = 0
    for row_index, row in enumerate(rows, start=1):
        counts[_row_key(row, key_fields, row_kind=row_kind, row_index=row_index)] += 1
        num_rows += 1
    return counts, num_rows


def _num_duplicate_keys(counts: Counter[tuple[Any, ...]]) -> int:
    return sum(1 for count in counts.values() if count > 1)


def _raise_on_duplicate_keys(
    counts: Counter[tuple[Any, ...]],
    key_fields: Sequence[str],
    row_kind: str,
) -> None:
    duplicates = [key for key, count in counts.items() if count > 1]
    if not duplicates:
        return
    preview = ", ".join(repr(key) for key in duplicates[:5])
    if len(duplicates) > 5:
        preview += ", ..."
    raise ValueError(
        f"Duplicate {row_kind} keys for key_fields {tuple(key_fields)}: {preview}"
    )


def _validate_same_length(y_true: Sequence[int], y_score: Sequence[float]) -> None:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")


def _finite_scores(y_score: Sequence[float]) -> list[float]:
    scores = [float(score) for score in y_score]
    finite_values = [score for score in scores if math.isfinite(score)]
    if len(finite_values) == len(scores):
        return scores
    fill_value = min(finite_values) - 1.0 if finite_values else 0.0
    return [score if math.isfinite(score) else fill_value for score in scores]


def _roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        return _manual_roc_auc_score(y_true, y_score)


def _average_precision_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    if sum(y_true) == 0:
        return 0.0
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_score))
    except ImportError:
        return _manual_average_precision_score(y_true, y_score)


def _manual_roc_auc_score(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    positives = [score for label, score in zip(y_true, y_score) if label == 1]
    negatives = [score for label, score in zip(y_true, y_score) if label == 0]
    if not positives or not negatives:
        raise ValueError("ROC AUC is undefined when only one class is present")

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _manual_average_precision_score(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> float:
    ranked = sorted(zip(y_score, y_true), reverse=True)
    total_positive = sum(y_true)
    if total_positive == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_positive


def _event_identity(event: dict[str, Any]) -> tuple[Any, Any]:
    return event.get("sequence"), event.get("track_id")


def _frame_iou(prediction: dict[str, Any], truth: dict[str, Any]) -> float:
    pred_start = int(prediction["frame_start"])
    pred_end = int(prediction["frame_end"])
    true_start = int(truth["frame_start"])
    true_end = int(truth["frame_end"])

    intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start) + 1)
    pred_length = max(0, pred_end - pred_start + 1)
    true_length = max(0, true_end - true_start + 1)
    union = pred_length + true_length - intersection
    return intersection / union if union else 0.0
