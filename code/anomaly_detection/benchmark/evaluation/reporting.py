from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from evaluation.io import load_label_rows, load_score_rows
from evaluation.metrics import alignment_report, align_scores_with_labels, evaluate_binary_scores
from evaluation.schema import validate_label_rows, validate_score_rows


def evaluate_score_file(
    score_path: Path,
    label_path: Path,
    output_json: Path | None = None,
    key_fields: Sequence[str] = ("sample_id",),
    k: int | None = None,
    require_unique_keys: bool = False,
    require_score_key_match: bool = False,
) -> dict[str, Any]:
    label_rows = validate_label_rows(
        load_label_rows(label_path),
        key_fields=key_fields,
        require_unique_keys=require_unique_keys,
    )
    score_rows = validate_score_rows(
        load_score_rows(score_path),
        key_fields=key_fields,
        require_unique_keys=require_unique_keys,
    )
    report = alignment_report(score_rows, label_rows, key_fields=key_fields)
    if require_score_key_match:
        _raise_on_score_key_mismatch(report)
    y_true, y_score = align_scores_with_labels(
        score_rows,
        label_rows,
        key_fields=key_fields,
        require_unique_label_keys=require_unique_keys,
        require_unique_score_keys=require_unique_keys,
    )
    metrics = evaluate_binary_scores(y_true, y_score, k=k)
    metrics.update(report)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return metrics


def summarize_metric_files(
    metric_files: Iterable[Path],
    output_csv: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_file in metric_files:
        row = json.loads(metric_file.read_text(encoding="utf-8"))
        if not isinstance(row, dict):
            raise ValueError(f"{metric_file} does not contain a JSON object")
        rows.append(row)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = _summary_fieldnames(rows)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def _summary_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    priority = ["method", "source", "split", "seed"]
    seen = set()
    fieldnames: list[str] = []
    for field in priority:
        if any(field in row for row in rows):
            fieldnames.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)
    return fieldnames


def _raise_on_score_key_mismatch(report: dict[str, int]) -> None:
    missing = int(report["num_missing_score_keys"])
    extra = int(report["num_extra_score_keys"])
    if missing == 0 and extra == 0:
        return
    raise ValueError(
        "Score keys do not exactly match label keys: "
        f"missing_score_keys={missing}, extra_score_keys={extra}"
    )
