from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentResult:
    method_name: str
    task: str
    split: str
    seed: int | None
    score_rows: list[dict[str, Any]]
    label_rows: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def scores_by_sample(self) -> dict[str, dict[str, Any]]:
        return {str(row["sample_id"]): row for row in self.score_rows}

    @property
    def labels_by_sample(self) -> dict[str, list[dict[str, Any]]]:
        labels: dict[str, list[dict[str, Any]]] = {}
        for row in self.label_rows:
            labels.setdefault(str(row["sample_id"]), []).append(row)
        return labels

    def to_report_context(self) -> dict[str, Any]:
        positive_labels = [
            row for row in self.label_rows if int(row.get("label", 0) or 0) == 1
        ]
        return {
            "method_name": self.method_name,
            "task": self.task,
            "split": self.split,
            "seed": self.seed,
            "metrics": dict(self.metrics),
            "labels_by_sample": self.labels_by_sample,
            "summary": {
                "num_scores": len(self.score_rows),
                "num_labels": len(self.label_rows),
                "num_positive_labels": len(positive_labels),
            },
        }


def load_experiment_result(manifest_path: str | Path, method_name: str | None = None) -> ExperimentResult:
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    run = _select_run(manifest, method_name)
    base_dir = manifest_path.parent

    score_file = _resolve_manifest_path(run["score_file"], base_dir)
    metric_file = run.get("metrics_file")
    label_file = manifest.get("label_file")

    score_rows = [_coerce_score_row(row) for row in _load_jsonl(score_file)]
    label_rows = []
    if label_file:
        label_rows = [_coerce_label_row(row) for row in _load_jsonl(_resolve_manifest_path(label_file, base_dir))]
    metrics = _load_json(_resolve_manifest_path(metric_file, base_dir)) if metric_file else {}

    selected_method = str(run.get("name") or metrics.get("method") or method_name or score_file.stem)
    selected_task = str(run.get("task") or metrics.get("task") or "")
    seed = _coerce_optional_int(manifest.get("seed", metrics.get("seed")))
    return ExperimentResult(
        method_name=selected_method,
        task=selected_task,
        split=str(manifest.get("split") or metrics.get("split") or ""),
        seed=seed,
        score_rows=score_rows,
        label_rows=label_rows,
        metrics=metrics,
    )


def write_scores_csv(result: ExperimentResult, output_csv: str | Path) -> dict[str, Any]:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "sequence",
        "track_id",
        "category_id",
        "category_name",
        "score",
        "used_sources",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.score_rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "sequence": row["sequence"],
                    "track_id": row["track_id"],
                    "category_id": row.get("category_id", ""),
                    "category_name": row.get("category_name", ""),
                    "score": row["score"],
                    "used_sources": result.method_name,
                }
            )
    return {
        "output_csv": str(output_csv),
        "num_scores": len(result.score_rows),
        "method": result.method_name,
    }


def _select_run(manifest: dict[str, Any], method_name: str | None) -> dict[str, Any]:
    runs = list(manifest.get("runs") or [])
    if not runs and manifest.get("score_file"):
        runs = [
            {
                "name": manifest.get("method") or Path(str(manifest["score_file"])).stem,
                "task": manifest.get("task", ""),
                "score_file": manifest["score_file"],
                "metrics_file": manifest.get("metrics_file"),
            }
        ]
    if not runs:
        raise ValueError("Experiment manifest does not contain any score runs")
    if method_name is None:
        return dict(runs[0])
    for run in runs:
        if run.get("name") == method_name:
            return dict(run)
    available = ", ".join(str(run.get("name", "")) for run in runs)
    raise ValueError(f"Method {method_name!r} was not found in manifest runs: {available}")


def _resolve_manifest_path(raw_path: str | Path | None, base_dir: Path) -> Path:
    if raw_path is None:
        raise ValueError("Missing required path in experiment manifest")
    normalized = str(raw_path).replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        return path
    base_candidate = base_dir / path
    if base_candidate.exists():
        return base_candidate
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return base_candidate


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _coerce_score_row(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    converted["sample_id"] = str(converted["sample_id"])
    converted["sequence"] = str(converted["sequence"])
    converted["track_id"] = str(converted["track_id"])
    converted["score"] = float(converted.get("score", 0.0) or 0.0)
    return converted


def _coerce_label_row(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    converted["sample_id"] = str(converted["sample_id"])
    converted["sequence"] = str(converted["sequence"])
    converted["track_id"] = str(converted["track_id"])
    for field_name in ("frame_start", "frame_end", "label", "injection_seed"):
        if field_name in converted and converted[field_name] not in (None, ""):
            converted[field_name] = int(converted[field_name])
    return converted


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
