from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from fusiontrack.method_registry import PROFILE_FIELDS, method_profile


METRIC_KEYS = ("auroc", "auprc", "f1", "precision_at_k", "recall_at_k")
REGISTRY_CATEGORY_FIELDS = (set(PROFILE_FIELDS) - {"name", "task"}) | {"aliases", "registry_status"}


@dataclass
class MethodProfile:
    method: str
    task: str
    split: str
    seed: int | None
    metrics: dict[str, Any]
    category: dict[str, Any]
    score_path: Path
    score_rows: list[dict[str, Any]]

    @property
    def scores_by_sample(self) -> dict[str, dict[str, Any]]:
        return {str(row["sample_id"]): row for row in self.score_rows}

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "split": self.split,
            "seed": self.seed,
            "metrics": self.metrics,
            "category": self.category,
            "score_path": self.score_path.name,
        }


@dataclass
class TaskDashboard:
    task: str
    labels: list[dict[str, Any]]
    methods: dict[str, MethodProfile]
    leaderboard: list[dict[str, Any]]
    anomaly_type_rows: list[dict[str, Any]]
    case_rows: dict[str, dict[str, list[dict[str, Any]]]]
    top_k: int
    key_policy: dict[str, Any] = field(default_factory=dict)

    @property
    def labels_by_sample(self) -> dict[str, dict[str, Any]]:
        return {str(row["sample_id"]): row for row in self.labels}

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "top_k": self.top_k,
            "num_labels": len(self.labels),
            "num_positive": sum(1 for row in self.labels if _is_positive(row)),
            "methods": {name: method.to_public_dict() for name, method in self.methods.items()},
            "leaderboard": self.leaderboard,
            "anomaly_type_rows": self.anomaly_type_rows,
            "case_rows": self.case_rows,
            "key_policy": self.key_policy or _task_key_policy(self.task),
        }


@dataclass
class FinalResultsDashboard:
    tasks: dict[str, TaskDashboard] = field(default_factory=dict)
    summary_text: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "summary_text": self.summary_text,
            "tasks": {name: task.to_public_dict() for name, task in self.tasks.items()},
        }


def load_final_results_dashboard(
    final_results_root: str | Path,
    individual_label_file: str | Path,
    group_label_file: str | Path,
    score_search_roots: Iterable[str | Path],
    registration_manifest: str | Path | None = None,
    top_k: int = 100,
    case_limit: int = 12,
) -> FinalResultsDashboard:
    final_results_root = Path(final_results_root)
    score_roots = [Path(root) for root in score_search_roots]
    summary_text = ""
    summary_md = final_results_root / "experiment_summary.md"
    if summary_md.exists():
        summary_text = summary_md.read_text(encoding="utf-8", errors="ignore")

    tasks = {
        "individual": _load_task_dashboard(
            task="individual",
            final_results_root=final_results_root,
            label_file=Path(individual_label_file),
            score_search_roots=score_roots,
            top_k=top_k,
            case_limit=case_limit,
        ),
        "group": _load_task_dashboard(
            task="group",
            final_results_root=final_results_root,
            label_file=Path(group_label_file),
            score_search_roots=score_roots,
            top_k=top_k,
            case_limit=case_limit,
        ),
    }
    if registration_manifest is not None:
        registration_path = Path(registration_manifest)
        if registration_path.exists():
            registration_task = _load_registration_task_dashboard(
                registration_manifest=registration_path,
                score_search_roots=[Path(root) for root in score_search_roots],
                top_k=top_k,
                case_limit=case_limit,
            )
            if registration_task.labels:
                tasks["registration"] = registration_task
            elif registration_task.methods:
                tasks["registration"] = registration_task
    return FinalResultsDashboard(tasks=tasks, summary_text=summary_text)


def _load_task_dashboard(
    task: str,
    final_results_root: Path,
    label_file: Path,
    score_search_roots: list[Path],
    top_k: int,
    case_limit: int,
) -> TaskDashboard:
    labels = [_coerce_label_row(row) for row in _load_jsonl(label_file)]
    summary_rows = _load_summary_rows(final_results_root, task)
    categorized = {
        row["method"]: row
        for row in _load_csv_optional(final_results_root / f"final_{task}_all_methods_categorized.csv")
    }
    methods: dict[str, MethodProfile] = {}
    for row in summary_rows:
        method_name = str(row["method"])
        score_path = resolve_score_path(row.get("source", ""), score_search_roots, task=task, method=method_name)
        score_rows = [_coerce_score_row(item) for item in _load_jsonl(score_path)]
        category = _category_with_registry(
            method_name=method_name,
            task=task,
            category=categorized.get(method_name, {}),
        )
        metrics = {
            key: _coerce_float(row.get(key, category.get(key, 0.0)))
            for key in METRIC_KEYS
        }
        metrics.update(
            {
                "num_score_rows": int(float(row.get("num_score_rows", len(score_rows)) or len(score_rows))),
                "num_missing_score_keys": int(float(row.get("num_missing_score_keys", 0) or 0)),
                "schema_diagnostics": _schema_diagnostics_from_summary_row(task, row, score_rows, labels),
            }
        )
        methods[method_name] = MethodProfile(
            method=method_name,
            task=str(row.get("task", task) or task),
            split=str(row.get("split", "")),
            seed=_coerce_optional_int(row.get("seed")),
            metrics=metrics,
            category={key: value for key, value in category.items() if key not in set(METRIC_KEYS)},
            score_path=score_path,
            score_rows=score_rows,
        )
    leaderboard = _build_leaderboard(methods)
    anomaly_type_rows = _build_anomaly_type_rows(methods, labels, top_k=top_k)
    case_rows = {
        method_name: _build_case_rows(method, labels, top_k=top_k, case_limit=case_limit)
        for method_name, method in methods.items()
    }
    return TaskDashboard(
        task=task,
        labels=labels,
        methods=methods,
        leaderboard=leaderboard,
        anomaly_type_rows=anomaly_type_rows,
        case_rows=case_rows,
        top_k=top_k,
        key_policy=_task_key_policy(task),
    )


def _load_registration_task_dashboard(
    registration_manifest: Path,
    score_search_roots: list[Path],
    top_k: int,
    case_limit: int,
) -> TaskDashboard:
    payload = json.loads(registration_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Registration manifest must be a JSON object.")
    if "runs" not in payload:
        raise ValueError("Registration manifest missing `runs` entries.")

    task_name = str(payload.get("task", "registration"))
    split = str(payload.get("split", "test"))
    seed = payload.get("seed")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Registration manifest runs is empty.")

    methods: dict[str, MethodProfile] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        method_name = str(run.get("name", "unknown")).strip() or "unknown"
        score_file = Path(str(run.get("score_file", "")))
        if not score_file.name:
            continue
        if not score_file.is_absolute():
            score_file = score_file if score_file.exists() else _resolve_registration_path(score_file, score_search_roots, task_name)
        score_rows = _coerce_score_rowed(_load_jsonl(score_file)) if score_file.exists() else []
        metric_path = Path(str(run.get("metrics_file", "")))
        if metric_path.name and not metric_path.is_absolute():
            if not metric_path.exists():
                metric_path = _resolve_registration_path(metric_path, score_search_roots, task_name)
        metrics_payload: dict[str, Any] = {}
        if metric_path.name and metric_path.exists():
            try:
                metrics_payload = json.loads(metric_path.read_text(encoding="utf-8"))
                if not isinstance(metrics_payload, dict):
                    metrics_payload = {}
            except json.JSONDecodeError:
                metrics_payload = {}
        schema_diagnostics = _coerce_mapping(
            metrics_payload.get("schema_diagnostics")
            or run.get("schema_diagnostics")
        )

        methods[method_name] = MethodProfile(
            method=method_name,
            task=task_name,
            split=split,
            seed=_coerce_optional_int(seed),
            metrics={
                "auroc": _coerce_float(metrics_payload.get("auroc", metrics_payload.get("success_rate", 0.0))),
                "auprc": _coerce_float(metrics_payload.get("auprc", 0.0)),
                "f1": _coerce_float(metrics_payload.get("f1", 0.0)),
                "precision_at_k": _coerce_float(metrics_payload.get("precision_at_k", 0.0)),
                "recall_at_k": _coerce_float(metrics_payload.get("recall_at_k", 0.0)),
                "num_score_rows": float(len(score_rows)),
                "num_missing_score_keys": 0.0,
                "num_pairs": _coerce_float(metrics_payload.get("num_pairs", len(score_rows))),
                "num_successful_pairs": _coerce_float(metrics_payload.get("num_successful_pairs", 0.0)),
                "num_failed_pairs": _coerce_float(metrics_payload.get("num_failed_pairs", metrics_payload.get("failures", 0.0))),
                "success_rate": _coerce_float(metrics_payload.get("success_rate", metrics_payload.get("auroc", 0.0))),
                "skip_rate": _coerce_float(metrics_payload.get("skip_rate", 0.0)),
                "rotation_error_deg_mean": _coerce_float(metrics_payload.get("rotation_error_deg_mean", 0.0)),
                "translation_error_mean": _coerce_float(metrics_payload.get("translation_error_mean", 0.0)),
                "chamfer_distance_mean": _coerce_float(metrics_payload.get("chamfer_distance_mean", 0.0)),
                "runtime_sec_mean": _coerce_float(metrics_payload.get("runtime_sec_mean", 0.0)),
                "schema_diagnostics": schema_diagnostics or _registration_schema_diagnostics(score_rows),
            },
            category=_category_with_registry(method_name, task_name, {}),
            score_path=score_file,
            score_rows=score_rows,
        )

    leaderboard = _build_leaderboard(methods)
    anomaly_type_rows = _build_anomaly_type_rows(methods, labels=[], top_k=top_k)
    case_rows = {
        method_name: _build_registration_case_rows(method, case_limit=case_limit)
        for method_name, method in methods.items()
    }
    return TaskDashboard(
        task=task_name,
        labels=[],
        methods=methods,
        leaderboard=leaderboard,
        anomaly_type_rows=anomaly_type_rows,
        case_rows=case_rows,
        top_k=top_k,
        key_policy=_task_key_policy(task_name),
    )


def _resolve_registration_path(path: Path, search_roots: list[Path], task_name: str) -> Path:
    if not path.name:
        raise FileNotFoundError(f"Registration manifest path missing file name: {path}")
    for root in search_roots:
        candidate = root / path
        if candidate.exists():
            return candidate
        if task_name:
            task_candidate = root / task_name / path.name
            if task_candidate.exists():
                return task_candidate
        direct = root / "registration_scores" / path.name
        if direct.exists():
            return direct
    raise FileNotFoundError(f"Could not resolve registration artifact path: {path}")


def _coerce_score_rowed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_coerce_score_row(row) for row in rows]


def resolve_score_path(source: str | Path, score_search_roots: Iterable[Path], task: str, method: str) -> Path:
    raw = str(source or "").replace("\\", "/")
    source_path = Path(raw)
    if source_path.exists():
        return source_path
    candidate_names = _candidate_score_names(source_path.name, method=method, task=task)
    matches: list[Path] = []
    for root in score_search_roots:
        if not root.exists():
            continue
        for name in candidate_names:
            matches.extend(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not resolve score file for {method!r}: {source}")
    return sorted(matches, key=lambda path: _score_path_rank(path, task=task))[0]


def _candidate_score_names(source_name: str, method: str, task: str) -> list[str]:
    names = [source_name] if source_name else []
    if method == "official_lmtad":
        names.append("official_lmtad_scores.jsonl")
    if method == "official_pidpm":
        names.append("official_pidpm_scores.jsonl")
    if method == "official_anomaly_transformer":
        names.append(f"official_anomaly_transformer_{task}_scores.jsonl")
    if method == "official_dcdetector":
        names.append(f"official_dcdetector_{task}_scores.jsonl")
    if method == "official_tranad":
        names.append(f"official_tranad_{task}_scores.jsonl")
    names.append(f"{method}.jsonl")
    seen: set[str] = set()
    unique = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _score_path_rank(path: Path, task: str) -> tuple[int, str]:
    text = path.as_posix().lower()
    rank = 10
    if "fusiontrack_val_results_20260521/fusiontrack_val" in text:
        rank -= 4
    if "fusiontrack_official_runs_tsad_20260521" in text:
        rank -= 3
    if f"/{task}/scores/" in text or f"_{task}_" in text:
        rank -= 2
    if "fusiontrack_smoke" in text or "smoke" in text:
        rank += 5
    return rank, text


def _build_leaderboard(methods: dict[str, MethodProfile]) -> list[dict[str, Any]]:
    rows = []
    for method in methods.values():
        category = method.category
        rows.append(
            {
                "method": method.method,
                "owner": category.get("owner", ""),
                "role": category.get("role", ""),
                "method_family": category.get("method_family", ""),
                "learning_type": category.get("learning_type", ""),
                "is_our_method": category.get("owner") == "our_method",
                **method.metrics,
            }
        )
    return sorted(rows, key=lambda row: float(row.get("auroc", 0.0)), reverse=True)


def _build_anomaly_type_rows(
    methods: dict[str, MethodProfile],
    labels: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    positive_by_type: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        if _is_positive(label):
            positive_by_type.setdefault(str(label.get("anomaly_type", "anomaly")), []).append(label)
    rows = []
    for method in methods.values():
        ranked_ids = [row["sample_id"] for row in _rank_scores(method.score_rows)[:top_k]]
        top_set = set(ranked_ids)
        scores = {row["sample_id"]: float(row.get("score", 0.0)) for row in method.score_rows}
        for anomaly_type, type_labels in sorted(positive_by_type.items()):
            sample_ids = [str(label["sample_id"]) for label in type_labels]
            hits = sum(1 for sample_id in sample_ids if sample_id in top_set)
            values = [scores.get(sample_id, 0.0) for sample_id in sample_ids]
            rows.append(
                {
                    "method": method.method,
                    "anomaly_type": anomaly_type,
                    "hits_at_k": hits,
                    "total_positive": len(sample_ids),
                    "recall_at_k": hits / len(sample_ids) if sample_ids else 0.0,
                    "mean_positive_score": sum(values) / len(values) if values else 0.0,
                }
            )
    return rows


def _build_case_rows(
    method: MethodProfile,
    labels: list[dict[str, Any]],
    top_k: int,
    case_limit: int,
) -> dict[str, list[dict[str, Any]]]:
    labels_by_sample = {str(label["sample_id"]): label for label in labels}
    positive_ids = {sample_id for sample_id, label in labels_by_sample.items() if _is_positive(label)}
    ranked = _rank_scores(method.score_rows)
    top_ids = {row["sample_id"] for row in ranked[:top_k]}
    true_positive = []
    false_positive = []
    for rank, row in enumerate(ranked, start=1):
        label = labels_by_sample.get(row["sample_id"], {})
        if row["sample_id"] in positive_ids and len(true_positive) < case_limit:
            true_positive.append(_case_row(row, label, "true_positive", rank))
        elif row["sample_id"] not in positive_ids and len(false_positive) < case_limit:
            false_positive.append(_case_row(row, label, "false_positive", rank))
        if len(true_positive) >= case_limit and len(false_positive) >= case_limit:
            break
    scores_by_sample = {row["sample_id"]: row for row in method.score_rows}
    false_negative = []
    for sample_id in sorted(
        positive_ids - top_ids,
        key=lambda value: float(scores_by_sample.get(value, {}).get("score", 0.0)),
    ):
        row = scores_by_sample.get(sample_id)
        if row is None:
            continue
        label = labels_by_sample.get(sample_id, {})
        false_negative.append(_case_row(row, label, "false_negative", _rank_of(ranked, sample_id)))
        if len(false_negative) >= case_limit:
            break
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _build_registration_case_rows(method: MethodProfile, case_limit: int) -> dict[str, list[dict[str, Any]]]:
    ranked_high = _rank_scores(method.score_rows)
    ranked_low = sorted(method.score_rows, key=lambda row: float(row.get("score", 0.0) or 0.0))
    failed = [
        row for row in ranked_high
        if row.get("success") in (False, "false", "False", 0, "0") or row.get("skipped") in (True, "true", "True", 1, "1")
    ]
    successful = [
        row for row in ranked_low
        if row.get("success") in (True, "true", "True", 1, "1") and row.get("skipped") not in (True, "true", "True", 1, "1")
    ]
    return {
        "true_positive": [
            _registration_case_row(row, "success", _rank_of(ranked_high, str(row.get("sample_id", ""))))
            for row in successful[:case_limit]
        ],
        "false_positive": [
            _registration_case_row(row, "high_error", rank)
            for rank, row in enumerate(ranked_high[:case_limit], start=1)
        ],
        "false_negative": [
            _registration_case_row(row, "failed_or_skipped", _rank_of(ranked_high, str(row.get("sample_id", ""))))
            for row in failed[:case_limit]
        ],
    }


def _registration_case_row(row: dict[str, Any], case_type: str, rank: int) -> dict[str, Any]:
    return {
        "case_type": case_type,
        "sample_id": str(row.get("sample_id", "")),
        "sequence": str(row.get("sequence", "")),
        "track_id": str(row.get("track_id", "")),
        "score": float(row.get("score", 0.0) or 0.0),
        "rank": rank,
        "label": 0,
        "anomaly_type": case_type,
        "frame_start": 0,
        "frame_end": 0,
    }


def _case_row(row: dict[str, Any], label: dict[str, Any], case_type: str, rank: int) -> dict[str, Any]:
    return {
        "case_type": case_type,
        "sample_id": str(row.get("sample_id", "")),
        "sequence": str(row.get("sequence", label.get("sequence", ""))),
        "track_id": str(row.get("track_id", label.get("track_id", ""))),
        "score": float(row.get("score", 0.0) or 0.0),
        "rank": rank,
        "label": int(label.get("label", 0) or 0),
        "anomaly_type": str(label.get("anomaly_type", "normal")),
        "frame_start": int(label.get("frame_start", 0) or 0),
        "frame_end": int(label.get("frame_end", 0) or 0),
    }


def _rank_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)


def _rank_of(ranked: list[dict[str, Any]], sample_id: str) -> int:
    for index, row in enumerate(ranked, start=1):
        if row["sample_id"] == sample_id:
            return index
    return 0


def _is_positive(row: dict[str, Any]) -> bool:
    return int(row.get("label", 0) or 0) == 1


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_csv_optional(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load_csv(path)


def _load_summary_rows(final_results_root: Path, task: str) -> list[dict[str, Any]]:
    json_path = final_results_root / f"final_{task}_all_methods_summary.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{json_path} must contain a JSON array")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(payload, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{json_path}:{index} is not a JSON object")
            rows.append(dict(row))
        return rows
    return _load_csv(final_results_root / f"final_{task}_all_methods_summary.csv")


def _schema_diagnostics_from_summary_row(
    task: str,
    row: dict[str, Any],
    score_rows: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostics = _coerce_mapping(row.get("schema_diagnostics"))
    if diagnostics:
        return diagnostics
    missing = _coerce_int(row.get("num_missing_score_keys"), 0)
    extra = _coerce_int(row.get("num_extra_score_keys"), 0)
    duplicate_labels = _coerce_int(row.get("num_duplicate_label_keys"), 0)
    duplicate_scores = _coerce_int(row.get("num_duplicate_score_keys"), 0)
    warnings = []
    if missing:
        warnings.append("missing_score_keys")
    if extra:
        warnings.append("extra_score_keys")
    if duplicate_labels:
        warnings.append("duplicate_label_keys")
    if duplicate_scores:
        warnings.append("duplicate_score_keys")
    key_policy = _task_key_policy(task)
    return {
        "schema_diagnostics_version": 1,
        "status": "warning" if warnings else "ok",
        "key_fields": key_policy["key_fields"],
        "fallback_key_fields": key_policy.get("fallback_key_fields", []),
        "label": {
            "num_rows": _coerce_int(row.get("num_label_rows"), len(labels)),
            "num_unique_keys": _coerce_int(row.get("num_unique_label_keys"), len(labels)),
            "num_duplicate_keys": duplicate_labels,
        },
        "score": {
            "num_rows": _coerce_int(row.get("num_score_rows"), len(score_rows)),
            "num_unique_keys": _coerce_int(row.get("num_unique_score_keys"), len(score_rows)),
            "num_duplicate_keys": duplicate_scores,
        },
        "alignment": {
            "num_missing_score_keys": missing,
            "num_extra_score_keys": extra,
        },
        "warnings": warnings,
    }


def _task_key_policy(task: str) -> dict[str, Any]:
    task_name = str(task or "").lower()
    if task_name == "group":
        return {
            "task": "group",
            "key_fields": ["sample_id", "window_id"],
            "fallback_key_fields": ["sample_id"],
            "scope": "group_window",
            "strict": True,
            "description": "Group strict protocol aligns labels and scores by sample_id + window_id; sample_id is kept as a legacy fallback for older score rows.",
        }
    if task_name == "registration":
        return {
            "task": "registration",
            "key_fields": ["sample_id"],
            "fallback_key_fields": [],
            "scope": "registration_pair",
            "strict": True,
            "description": "Registration diagnostics align each score row by point-cloud pair sample_id.",
        }
    return {
        "task": "individual",
        "key_fields": ["sample_id"],
        "fallback_key_fields": [],
        "scope": "individual_track",
        "strict": True,
        "description": "Individual strict protocol aligns each trajectory label and score row by sample_id.",
    }


def _registration_schema_diagnostics(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_diagnostics_version": 1,
        "status": "ok",
        "key_fields": ["sample_id"],
        "label": {
            "num_rows": 0,
            "num_unique_keys": 0,
            "num_duplicate_keys": 0,
        },
        "score": {
            "num_rows": len(score_rows),
            "num_unique_keys": len({str(row.get("sample_id", "")) for row in score_rows}),
            "num_duplicate_keys": 0,
        },
        "alignment": {
            "num_missing_score_keys": 0,
            "num_extra_score_keys": 0,
        },
        "warnings": [],
    }


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    raw = value.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _category_with_registry(
    method_name: str,
    task: str,
    category: dict[str, Any],
) -> dict[str, Any]:
    registry_profile = method_profile(method_name, task)
    merged = {
        key: value
        for key, value in registry_profile.items()
        if key not in {"name", "task"} and value not in (None, "")
    }
    for key, value in category.items():
        if key not in REGISTRY_CATEGORY_FIELDS and value not in (None, ""):
            merged[key] = value
    return merged


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
    converted["sequence"] = str(converted.get("sequence", ""))
    converted["track_id"] = str(converted.get("track_id", ""))
    converted["score"] = float(converted.get("score", 0.0) or 0.0)
    return converted


def _coerce_label_row(row: dict[str, Any]) -> dict[str, Any]:
    converted = dict(row)
    converted["sample_id"] = str(converted["sample_id"])
    converted["sequence"] = str(converted.get("sequence", ""))
    converted["track_id"] = str(converted.get("track_id", ""))
    for field_name in ("frame_start", "frame_end", "label", "injection_seed"):
        if field_name in converted and converted[field_name] not in (None, ""):
            converted[field_name] = int(converted[field_name])
    return converted


def _coerce_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))
