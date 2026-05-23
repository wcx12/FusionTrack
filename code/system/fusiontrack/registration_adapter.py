from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _method_score(row: dict[str, Any]) -> float:
    chamfer = _coerce_float(row.get("chamfer_distance", 0.0), 0.0)
    rotation = _coerce_float(row.get("rotation_error_deg", 0.0), 0.0)
    translation = _coerce_float(row.get("translation_error", 0.0), 0.0)
    runtime = _coerce_float(row.get("runtime_sec", 0.0), 0.0)
    success = bool(row.get("success", False))

    # We model anomaly score as registration inconsistency:
    # - larger pose/Chamfer error => higher score (more abnormal)
    # - failed registration is treated as highly anomalous
    pose_error = chamfer + 0.02 * rotation + 2.0 * translation
    return max(0.0, pose_error + 0.5 * runtime + (1.0 if not success else 0.0))


def _registration_component_scores(row: dict[str, Any], score: float) -> dict[str, float]:
    rotation = _coerce_float(row.get("rotation_error_deg", 0.0), 0.0)
    translation = _coerce_float(row.get("translation_error", 0.0), 0.0)
    chamfer = _coerce_float(row.get("chamfer_distance", 0.0), 0.0)
    runtime = _coerce_float(row.get("runtime_sec", 0.0), 0.0)
    return {
        "registration_rotation_error": rotation,
        "registration_translation_error": translation,
        "registration_chamfer_distance": chamfer,
        "registration_runtime_sec": runtime,
        "registration_error_score": score,
    }


def _point_layout(seed_text: str) -> list[tuple[int, float, float]]:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    base_x = int(digest[0]) / 255.0 * 40.0 + 20.0
    base_y = int(digest[1]) / 255.0 * 40.0 + 20.0
    drift_x = (int(digest[2]) / 255.0 - 0.5) * 4.0
    drift_y = (int(digest[3]) / 255.0 - 0.5) * 4.0
    return [
        (0, float(base_x), float(base_y)),
        (8, float(base_x + drift_x), float(base_y + drift_y)),
        (16, float(base_x + 2 * drift_x), float(base_y + 2 * drift_y)),
    ]


def _registration_point_preview(seed_text: str, row: dict[str, Any]) -> dict[str, list[list[float]]]:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    phase = int(digest[4]) / 255.0 * math.pi
    rotation = min(_coerce_float(row.get("rotation_error_deg", 0.0), 0.0) / 45.0, 1.0)
    translation = min(_coerce_float(row.get("translation_error", 0.0), 0.0), 1.0)
    chamfer = min(_coerce_float(row.get("chamfer_distance", 0.0), 0.0) * 4.0, 1.0)
    source: list[list[float]] = []
    reference: list[list[float]] = []
    aligned: list[list[float]] = []
    for index in range(18):
        angle = phase + index * (math.pi * 2.0 / 18.0)
        radius = 0.7 + 0.22 * math.sin(index * 1.7 + phase)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0.25 * math.sin(angle * 2.0)
        source.append([round(x, 4), round(y, 4), round(z, 4)])
        reference.append([round(x + 0.18, 4), round(y - 0.14, 4), round(z + 0.08, 4)])
        residual_x = 0.18 * rotation + 0.12 * translation
        residual_y = -0.14 * rotation + 0.08 * chamfer
        residual_z = 0.08 * rotation
        aligned.append([round(x + 0.18 - residual_x, 4), round(y - 0.14 - residual_y, 4), round(z + 0.08 - residual_z, 4)])
    return {"source": source, "reference": reference, "aligned": aligned}


def _iter_rows_by_method(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("pair_results", []):
        if not isinstance(row, dict):
            continue
        method = str(row.get("method", "unknown")).strip() or "unknown"
        by_method[method].append(row)
    return by_method


def _build_sample_key(method: str, row: dict[str, Any]) -> str:
    return f"{method}:{int(row.get('batch_idx', 0))}:{int(row.get('sample_idx', 0))}:{int(row.get('group_ref_idx', 0))}"


def _build_fused_records(sample_row: dict[str, Any], method: str, score: float) -> dict[str, Any]:
    sample_id = sample_row["sample_id"]
    trajectory = sample_row["trajectory"]

    points: list[dict[str, Any]] = []
    for point in trajectory.get("points", []):
        if not isinstance(point, dict):
            continue
        frame_id = point.get("frame_id")
        fused = point.get("fused")
        if frame_id is None or not isinstance(fused, dict):
            continue
        points.append(
            {
                "frame_id": int(frame_id),
                "fused": {
                    "center_xy": [
                        float(fused.get("center_xy", [0.0, 0.0])[0]),
                        float(fused.get("center_xy", [0.0, 0.0])[1]),
                    ],
                    "confidence": float(fused.get("confidence", score)),
                },
            }
        )

    return {
        "sample_id": sample_id,
        "sequence": trajectory["sequence"],
        "track_id": trajectory["track_id"],
        "category_id": 0,
        "category_name": "registration-pair",
        "source": method,
        "points": points,
    }


def _build_metric_row(method: str, payload: dict[str, Any], method_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "task": "registration",
        "split": "test",
        "seed": payload.get("args", {}).get("seed", None),
        "auroc": _coerce_float(method_summary.get("auroc", 0.0), 0.0),
        "auprc": _coerce_float(method_summary.get("auprc", 0.0), 0.0),
        "f1": _coerce_float(method_summary.get("f1", 0.0), 0.0),
        "precision_at_k": _coerce_float(method_summary.get("precision_at_k", 0.0), 0.0),
        "recall_at_k": _coerce_float(method_summary.get("recall_at_k", 0.0), 0.0),
        "num_pairs": int(method_summary.get("num_pairs", 0) or 0),
        "success_rate": _coerce_float(method_summary.get("success_rate", 0.0), 0.0),
        "runtime_sec_mean": _coerce_float(method_summary.get("runtime_sec_mean", 0.0), 0.0),
        "chamfer_distance_mean": _coerce_float(method_summary.get("chamfer_distance_mean", 0.0), 0.0),
    }


def build_registration_experiment_bundle(
    summary_path: str | Path,
    work_root: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing registration benchmark summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Registration summary must be a JSON object")

    benchmark = payload.get("benchmark", {})
    if not isinstance(benchmark, dict) or not benchmark:
        raise ValueError("Registration summary is missing per-method benchmark metrics")

    by_method = _iter_rows_by_method(payload)
    if not by_method:
        raise ValueError("Registration summary has no pair_results records")

    work_root = Path(work_root)
    score_dir = work_root / "registration_scores"
    metrics_dir = work_root / "registration_metrics"
    artifact_dir = work_root / "registration_artifacts"
    score_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_runs = []
    fused_records: list[dict[str, Any]] = []

    for method in sorted(by_method):
        rows = by_method[method]
        summary = benchmark.get(method)
        if not isinstance(summary, dict):
            raise ValueError(f"Missing benchmark summary metrics for method {method}")

        method_id = _safe_name(method)
        score_path = score_dir / f"{method_id}_registration_scores.jsonl"
        metric_path = metrics_dir / f"{method_id}_registration_metrics.json"

        with score_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                score = _method_score(row)
                sequence = str(row.get("sequence", f"batch_{int(row.get('batch_idx', 0)):04d}"))
                track_id = str(row.get("group_ref_idx", index))
                sample_id = _build_sample_key(method, row)
                sample_row = {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "track_id": track_id,
                    "category_id": 0,
                    "category_name": "registration",
                    "score": score,
                    "used_sources": "registration",
                    "source": method,
                    "rotation_error_deg": row.get("rotation_error_deg"),
                    "translation_error": row.get("translation_error"),
                    "chamfer_distance": row.get("chamfer_distance"),
                    "runtime_sec": row.get("runtime_sec"),
                    "success": bool(row.get("success", False)),
                    "skipped": bool(row.get("skipped", False)),
                    "component_scores": _registration_component_scores(row, score),
                    "registration_points": _registration_point_preview(sample_id, row),
                    "metadata": {
                        "method": method,
                        "group_ref_idx": int(row.get("group_ref_idx", 0)),
                        "batch_idx": int(row.get("batch_idx", 0)),
                        "sample_idx": int(row.get("sample_idx", 0)),
                        "error": str(row.get("error", "")),
                    },
                }
                row["sample_id"] = sample_id
                trajectory_seed = f"{method}:{sample_id}"
                traj_points = _point_layout(trajectory_seed)
                trajectory = {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "track_id": track_id,
                    "points": [
                        {
                            "frame_id": frame,
                            "fused": {
                                "center_xy": [x, y],
                                "confidence": round(score, 6),
                            },
                        }
                        for frame, x, y in traj_points
                    ],
                }
                trajectory_row = _build_fused_records(
                    {"sample_id": sample_id, "trajectory": trajectory}, method, score
                )
                trajectory_row["method"] = method
                fused_records.append(trajectory_row)
                handle.write(json.dumps(sample_row, ensure_ascii=False) + "\n")

        metric_path.write_text(
            json.dumps(_build_metric_row(method, payload, summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_runs.append(
            {
                "name": method,
                "task": "registration",
                "score_file": str((score_path).relative_to(work_root)),
                "metrics_file": str(metric_path.relative_to(work_root)),
            }
        )

    manifest = {
        "split": split,
        "task": "registration",
        "seed": payload.get("args", {}).get("seed", None),
        "manifest_source": str(summary_path),
        "runs": manifest_runs,
    }
    manifest_path = artifact_dir / "registration_experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    fused_path = artifact_dir / "registration_fused_trajectories.jsonl"
    with fused_path.open("w", encoding="utf-8") as handle:
        for row in fused_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "manifest_path": str(manifest_path),
        "fused_jsonl": str(fused_path),
        "score_files": [str(run["score_file"]) for run in manifest_runs],
        "num_methods": len(manifest_runs),
        "num_scores": len(fused_records),
    }
