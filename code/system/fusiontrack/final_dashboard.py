from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from fusiontrack.event_segments import event_segments_from_frame_scores, normalize_frame_event_scores
from fusiontrack.final_results import FinalResultsDashboard
from fusiontrack.visualization import (
    _copy_background_asset,
    _fallback_scene_size,
    _safe_name,
    _trajectory_frame_points,
)


def build_final_dashboard(
    dashboard: FinalResultsDashboard,
    output_dir: str | Path,
    fused_jsonl: str | Path | None = None,
    data_root: str | Path | None = None,
    top_sequences: int = 5,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    dashboard_data = _json_safe(dashboard.to_public_dict())
    if provenance is not None:
        dashboard_data["provenance"] = _json_safe(_build_public_provenance(provenance))
    playback_payloads = {}
    if fused_jsonl is not None:
        playback_payloads = _json_safe(_build_playback_payloads(
            dashboard=dashboard,
            fused_jsonl=Path(fused_jsonl),
            data_root=Path(data_root) if data_root is not None else Path("data") / "VT-Tiny-MOT",
            assets_dir=assets_dir,
            top_sequences=top_sequences,
        ))
    (assets_dir / "final_dashboard_data.json").write_text(
        json.dumps(dashboard_data, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (assets_dir / "final_playback_data.json").write_text(
        json.dumps(playback_payloads, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    html_text = _render_html(dashboard_data, playback_payloads)
    report_html = output_dir / "index.html"
    report_html.write_text(html_text, encoding="utf-8")
    return {
        "report_html": str(report_html),
        "assets_dir": str(assets_dir),
        "num_tasks": len(dashboard.tasks),
        "num_methods": sum(len(task.methods) for task in dashboard.tasks.values()),
        "playback_sequences": list(playback_payloads),
    }


def _build_public_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    dataset_manifest = provenance.get("dataset_manifest")
    if not isinstance(dataset_manifest, dict):
        dataset_manifest = {}
    splits = dataset_manifest.get("splits")
    split_names = sorted(str(name) for name in splits) if isinstance(splits, dict) else []
    score_roots = provenance.get("score_search_roots")
    if not isinstance(score_roots, list):
        score_roots = []
    parameters = {
        "top_sequences": provenance.get("top_sequences"),
        "top_k": provenance.get("top_k"),
        "case_limit": provenance.get("case_limit"),
    }
    return {
        "mode": str(provenance.get("mode") or "final_results_dashboard"),
        "generated_at_utc": provenance.get("generated_at_utc"),
        "dataset": {
            "name": dataset_manifest.get("dataset_name"),
            "status": dataset_manifest.get("status"),
            "fingerprint": dataset_manifest.get("dataset_fingerprint"),
            "splits": split_names,
            "manifest": _public_path_hint(provenance.get("dataset_manifest_path")),
        },
        "inputs": {
            "final_results_root": _public_path_hint(provenance.get("final_results_root")),
            "individual_label_file": _public_path_hint(provenance.get("individual_label_file")),
            "group_label_file": _public_path_hint(provenance.get("group_label_file")),
            "score_search_root_count": len(score_roots),
            "score_search_roots": [_public_path_hint(path) for path in score_roots],
            "fused_jsonl": _public_path_hint(provenance.get("fused_jsonl")),
            "registration_manifest": _public_path_hint(provenance.get("registration_manifest")),
            "registration_fused_jsonl": _public_path_hint(provenance.get("registration_fused_jsonl")),
        },
        "parameters": {key: value for key, value in parameters.items() if value is not None},
    }


def _public_path_hint(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value)
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.name
    return raw.replace("\\", "/")


def _build_playback_payloads(
    dashboard: FinalResultsDashboard,
    fused_jsonl: Path,
    data_root: Path,
    assets_dir: Path,
    top_sequences: int,
) -> dict[str, Any]:
    if not dashboard.tasks:
        return {}

    selected_sequences: list[str] = []
    for task in dashboard.tasks.values():
        for sequence in _selected_sequences_for_task(task, top_sequences=top_sequences):
            if sequence and sequence not in selected_sequences:
                selected_sequences.append(sequence)
    if not selected_sequences:
        for task in dashboard.tasks.values():
            if task.labels:
                selected_sequences = [str(task.labels[0].get("sequence", ""))]
                break
    priority_samples_by_sequence: dict[str, set[str]] = defaultdict(set)
    for task in dashboard.tasks.values():
        for sample_id, sequence in _selected_case_samples_for_task(task):
            if sample_id and sequence:
                priority_samples_by_sequence[sequence].add(sample_id)
    selected_set = set(selected_sequences)
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with fused_jsonl.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            trajectory = json.loads(stripped)
            sequence = str(trajectory.get("sequence", ""))
            if sequence in selected_set:
                by_sequence[sequence].append(trajectory)
    labels_by_task_sample = {
        task_name: _aggregate_labels_by_sample(task.labels)
        for task_name, task in dashboard.tasks.items()
    }
    labels_by_task_sample_rows = {
        task_name: _group_labels_by_sample(task.labels)
        for task_name, task in dashboard.tasks.items()
    }
    labels_by_task_sequence: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task_name, task in dashboard.tasks.items():
        labels_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for label in task.labels:
            labels_by_sequence[str(label.get("sequence", ""))].append(label)
        labels_by_task_sequence[task_name] = labels_by_sequence
    scores_by_task_method = {
        task_name: {
            method_name: _aggregate_scores_by_sample(method.score_rows)
            for method_name, method in task.methods.items()
        }
        for task_name, task in dashboard.tasks.items()
    }
    payloads = {}
    for sequence in selected_sequences:
        trajectories = by_sequence.get(sequence, [])
        background_asset, background_size, background_frames = _copy_background_asset(
            sequence,
            trajectories,
            data_root,
            assets_dir,
        )
        modality_audit = _sequence_modality_audit(
            sequence=sequence,
            trajectories=trajectories,
            background_asset=background_asset,
            background_frames=background_frames,
        )
        tracks = []
        frame_ids: list[int] = []
        priority_samples = priority_samples_by_sequence.get(sequence, set())
        for trajectory in _prioritized_trajectories(trajectories, priority_samples)[:160]:
            frame_points = _trajectory_frame_points(trajectory)
            if not frame_points:
                continue
            frame_ids.extend(point[0] for point in frame_points)
            sample_id = str(trajectory["sample_id"])
            points = [
                {"frame": frame, "x": round(x, 3), "y": round(y, 3)}
                for frame, x, y in frame_points
            ]
            task_scores = {
                task_name: {
                    method_name: round(float(rows.get(sample_id, {}).get("score", 0.0) or 0.0), 6)
                    for method_name, rows in method_scores.items()
                }
                for task_name, method_scores in scores_by_task_method.items()
            }
            task_score_rows = {
                task_name: {
                    method_name: rows.get(sample_id, {})
                    for method_name, rows in method_scores.items()
                }
                for task_name, method_scores in scores_by_task_method.items()
            }
            task_labels = {
                task_name: _track_label_payload(
                    labels.get(sample_id, {}),
                    default_start=frame_points[0][0],
                    default_end=frame_points[-1][0],
                )
                for task_name, labels in labels_by_task_sample.items()
            }
            task_segments = {
                task_name: sorted(
                    labels_by_task_sample_rows.get(task_name, {}).get(sample_id, []),
                    key=lambda row: int(row.get("frame_start", 0) or 0),
                )
                for task_name in labels_by_task_sample
            }
            task_score_components = {
                task_name: {
                    method_name: _score_component_payload(method_rows)
                    for method_name, method_rows in method_rows_by_method.items()
                }
                for task_name, method_rows_by_method in task_score_rows.items()
            }
            task_score_decomp = {
                task_name: {
                    method_name: _score_decomposition(method_rows)
                    for method_name, method_rows in method_rows_by_method.items()
                }
                for task_name, method_rows_by_method in task_score_rows.items()
            }
            individual_scores = task_scores.get("individual") or next(iter(task_scores.values()), {})
            individual_label = task_labels.get("individual") or next(iter(task_labels.values()), {})
            tracks.append(
                {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "track_id": str(trajectory.get("track_id", "")),
                    "category": trajectory.get("category_name", "") or "",
                    "method_scores": individual_scores,
                    "task_scores": task_scores,
                    "task_score_rows": task_score_rows,
                    "task_score_components": task_score_components,
                    "task_score_decomposition": task_score_decomp,
                    "task_labels": task_labels,
                    "task_segments": task_segments,
                    "label": int(individual_label.get("label", 0) or 0),
                    "anomaly_type": str(individual_label.get("anomaly_type", "normal")),
                    "frame_start": int(individual_label.get("frame_start", frame_points[0][0]) or frame_points[0][0]),
                    "frame_end": int(individual_label.get("frame_end", frame_points[-1][0]) or frame_points[-1][0]),
                    "points": points,
                }
            )
        width, height = background_size or _fallback_scene_size(trajectories)
        frame_start = min(frame_ids) if frame_ids else 0
        frame_end = max(frame_ids) if frame_ids else 0
        stats_by_task = {}
        for task_name, labels_by_sequence in labels_by_task_sequence.items():
            sequence_labels = labels_by_sequence.get(sequence, [])
            scored_tracks = [
                track
                for track in tracks
                if any(
                    float(value or 0.0) != 0.0
                    for value in (track.get("task_scores", {}).get(task_name, {}) or {}).values()
                )
            ]
            has_labels = bool(sequence_labels)
            stats_by_task[task_name] = {
                "sequence_sample_count": len(sequence_labels) if has_labels else len(scored_tracks),
                "sequence_anomaly_count": (
                    sum(1 for row in sequence_labels if int(row.get("label", 0) or 0) == 1)
                    if has_labels
                    else sum(1 for track in scored_tracks if float(next(iter((track.get("task_scores", {}).get(task_name, {}) or {"": 0}).values()), 0.0) or 0.0) > 0.0)
                ),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "visualized_tracks": len(tracks),
            }
        default_stats = (
            stats_by_task.get("individual")
            or next(iter(stats_by_task.values()), {})
            or {
                "sequence_sample_count": len(tracks),
                "sequence_anomaly_count": 0,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "visualized_tracks": len(tracks),
            }
        )
        media = _sequence_media_payload(
            stats_by_task=stats_by_task,
            background_asset=background_asset,
            background_frames=background_frames,
        )
        payloads[sequence] = {
            "sequence": sequence,
            "background": f"assets/{background_asset.name}" if background_asset else None,
            "background_frames": [
                {
                    "frame": int(item["frame"]),
                    "src": f"assets/{item['path'].name}",
                    **(
                        {"fallback_src": f"assets/{background_asset.name}"}
                        if background_asset and item["path"].name != background_asset.name
                        else {}
                    ),
                }
                for item in background_frames
            ],
            "size": {"width": width, "height": height},
            "frame_range": [frame_start, frame_end],
            "stats": default_stats,
            "stats_by_task": stats_by_task,
            "modality_audit": modality_audit,
            "media": media,
            "tracks": tracks,
        }
    return payloads


def _coverage(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 6)


def _sequence_modality_audit(
    sequence: str,
    trajectories: list[dict[str, Any]],
    background_asset: Path | None,
    background_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    point_count = 0
    fused_points = 0
    rgb_points = 0
    thermal_points = 0
    modal_pair_count = 0
    modal_offset_sum = 0.0
    modal_offset_max = 0.0
    for trajectory in trajectories:
        for point in trajectory.get("points", []):
            point_count += 1
            fused = point.get("fused") or {}
            if fused.get("center_xy"):
                fused_points += 1
            rgb = point.get("rgb") or {}
            thermal = point.get("thermal") or {}
            if rgb.get("file"):
                rgb_points += 1
            if thermal.get("file"):
                thermal_points += 1
            modal = point.get("modal") or {}
            if modal.get("offset_distance") is not None:
                offset = float(modal.get("offset_distance") or 0.0)
                modal_pair_count += 1
                modal_offset_sum += offset
                modal_offset_max = max(modal_offset_max, offset)
    missing_rgb = max(point_count - rgb_points, 0)
    missing_thermal = max(point_count - thermal_points, 0)
    if point_count == 0:
        status = "no_tracks"
    elif not background_asset:
        status = "missing_background"
    elif missing_rgb or missing_thermal:
        status = "partial_modality"
    else:
        status = "ok"
    return {
        "sequence": sequence,
        "trajectory_count": len(trajectories),
        "point_count": point_count,
        "fused_point_count": fused_points,
        "rgb_point_count": rgb_points,
        "thermal_point_count": thermal_points,
        "missing_rgb_points": missing_rgb,
        "missing_thermal_points": missing_thermal,
        "fused_coverage": _coverage(fused_points, point_count),
        "rgb_coverage": _coverage(rgb_points, point_count),
        "thermal_coverage": _coverage(thermal_points, point_count),
        "background_frame_count": len(background_frames),
        "background_status": "available" if background_asset else "missing",
        "modal_pair_count": modal_pair_count,
        "modal_offset_mean": round(modal_offset_sum / modal_pair_count, 6) if modal_pair_count else 0.0,
        "modal_offset_max": round(modal_offset_max, 6),
        "status": status,
    }


def _sequence_media_payload(
    stats_by_task: dict[str, dict[str, Any]],
    background_asset: Path | None,
    background_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    has_original_background = bool(background_asset or background_frames)
    registration_count = int(float((stats_by_task.get("registration") or {}).get("sequence_sample_count", 0) or 0))
    non_registration_count = sum(
        int(float((stats or {}).get("sequence_sample_count", 0) or 0))
        for task_name, stats in stats_by_task.items()
        if task_name != "registration"
    )
    if has_original_background:
        kind = "original_video_background"
        label_key = "mediaKindVideo"
        explanation_key = "backgroundLoading"
    elif registration_count > 0 and non_registration_count == 0:
        kind = "registration_point_cloud"
        label_key = "mediaKindRegistration"
        explanation_key = "registrationNoVideoBackground"
    else:
        kind = "track_only_missing_background"
        label_key = "mediaKindTrackOnly"
        explanation_key = "sequenceNoVideoBackground"
    return {
        "kind": kind,
        "label_key": label_key,
        "explanation_key": explanation_key,
        "has_original_background": has_original_background,
        "background_frame_count": len(background_frames),
        "registration_sample_count": registration_count,
    }


def _selected_sequences_for_task(task: Any, top_sequences: int) -> list[str]:
    sequences: list[str] = []
    for _, sequence in _selected_case_samples_for_task(task):
        if sequence and sequence not in sequences:
            sequences.append(sequence)
        if len(sequences) >= top_sequences:
            return sequences
    if not sequences and task.labels:
        sequence = str(task.labels[0].get("sequence", ""))
        if sequence:
            sequences.append(sequence)
    return sequences


def _selected_case_samples_for_task(task: Any) -> list[tuple[str, str]]:
    default_method = _default_method(task.leaderboard)
    cases = task.case_rows.get(default_method, {})
    samples: list[tuple[str, str]] = []
    for case_type in ("true_positive", "false_positive", "false_negative"):
        for row in cases.get(case_type, [])[:4]:
            sample_id = str(row.get("sample_id", ""))
            sequence = str(row.get("sequence") or sample_id.split(":", 1)[0])
            samples.append((sample_id, sequence))
    if samples:
        return samples
    method = task.methods.get(default_method)
    if method is None and task.methods:
        method = next(iter(task.methods.values()))
    if method is not None:
        ranked = sorted(
            method.score_rows,
            key=lambda row: float(row.get("score", 0.0) or 0.0),
            reverse=True,
        )
        for row in ranked[:12]:
            sample_id = str(row.get("sample_id", ""))
            sequence = str(row.get("sequence") or sample_id.split(":", 1)[0])
            if sample_id and sequence:
                samples.append((sample_id, sequence))
    return samples


def _prioritized_trajectories(
    trajectories: list[dict[str, Any]],
    priority_samples: set[str],
) -> list[dict[str, Any]]:
    if not priority_samples:
        return trajectories
    indexed = list(enumerate(trajectories))
    indexed.sort(
        key=lambda item: (
            0 if str(item[1].get("sample_id", "")) in priority_samples else 1,
            item[0],
        )
    )
    return [trajectory for _, trajectory in indexed]


def _aggregate_labels_by_sample(labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for row in labels:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        label = int(row.get("label", 0) or 0)
        frame_start = int(row.get("frame_start", 0) or 0)
        frame_end = int(row.get("frame_end", frame_start) or frame_start)
        existing = aggregated.get(sample_id)
        if existing is None:
            aggregated[sample_id] = {
                "sample_id": sample_id,
                "sequence": str(row.get("sequence", "")),
                "track_id": str(row.get("track_id", "")),
                "label": label,
                "anomaly_type": str(row.get("anomaly_type", "normal")),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "num_windows": 1,
                "positive_windows": 1 if label == 1 else 0,
            }
            continue
        existing["label"] = max(int(existing.get("label", 0) or 0), label)
        existing["frame_start"] = min(int(existing.get("frame_start", frame_start) or frame_start), frame_start)
        existing["frame_end"] = max(int(existing.get("frame_end", frame_end) or frame_end), frame_end)
        existing["num_windows"] = int(existing.get("num_windows", 1) or 1) + 1
        existing["positive_windows"] = int(existing.get("positive_windows", 0) or 0) + (1 if label == 1 else 0)
        if label == 1 and str(existing.get("anomaly_type", "normal")) == "normal":
            existing["anomaly_type"] = str(row.get("anomaly_type", "anomaly"))
    return aggregated


def _aggregate_scores_by_sample(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        score = float(row.get("score", 0.0) or 0.0)
        existing = aggregated.get(sample_id)
        if existing is None or score > float(existing.get("score", 0.0) or 0.0):
            converted = dict(row)
            converted["sample_id"] = sample_id
            converted["score"] = score
            aggregated[sample_id] = converted
    return aggregated


def _group_labels_by_sample(labels: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in labels:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            continue
        grouped.setdefault(sample_id, []).append(
            {
                "label": int(row.get("label", 0) or 0),
                "anomaly_type": str(row.get("anomaly_type", "normal")),
                "frame_start": int(row.get("frame_start", 0) or 0),
                "frame_end": int(row.get("frame_end", 0) or 0),
                "sample_id": sample_id,
                "sequence": str(row.get("sequence", "")),
                "track_id": str(row.get("track_id", "")),
            }
        )
    return grouped


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _normalize_weighted_score(row: dict[str, Any], key_prefix: str, fallback_weight: float = 1.0) -> float:
    values = [
        _coerce_float(value, 0.0)
        for key, value in (row.get("component_scores") or {}).items()
        if isinstance(key, str) and key.startswith(f"{key_prefix}_") and _coerce_float(value, 0.0) > 0
    ]
    if values:
        return max(values)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if key_prefix == "individual":
        metadata_score = _coerce_float(metadata.get("individual_raw_score"), 0.0)
        return metadata_score if metadata_score else fallback_weight * _coerce_float(row.get("score"), 0.0)
    if key_prefix == "group":
        metadata_score = _coerce_float(metadata.get("group_raw_score"), 0.0)
        return metadata_score if metadata_score else fallback_weight * _coerce_float(row.get("score"), 0.0)
    return fallback_weight * _coerce_float(row.get("score"), 0.0)


def _source_tokens(row: dict[str, Any]) -> set[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    values = [row.get("used_sources"), metadata.get("used_sources"), row.get("source")]
    tokens: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        if isinstance(value, (list, tuple, set)):
            parts = value
        else:
            text = str(value).replace(",", "|").replace(" ", "|")
            parts = text.split("|")
        for part in parts:
            token = str(part).strip().lower()
            if token:
                tokens.add(token)
                if "individual" in token:
                    tokens.add("individual")
                if "group" in token:
                    tokens.add("group")
                if "registration" in token:
                    tokens.add("registration")
    return tokens


def _component_float(components: dict[str, Any], key: str) -> float | None:
    if key not in components:
        return None
    try:
        value = float(components.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _score_decomposition(row: dict[str, Any]) -> dict[str, float]:
    components = row.get("component_scores") if isinstance(row.get("component_scores"), dict) else {}
    source_tokens = _source_tokens(row)
    has_individual = "individual" in source_tokens
    has_group = "group" in source_tokens
    has_registration = "registration" in source_tokens
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    individual_raw = _coerce_float(metadata.get("individual_raw_score", 0.0), 0.0)
    group_raw = _coerce_float(metadata.get("group_raw_score", 0.0), 0.0)
    alpha = _coerce_float(
        metadata.get("alpha", 0.65) if isinstance(metadata.get("alpha"), (int, float)) else 0.65,
        0.65,
    )
    fused = _component_float(components, "S_fused")
    if fused is None:
        fused = _coerce_float(row.get("score", 0.0), 0.0)
    explicit_ind = _component_float(components, "S_ind")
    explicit_grp = _component_float(components, "S_grp")
    explicit_evt = _component_float(components, "S_event")
    ind = explicit_ind if explicit_ind is not None else 0.0
    grp = explicit_grp if explicit_grp is not None else 0.0
    evt = explicit_evt if explicit_evt is not None else _coerce_float(row.get("event_score", 0.0), 0.0)

    if has_registration:
        registration_values = [
            _coerce_float(value, 0.0)
            for key, value in components.items()
            if isinstance(key, str) and key.startswith("registration_")
        ]
        evt = max(registration_values or [fused])

    if has_individual and has_group:
        if explicit_ind is None:
            ind = _normalize_weighted_score(row, "individual", fallback_weight=alpha)
        if explicit_grp is None:
            grp = _normalize_weighted_score(row, "group", fallback_weight=(1 - alpha))
        if not ind and individual_raw:
            ind = individual_raw
        if not grp and group_raw:
            grp = group_raw
    elif has_individual:
        if explicit_ind is None:
            ind = _normalize_weighted_score(row, "individual", fallback_weight=1.0)
    elif has_group:
        if explicit_grp is None:
            grp = _normalize_weighted_score(row, "group", fallback_weight=1.0)

    return {
        "S_ind": float(ind),
        "S_grp": float(grp),
        "S_event": float(evt),
        "S_fused": float(fused),
        "individual_source": 1.0 if has_individual else 0.0,
        "group_source": 1.0 if has_group else 0.0,
        "alpha": float(alpha),
        "component_count": float(len(components)),
    }


def _score_component_payload(method_rows: dict[str, Any]) -> dict[str, Any]:
    frame_event_scores = normalize_frame_event_scores(method_rows.get("frame_event_scores", []))
    raw_event_segments = method_rows.get("event_segments", [])
    event_segments = [
        dict(segment)
        for segment in raw_event_segments
        if isinstance(segment, dict)
    ] if isinstance(raw_event_segments, list) else []
    if not event_segments:
        event_segments = event_segments_from_frame_scores(frame_event_scores)
    event_score = method_rows.get("event_score")
    if event_score is None:
        event_score = max((float(row.get("score", 0.0) or 0.0) for row in frame_event_scores), default=0.0)
    return {
        "score": round(float(method_rows.get("score", 0.0) or 0.0), 6),
        "used_sources": str(method_rows.get("used_sources", "")),
        "source": str(method_rows.get("source", "")),
        "event_score": event_score,
        "event_segments": event_segments,
        "frame_event_scores": frame_event_scores,
        "component_scores": method_rows.get("component_scores", {}),
        "metadata": method_rows.get("metadata", {}),
        "rotation_error_deg": method_rows.get("rotation_error_deg"),
        "translation_error": method_rows.get("translation_error"),
        "chamfer_distance": method_rows.get("chamfer_distance"),
        "runtime_sec": method_rows.get("runtime_sec"),
        "success": method_rows.get("success"),
        "skipped": method_rows.get("skipped"),
        "registration_points": method_rows.get("registration_points"),
    }


def _track_label_payload(label: dict[str, Any], default_start: int, default_end: int) -> dict[str, Any]:
    return {
        "label": int(label.get("label", 0) or 0),
        "anomaly_type": str(label.get("anomaly_type", "normal")),
        "frame_start": int(label.get("frame_start", default_start) or default_start),
        "frame_end": int(label.get("frame_end", default_end) or default_end),
        "num_windows": int(label.get("num_windows", 0) or 0),
        "positive_windows": int(label.get("positive_windows", 0) or 0),
    }


def _default_method(leaderboard: list[dict[str, Any]]) -> str:
    for row in leaderboard:
        if row.get("is_our_method"):
            return str(row["method"])
    return str(leaderboard[0]["method"]) if leaderboard else ""


def _render_html(dashboard_data: dict[str, Any], playback_payloads: dict[str, Any]) -> str:
    dashboard_json = json.dumps(dashboard_data, ensure_ascii=True).replace("</", "<\\/")
    playback_json = json.dumps(playback_payloads, ensure_ascii=True).replace("</", "<\\/")
    initial_task = "individual" if "individual" in dashboard_data["tasks"] else next(iter(dashboard_data["tasks"]), "")
    initial_method = _default_method(dashboard_data["tasks"].get(initial_task, {}).get("leaderboard", []))
    initial_sequence = next(iter(playback_payloads), "")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FusionTrack 最终结果看板</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f5f7fb; line-height: 1.5; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 24px 24px 40px; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 28px; line-height: 1.15; }}
    h2 {{ margin: 0; font-size: 18px; line-height: 1.25; }}
    .subtle {{ color: #64748b; font-size: 13px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: end; justify-content: flex-end; }}
    label {{ display: grid; gap: 5px; color: #475569; font-size: 12px; font-weight: 700; }}
    select, button, input {{ min-height: 44px; border: 1px solid #cbd5e1; border-radius: 7px; padding: 8px 10px; background: white; color: #0f172a; }}
    select {{ min-width: 170px; }}
    button {{ cursor: pointer; font-weight: 700; transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease, transform 160ms ease; touch-action: manipulation; }}
    button:hover {{ border-color: #94a3b8; background: #f8fafc; }}
    button:active {{ transform: translateY(1px); }}
    select:focus-visible, button:focus-visible, input:focus-visible {{ outline: 3px solid rgba(14, 116, 144, 0.28); outline-offset: 2px; }}
    input[type="range"] {{ min-height: 32px; padding: 0; accent-color: #0f766e; cursor: pointer; }}
    .panel {{ background: white; border: 1px solid #e1e7ef; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }}
    .card {{ background: white; border: 1px solid #e1e7ef; border-radius: 8px; padding: 11px 13px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035); }}
    .card > div:first-child {{ color: #64748b; font-size: 12px; font-weight: 700; }}
    .value {{ margin-top: 2px; font-size: 25px; line-height: 1.05; font-weight: 800; color: #0f172a; font-variant-numeric: tabular-nums; }}
    .leaderboard, .type-table, .case-list {{ width: 100%; min-width: 760px; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #475569; font-weight: 800; background: #f8fafc; position: sticky; top: 0; z-index: 1; }}
    .metric {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 3px 8px; background: #eef2ff; color: #3730a3; font-size: 12px; }}
    .our {{ background: #ecfdf5; color: #047857; }}
    .baseline {{ background: #f8fafc; color: #475569; }}
    .case-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
    .case-tab {{ border: 1px solid #cbd5e1; background: white; border-radius: 999px; padding: 7px 12px; }}
    .case-tab.active {{ background: #111827; border-color: #111827; color: white; }}
    .player {{ margin-top: 0; }}
    .section-heading {{ display: flex; justify-content: space-between; gap: 14px; align-items: start; margin-bottom: 12px; }}
    .section-heading .subtle {{ max-width: 780px; text-align: right; font-variant-numeric: tabular-nums; }}
    .control-surface {{ display: grid; gap: 10px; margin-bottom: 12px; padding: 12px; border: 1px solid #e2e8f0; border-radius: 8px; background: #f8fafc; }}
    .player-tools {{ display: grid; grid-template-columns: auto minmax(220px, 1fr) auto; gap: 12px; align-items: end; }}
    .secondary-button {{ padding: 8px 14px; }}
    .secondary-button.active {{ background: #111827; border-color: #111827; color: white; }}
    .mode-switch {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .view-mode-button {{ min-height: 44px; padding: 7px 12px; }}
    .view-mode-button.active {{ background: #111827; border-color: #111827; color: white; }}
    .view-mode-button:disabled {{ opacity: 0.45; cursor: not-allowed; background: #f8fafc; color: #64748b; }}
    .layer-switch {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .layer-button {{ min-height: 44px; padding: 7px 12px; }}
    .layer-button.active {{ background: #111827; border-color: #111827; color: white; }}
    .mode-switch[hidden], .heat-controls[hidden], .layer-switch[hidden], .comparison-grid[hidden], .single-view[hidden], .registration-playback[hidden] {{ display: none; }}
    .heat-controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
    .heat-controls label {{ min-width: 180px; max-width: 260px; }}
    .sequence-stats {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 8px; }}
    .sequence-stat {{ border: 1px solid #e1e7ef; border-radius: 7px; padding: 8px 10px; background: white; }}
    .sequence-stat span {{ display: block; color: #64748b; font-size: 12px; }}
    .sequence-stat strong {{ display: block; margin-top: 3px; font-size: 17px; }}
    .background-notice {{ border: 1px solid #fed7aa; border-radius: 7px; background: #fff7ed; color: #9a3412; padding: 9px 11px; font-size: 13px; line-height: 1.5; }}
    .background-notice[hidden] {{ display: none; }}
    #frameBadge {{ color: #475569; font-size: 13px; font-variant-numeric: tabular-nums; }}
    .canvas-shell {{ background: #111827; border-radius: 8px; padding: 10px; }}
    .comparison-grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; }}
    .video-panel {{ min-width: 0; margin: 0; border: 1px solid #1f2937; border-radius: 8px; padding: 9px; background: #0f172a; }}
    .video-panel figcaption {{ display: flex; align-items: center; min-height: 24px; margin: 0 0 7px; color: #f8fafc; font-size: 12px; font-weight: 800; }}
    .registration-playback {{ display: grid; grid-template-columns: minmax(320px, 1.6fr) minmax(260px, 0.8fr); gap: 12px; }}
    .registration-canvas-shell {{ background: #0f172a; border: 1px solid #1f2937; border-radius: 8px; padding: 10px; min-width: 0; }}
    .registration-canvas-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 8px; color: #f8fafc; }}
    .registration-canvas-head strong {{ font-size: 13px; }}
    .registration-canvas-head span {{ color: #cbd5e1; font-size: 12px; }}
    .registration-side-panel {{ border: 1px solid #dbe4ee; border-radius: 8px; background: #f8fafc; padding: 12px; min-width: 0; }}
    .registration-side-panel h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .registration-side-panel .explain-metrics {{ grid-template-columns: 1fr; }}
    .registration-cloud-legend {{ display: flex; flex-wrap: wrap; gap: 9px; margin-top: 9px; color: #cbd5e1; font-size: 12px; }}
    canvas {{ display: block; width: 100%; height: auto; background: #e2e8f0; border-radius: 6px; }}
    section {{ margin-top: 16px; }}
    .analysis-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .analysis-tab.active {{ background: #111827; border-color: #111827; color: white; }}
    .analysis-panel-block[hidden] {{ display: none; }}
    .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; border: 1px solid #eef2f7; border-radius: 8px; }}
    .table-scroll table {{ background: white; }}
    .help-button {{ background: #0f766e; border-color: #0f766e; color: white; }}
    .help-button:hover {{ background: #115e59; border-color: #115e59; }}
    .protocol-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin: 0 0 16px; }}
    .protocol-note {{ border-left: 4px solid #0f766e; }}
    .protocol-note strong {{ display: block; margin-bottom: 6px; color: #0f172a; }}
    .protocol-card h3, .insight-card h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .type-cloud {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }}
    .type-chip {{ display: inline-flex; gap: 6px; align-items: center; min-height: 30px; border: 1px solid #dbe4ee; border-radius: 999px; padding: 4px 9px; background: #f8fafc; color: #334155; font-size: 12px; }}
    .type-chip strong {{ color: #0f172a; font-variant-numeric: tabular-nums; }}
    .insight-grid {{ display: grid; grid-template-columns: minmax(250px, 0.9fr) minmax(280px, 1.1fr) minmax(260px, 0.9fr); gap: 12px; margin-top: 12px; }}
    .insight-card {{ border: 1px solid #e1e7ef; border-radius: 8px; background: #f8fafc; padding: 12px; min-width: 0; }}
    .insight-grid-large {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; margin-top: 12px; }}
    .track-rank-list {{ display: grid; gap: 7px; max-height: 260px; overflow: auto; padding-right: 2px; }}
    .track-rank-item {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: center; min-height: 44px; border: 1px solid #dbe4ee; border-radius: 7px; background: white; padding: 8px 10px; text-align: left; }}
    .track-rank-item.active {{ border-color: #0f766e; box-shadow: inset 3px 0 0 #0f766e; }}
    .track-rank-item .score {{ font-weight: 800; font-variant-numeric: tabular-nums; color: #be123c; }}
    .explain-metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
    .explain-metric {{ border: 1px solid #dbe4ee; border-radius: 7px; background: white; padding: 8px; }}
    .explain-metric span {{ display: block; color: #64748b; font-size: 12px; }}
    .explain-metric strong {{ display: block; margin-top: 2px; font-variant-numeric: tabular-nums; }}
    .explain-reason {{ margin-top: 10px; color: #334155; font-size: 13px; }}
    .submodule-switch {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 10px; align-items: center; }}
    .submodule-tab {{ min-height: 34px; padding: 6px 10px; border-radius: 999px; font-size: 12px; }}
    .submodule-tab.active {{ background: #111827; border-color: #111827; color: white; }}
    .flow-steps {{ display: grid; gap: 7px; }}
    .flow-step {{ display: grid; grid-template-columns: 1fr auto; align-items: center; border: 1px solid #dbe4ee; border-radius: 7px; padding: 8px 10px; background: #f8fafc; font-size: 12px; }}
    .flow-step-text {{ color: #334155; }}
    .flow-step-state {{ font-size: 12px; font-weight: 700; color: #334155; }}
    .flow-step.done {{ border-color: #cbd5e1; background: #ffffff; }}
    .flow-step.active {{ border-color: #0f766e; background: #ecfeff; }}
    .flow-step.pending {{ border-color: #cbd5e1; opacity: 0.8; }}
    .flow-step.done .flow-step-state {{ color: #16a34a; }}
    .flow-step.active .flow-step-state {{ color: #0ea5e9; }}
    .flow-step.pending .flow-step-state {{ color: #94a3b8; }}
    .decomp-bar {{ display: grid; gap: 7px; }}
    .decomp-row {{ display: grid; align-items: center; grid-template-columns: 68px 1fr 54px; gap: 8px; font-size: 12px; color: #334155; }}
    .decomp-track {{ position: relative; height: 10px; border-radius: 999px; background: #dbe4ee; overflow: hidden; }}
    .decomp-fill {{ height: 100%; background: linear-gradient(90deg, #0284c7, #22c55e); border-radius: inherit; }}
    .timeline {{ display: grid; gap: 8px; margin-top: 10px; }}
    .timeline-item {{ position: relative; border: 1px solid #dbe4ee; border-radius: 7px; padding: 8px; background: #f8fafc; }}
    .timeline-label {{ font-size: 12px; color: #334155; margin-bottom: 6px; }}
    .timeline-strip {{ position: relative; height: 14px; border-radius: 7px; background: #e2e8f0; overflow: hidden; }}
    .timeline-segment {{ position: absolute; top: 0; height: 100%; }}
    .timeline-segment.gt {{ background: #7c3aed; opacity: 0.7; }}
    .timeline-segment.pred {{ background: #dc2626; opacity: 0.7; }}
    .timeline-segment.overlap {{ background: #059669; opacity: 0.8; }}
    .method-summary {{ display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)); gap: 8px; margin-bottom: 12px; }}
    .method-summary-item {{ border: 1px solid #e1e7ef; border-radius: 7px; background: #f8fafc; padding: 8px 10px; }}
    .method-summary-item span {{ display: block; color: #64748b; font-size: 12px; }}
    .method-summary-item strong {{ display: block; margin-top: 2px; }}
    .data-flow-grid {{ display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 10px; }}
    .data-flow-card {{ border: 1px solid #e1e7ef; border-radius: 7px; background: #f8fafc; padding: 10px; }}
    .data-flow-card span {{ display: block; color: #64748b; font-size: 12px; }}
    .data-flow-card strong {{ display: block; margin-top: 3px; font-size: 18px; font-variant-numeric: tabular-nums; }}
    .provenance-panel {{ grid-column: 1 / -1; border: 1px solid #dbe4ee; border-radius: 8px; background: #ffffff; padding: 12px; }}
    .provenance-panel h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .provenance-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 8px; }}
    .provenance-item {{ border: 1px solid #edf2f7; border-radius: 7px; background: #f8fafc; padding: 9px; min-width: 0; }}
    .provenance-item span {{ display: block; color: #64748b; font-size: 12px; }}
    .provenance-item strong {{ display: block; margin-top: 3px; overflow-wrap: anywhere; color: #0f172a; font-size: 13px; }}
    .status-pill {{ display: inline-flex; align-items: center; min-height: 24px; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 800; background: #e2e8f0; color: #334155; }}
    .status-pill.ok {{ background: #dcfce7; color: #166534; }}
    .status-pill.partial {{ background: #fef3c7; color: #92400e; }}
    .status-pill.missing {{ background: #fee2e2; color: #991b1b; }}
    .mini-chart {{ width: 100%; height: 96px; margin-top: 10px; border: 1px solid #dbe4ee; border-radius: 7px; background: white; }}
    .event-card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 8px; margin-top: 10px; }}
    .event-card {{ border: 1px solid #dbe4ee; border-radius: 7px; background: white; padding: 9px; font-size: 12px; }}
    .event-card strong {{ display: block; margin-bottom: 4px; color: #0f172a; }}
    .registration-preview {{ margin-top: 10px; border: 1px solid #dbe4ee; border-radius: 7px; background: white; padding: 8px; }}
    .registration-preview svg {{ display: block; width: 100%; height: 170px; }}
    .legend-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; color: #475569; font-size: 12px; }}
    .legend-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; }}
    dialog {{ width: min(920px, calc(100vw - 32px)); max-height: min(760px, calc(100vh - 32px)); border: 1px solid #cbd5e1; border-radius: 8px; padding: 0; color: #172033; box-shadow: 0 24px 70px rgba(15, 23, 42, 0.28); }}
    dialog::backdrop {{ background: rgba(15, 23, 42, 0.42); }}
    .help-dialog-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 16px 18px; border-bottom: 1px solid #e5e7eb; background: #f8fafc; }}
    .help-dialog-body {{ padding: 16px 18px 20px; overflow: auto; }}
    .help-section {{ margin-top: 16px; }}
    .help-section:first-child {{ margin-top: 0; }}
    .help-section h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .help-section p {{ margin: 6px 0; }}
    .help-section ul {{ margin: 8px 0 0; padding-left: 20px; }}
    @media (prefers-reduced-motion: reduce) {{
      button {{ transition: none; }}
    }}
    @media (max-width: 960px) {{
      main {{ padding: 16px; }}
      header {{ display: grid; }}
      .cards {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      select {{ min-width: 0; width: 100%; }}
      .player-tools {{ grid-template-columns: 1fr; }}
      .heat-controls label {{ max-width: none; width: 100%; }}
      .sequence-stats {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .comparison-grid {{ grid-template-columns: 1fr; }}
      .registration-playback {{ grid-template-columns: 1fr; }}
      .toolbar {{ display: grid; width: 100%; justify-content: stretch; }}
      .toolbar label {{ width: 100%; }}
      .section-heading {{ display: grid; }}
      .section-heading .subtle {{ text-align: left; }}
      .control-surface {{ padding: 10px; }}
      .mode-switch button, .layer-switch button {{ flex: 1 1 140px; }}
      .protocol-strip, .insight-grid, .method-summary, .data-flow-grid {{ grid-template-columns: 1fr; }}
      .insight-grid-large {{ grid-template-columns: 1fr; }}
      .explain-metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1 data-i18n="title">FusionTrack 最终结果看板</h1>
        <div class="subtle" data-i18n="subtitle">多方法多模态异常检测实验展示</div>
      </div>
      <div class="toolbar">
        <label><span data-i18n="language">语言</span>
          <select id="languageSelector">
            <option value="zh">中文</option>
            <option value="en">English</option>
          </select>
        </label>
        <label><span data-i18n="task">任务</span>
          <select id="taskSelector"></select>
        </label>
        <label><span data-i18n="method">方法</span>
          <select id="methodSelector"></select>
        </label>
        <label><span data-i18n="sequence">序列</span>
          <select id="sequenceSelector"></select>
        </label>
        <button type="button" class="help-button" id="helpButton" data-i18n="helpButton">说明</button>
      </div>
    </header>
    <div id="cards" class="cards"></div>

    <section class="protocol-strip" aria-label="Anomaly protocol overview">
      <div class="panel protocol-note">
        <strong data-i18n="protocolTitle">异常协议</strong>
        <div class="subtle" data-i18n="protocolNote">当前标签来自规则化 synthetic anomaly injection；背景帧仍是原始视频，异常主要体现在轨迹、热力和群体关系层。</div>
      </div>
      <div class="panel protocol-card" id="individualProtocol"></div>
      <div class="panel protocol-card" id="groupProtocol"></div>
      <div class="panel protocol-card" id="registrationProtocol"></div>
    </section>

    <section class="panel player">
      <div class="section-heading">
        <h2 data-i18n="interactivePlayback">Interactive playback</h2>
        <div class="subtle" id="playbackReadout">No playback loaded</div>
      </div>
      <div class="control-surface">
        <div class="player-tools">
          <button type="button" class="secondary-button" id="playToggle">Play</button>
          <label><span data-i18n="frame">帧</span>
            <input id="frameSlider" type="range" min="0" max="0" value="0">
          </label>
          <span id="frameBadge">0 / 0</span>
        </div>
        <div class="mode-switch" aria-label="Visualization mode">
          <button type="button" class="view-mode-button active" data-view-mode="comparison" data-i18n="viewComparison">四画面对比</button>
          <button type="button" class="view-mode-button" data-view-mode="single" data-i18n="viewSingle">单画面模式</button>
        </div>
        <div class="layer-switch" id="singleLayerSwitch" aria-label="Single playback layer" hidden>
          <span class="subtle" data-i18n="singleLayerLabel">单画面图层</span>
          <button type="button" class="layer-button" data-layer="tracks" data-i18n="layerTracks">Tracks</button>
          <button type="button" class="layer-button active" data-layer="both" data-i18n="layerBoth">Heat + Tracks</button>
          <button type="button" class="layer-button" data-layer="heatmap" data-i18n="layerHeatmap">Heatmap</button>
        </div>
        <div class="heat-controls">
          <label><span data-i18n="heatOpacityLabel">热力透明度</span>
            <input id="heatOpacity" type="range" min="0" max="100" value="64">
          </label>
          <label><span data-i18n="timeWindowLabel">时间窗口</span>
            <input id="heatWindow" type="range" min="12" max="120" value="36">
          </label>
          <label><span data-i18n="eventThresholdLabel">Event threshold</span>
            <input id="eventThreshold" type="range" min="0" max="100" value="0">
            <span id="eventThresholdReadout" class="subtle">0.00</span>
          </label>
          <label><span data-i18n="playSpeedLabel">播放速度</span>
            <input id="playSpeed" type="range" min="20" max="300" step="10" value="100">
            <span id="playSpeedReadout" class="subtle">1.0x</span>
          </label>
        </div>
        <div id="sequenceStats" class="sequence-stats"></div>
        <div id="backgroundNotice" class="background-notice" role="status" hidden></div>
      </div>
      <div id="comparisonView" class="comparison-grid">
        <figure class="video-panel">
          <figcaption data-i18n="panelOriginal">原视频</figcaption>
          <canvas id="originalCanvas" width="960" height="612"></canvas>
        </figure>
        <figure class="video-panel">
          <figcaption data-i18n="panelHeatmap">热力图</figcaption>
          <canvas id="heatmapCanvas" width="960" height="612"></canvas>
        </figure>
        <figure class="video-panel">
          <figcaption data-i18n="panelTracks">轨迹</figcaption>
          <canvas id="tracksCanvas" width="960" height="612"></canvas>
        </figure>
        <figure class="video-panel">
          <figcaption data-i18n="panelBoth">热力 + 轨迹</figcaption>
          <canvas id="bothCanvas" width="960" height="612"></canvas>
        </figure>
      </div>
      <div id="singleView" class="single-view" hidden>
        <div class="canvas-shell"><canvas id="singleCanvas" width="960" height="612"></canvas></div>
      </div>
      <div id="registrationPlaybackView" class="registration-playback" hidden>
        <div class="registration-canvas-shell">
          <div class="registration-canvas-head">
            <strong data-i18n="registrationPlaybackTitle">Point-cloud registration view</strong>
            <span data-i18n="registrationPlaybackNote">Source, reference, and aligned point clouds rotate in the diagnostic canvas.</span>
          </div>
          <canvas id="registrationCanvas" width="960" height="520"></canvas>
          <div class="registration-cloud-legend">
            <span><i class="legend-dot" style="background:#3b82f6"></i><span data-i18n="registrationSourceCloud">Source</span></span>
            <span><i class="legend-dot" style="background:#22c55e"></i><span data-i18n="registrationReferenceCloud">Reference</span></span>
            <span><i class="legend-dot" style="background:#ef4444"></i><span data-i18n="registrationAlignedCloud">Aligned</span></span>
          </div>
        </div>
        <div class="registration-side-panel">
          <h3 data-i18n="registrationEvidenceTitle">Registration evidence</h3>
          <div id="registrationPlaybackSummary" class="subtle"></div>
        </div>
      </div>
      <div class="insight-grid">
        <div class="insight-card">
          <h3 data-i18n="trackRankTitle">当前高风险轨迹</h3>
          <div class="track-rank-list" id="trackRankList"></div>
        </div>
        <div class="insight-card">
          <h3 data-i18n="explainTitle">为什么判为异常</h3>
          <div id="explanationPanel" class="subtle"></div>
        </div>
        <div class="insight-card">
          <h3 data-i18n="groupInsightTitle">群体关系</h3>
          <div id="groupInsightPanel" class="subtle"></div>
        </div>
      </div>
      <div class="insight-grid-large">
        <div class="insight-card">
          <h3 data-i18n="methodFlowTitle">Method flow</h3>
          <div id="methodFlowPanel" class="flow-steps subtle">--</div>
          <div id="flowReadout" class="subtle" style="margin-top: 8px;"></div>
        </div>
        <div class="insight-card">
          <h3 data-i18n="submoduleTitle">Individual submodules</h3>
          <div class="submodule-switch" id="submoduleSwitch">
            <span class="subtle" data-i18n="submodulePrefix">Submodule</span>
            <button type="button" class="submodule-tab active" data-submodule="route" data-i18n="submoduleRoute">Route</button>
            <button type="button" class="submodule-tab" data-submodule="speed" data-i18n="submoduleSpeed">Speed</button>
            <button type="button" class="submodule-tab" data-submodule="shape" data-i18n="submoduleShape">Shape</button>
          </div>
          <div id="submodulePanel" class="subtle"></div>
          <div class="decomp-bar" id="scoreCompositionPanel"></div>
        </div>
        <div class="insight-card">
          <h3 data-i18n="timelineTitle">Event timeline</h3>
          <div id="eventTimelinePanel" class="subtle"></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="section-heading">
        <h2 data-i18n="analysisTitle">实验分析</h2>
      </div>
      <div class="analysis-tabs">
        <button type="button" class="analysis-tab active" data-panel="leaderboard" data-i18n="tabLeaderboard">方法排名</button>
        <button type="button" class="analysis-tab" data-panel="types" data-i18n="tabTypes">异常类型分析</button>
        <button type="button" class="analysis-tab" data-panel="cases" data-i18n="tabCases">典型案例</button>
        <button type="button" class="analysis-tab" data-panel="methods" data-i18n="tabMethods">算法接入</button>
        <button type="button" class="analysis-tab" data-panel="dataflow" data-i18n="tabDataFlow">数据流审计</button>
      </div>
      <div class="analysis-panel-block" data-analysis-panel="leaderboard">
        <div class="table-scroll"><table class="leaderboard" id="leaderboardTable"></table></div>
      </div>
      <div class="analysis-panel-block" data-analysis-panel="types" hidden>
        <div class="table-scroll"><table class="type-table" id="typeTable"></table></div>
      </div>
      <div class="analysis-panel-block" data-analysis-panel="cases" hidden>
        <div class="case-tabs">
          <button type="button" class="case-tab active" data-case="true_positive">True Positive</button>
          <button type="button" class="case-tab" data-case="false_positive">False Positive</button>
          <button type="button" class="case-tab" data-case="false_negative">False Negative</button>
        </div>
        <div class="table-scroll"><table class="case-list" id="caseTable"></table></div>
      </div>
      <div class="analysis-panel-block" data-analysis-panel="methods" hidden>
        <div id="methodSummary" class="method-summary"></div>
        <div class="table-scroll"><table class="leaderboard" id="methodStatusTable"></table></div>
      </div>
      <div class="analysis-panel-block" data-analysis-panel="dataflow" hidden>
        <div id="dataFlowPanel" class="data-flow-grid"></div>
      </div>
    </section>
    <dialog id="helpDialog">
      <div class="help-dialog-head">
        <h2 data-i18n="helpTitle">网页说明</h2>
        <button type="button" class="secondary-button" id="helpClose" data-i18n="closeHelp">关闭</button>
      </div>
      <div class="help-dialog-body" id="helpBody"></div>
    </dialog>
  </main>
  <script id="dashboardData" type="application/json">{dashboard_json}</script>
  <script id="playbackData" type="application/json">{playback_json}</script>
  <script>
    (() => {{
      const dashboard = JSON.parse(document.getElementById("dashboardData").textContent);
      const playbackData = JSON.parse(document.getElementById("playbackData").textContent);
      const languageSelector = document.getElementById("languageSelector");
      const taskSelector = document.getElementById("taskSelector");
      const methodSelector = document.getElementById("methodSelector");
      const cards = document.getElementById("cards");
      const individualProtocol = document.getElementById("individualProtocol");
      const groupProtocol = document.getElementById("groupProtocol");
      const registrationProtocol = document.getElementById("registrationProtocol");
      const leaderboardTable = document.getElementById("leaderboardTable");
      const typeTable = document.getElementById("typeTable");
      const caseTable = document.getElementById("caseTable");
      const methodSummary = document.getElementById("methodSummary");
      const methodStatusTable = document.getElementById("methodStatusTable");
      const dataFlowPanel = document.getElementById("dataFlowPanel");
      const helpButton = document.getElementById("helpButton");
      const helpDialog = document.getElementById("helpDialog");
      const helpClose = document.getElementById("helpClose");
      const helpBody = document.getElementById("helpBody");
      const trackRankList = document.getElementById("trackRankList");
      const explanationPanel = document.getElementById("explanationPanel");
      const groupInsightPanel = document.getElementById("groupInsightPanel");
      const methodFlowPanel = document.getElementById("methodFlowPanel");
      const flowReadout = document.getElementById("flowReadout");
      const submoduleSwitch = document.getElementById("submoduleSwitch");
      const submodulePanel = document.getElementById("submodulePanel");
      const scoreCompositionPanel = document.getElementById("scoreCompositionPanel");
      const eventTimelinePanel = document.getElementById("eventTimelinePanel");
      const submoduleTabs = Array.from(document.querySelectorAll(".submodule-tab"));
      const caseTabs = Array.from(document.querySelectorAll(".case-tab"));
      const analysisTabs = Array.from(document.querySelectorAll(".analysis-tab"));
      const analysisPanels = Array.from(document.querySelectorAll("[data-analysis-panel]"));
      const canvases = {{
        original: document.getElementById("originalCanvas"),
        heatmap: document.getElementById("heatmapCanvas"),
        tracks: document.getElementById("tracksCanvas"),
        both: document.getElementById("bothCanvas"),
        single: document.getElementById("singleCanvas"),
        registration: document.getElementById("registrationCanvas")
      }};
      const comparisonView = document.getElementById("comparisonView");
      const singleView = document.getElementById("singleView");
      const registrationPlaybackView = document.getElementById("registrationPlaybackView");
      const registrationPlaybackSummary = document.getElementById("registrationPlaybackSummary");
      const modeSwitch = document.querySelector(".mode-switch");
      const singleLayerSwitch = document.getElementById("singleLayerSwitch");
      const playbackReadout = document.getElementById("playbackReadout");
      const sequenceSelector = document.getElementById("sequenceSelector");
      const playToggle = document.getElementById("playToggle");
      const frameSlider = document.getElementById("frameSlider");
      const frameBadge = document.getElementById("frameBadge");
      const heatControlsPanel = document.querySelector(".heat-controls");
      const heatOpacity = document.getElementById("heatOpacity");
      const heatWindow = document.getElementById("heatWindow");
      const eventThreshold = document.getElementById("eventThreshold");
      const eventThresholdReadout = document.getElementById("eventThresholdReadout");
      const playSpeed = document.getElementById("playSpeed");
      const playSpeedReadout = document.getElementById("playSpeedReadout");
      const sequenceStats = document.getElementById("sequenceStats");
      const backgroundNotice = document.getElementById("backgroundNotice");
      const viewModeButtons = Array.from(document.querySelectorAll(".view-mode-button"));
      const layerButtons = Array.from(document.querySelectorAll(".layer-button"));
      const translations = {{
        zh: {{}}, /*
          documentTitle: "FusionTrack 最终结果看板",
          title: "FusionTrack 最终结果看板",
          subtitle: "多方法多模态异常检测实验展示",
          language: "语言",
          task: "任务",
          method: "方法",
          sequence: "序列",
          helpButton: "说明",
          helpTitle: "网页说明",
          closeHelp: "关闭",
          protocolTitle: "异常协议",
          protocolNote: "当前标签来自规则化 synthetic anomaly injection；背景帧仍是原始视频，异常主要体现在轨迹、热力和群体关系层。",
          cardMethods: "方法数",
          cardLabels: "总标签数",
          cardPositives: "总异常数",
          cardAuroc: "当前 AUROC",
          sequenceSampleCount: "当前序列样本数",
          sequenceAnomalyCount: "当前序列异常数",
          sequenceFrameRange: "当前序列帧范围",
          sequenceVisualizedTracks: "可视化轨迹数",
          analysisTitle: "实验分析",
          tabLeaderboard: "方法排名",
          tabTypes: "异常类型分析",
          tabCases: "典型案例",
          tabMethods: "算法接入",
          interactivePlayback: "动态可视化",
          frame: "帧",
          play: "播放",
          pause: "暂停",
          viewComparison: "四画面对比",
          viewSingle: "单画面模式",
          singleLayerLabel: "单画面图层",
          panelOriginal: "原视频",
          panelHeatmap: "热力图",
          panelTracks: "轨迹",
          panelBoth: "热力 + 轨迹",
          layerTracks: "轨迹",
          layerBoth: "热力 + 轨迹",
          layerHeatmap: "热力图",
          heatOpacityLabel: "热力透明度",
          timeWindowLabel: "时间窗口",
          playSpeedLabel: "播放速度",
          noPlayback: "当前任务没有可播放轨迹。",
          playbackPrefix: "可视化",
          visibleTracks: "条轨迹",
          methodHeader: "方法",
          roleHeader: "角色",
          anomalyTypeHeader: "异常类型",
          hitsHeader: "命中@K",
          totalHeader: "总数",
          recallHeader: "召回@K",
          meanScoreHeader: "平均正样本分数",
          sampleHeader: "样本",
          typeHeader: "类型",
          scoreHeader: "分数",
          rankHeader: "排名",
          framesHeader: "帧范围",
          sourceHeader: "来源",
          familyHeader: "方法族",
          learningHeader: "学习类型",
          statusHeader: "状态",
          trackRankTitle: "当前高风险轨迹",
          explainTitle: "为什么判为异常",
          groupInsightTitle: "群体关系",
          selectedTrack: "选中轨迹",
          anomalyLabel: "标签",
          anomalyScore: "异常分数",
          anomalyTypeLabel: "异常类型",
          frameRangeLabel: "标签帧段",
          motionLengthLabel: "轨迹长度",
          avgSpeedLabel: "平均速度",
          displacementLabel: "首尾位移",
          currentNeighborsLabel: "当前邻近对象",
          centroidRadiusLabel: "群体半径",
          syntheticLabel: "规则注入异常",
          normalLabel: "正常/未标注异常",
          methodSummaryTitle: "算法接入状态",
          integratedStatus: "已接入最终结果",
          proposedStatus: "当前主方法",
          baselineStatus: "对比基线",
          noTrackSelected: "当前没有可解释轨迹。",
          truePositive: "正确检出",
          falsePositive: "误报",
          falseNegative: "漏报",
          taskIndividual: "Individual",
          taskGroup: "Group",
          view_comparison: "四画面对比",
          view_single: "单画面模式",
          layer_tracks: "轨迹",
        layer_both: "热力 + 轨迹",
          layer_heatmap: "热力图",
          methodFlowTitle: "方法流程",
          flowStepPrepare: "准备输入",
          flowStepFeatures: "特征构建",
          flowStepIndividual: "单目标分支",
          flowStepGroup: "群体分支",
          flowStepFusion: "融合得分",
          flowReadoutTask: "任务",
          flowReadoutMethod: "方法",
          flowReadoutScore: "当前轨迹分数",
          methodFlowDone: "已完成",
          methodFlowActive: "进行中",
          methodFlowPending: "待执行",
          submoduleTitle: "Individual 子模块",
          submodulePrefix: "子模块",
          submoduleRoute: "轨迹/路由",
          submoduleSpeed: "速度",
          submoduleShape: "形状",
          noSubmoduleData: "当前轨迹暂无单目标子模块证据。",
          timelineTitle: "事件级时间线",
          timelineGt: "真实异常段",
          timelinePred: "模型预测段",
          noTimeline: "无可展示时间线",
          compSInd: "个体分数",
          compSGrp: "群体分数",
          compSEvent: "事件分数",
          compSFused: "融合分数",
          compNoData: "当前轨迹无分数分解数据。"
        }},
        */ en: {{
          documentTitle: "FusionTrack Final Results Dashboard",
          title: "Final Results Dashboard",
          subtitle: "Multi-method FusionTrack anomaly benchmark",
          language: "Language",
          task: "Task",
          method: "Method",
          sequence: "Sequence",
          helpButton: "Help",
          helpTitle: "Dashboard Help",
          closeHelp: "Close",
          protocolTitle: "Anomaly Protocol",
          protocolNote: "Labels come from a rule-based synthetic anomaly injection protocol. The background frames are original video frames; anomalies are expressed in tracks, heatmaps, and group relations.",
          cardMethods: "Methods",
          cardLabels: "Total labels",
          cardPositives: "Total anomalies",
          cardAuroc: "Selected AUROC",
          sequenceSampleCount: "Sequence samples",
          sequenceAnomalyCount: "Sequence anomalies",
          sequenceFrameRange: "Sequence frame range",
          sequenceVisualizedTracks: "Visualized tracks",
          analysisTitle: "Experiment Analysis",
          tabLeaderboard: "Method Ranking",
          tabTypes: "Anomaly-Type Analysis",
          tabCases: "Representative Cases",
          tabMethods: "Algorithm Status",
          interactivePlayback: "Interactive Playback",
          frame: "Frame",
          play: "Play",
          pause: "Pause",
          viewComparison: "Four-panel comparison",
          viewSingle: "Single view",
          singleLayerLabel: "Single-view layer",
          panelOriginal: "Original",
          panelHeatmap: "Heatmap",
          panelTracks: "Tracks",
          panelBoth: "Heat + Tracks",
          layerTracks: "Tracks",
          layerBoth: "Heat + Tracks",
          layerHeatmap: "Heatmap",
          heatOpacityLabel: "Heat opacity",
          timeWindowLabel: "Time window",
          playSpeedLabel: "Play speed",
          noPlayback: "Playback is not available for the current task.",
          backgroundLoading: "Loading original background frame...",
          backgroundLoadFailed: "Original background frame failed to load. Check whether the static assets folder was published with the page.",
          registrationNoVideoBackground: "Registration is a point-cloud alignment diagnostic task, so it has no VT-Tiny-MOT original video background. Inspect the registration evidence panel for source, reference, and aligned point clouds.",
          registrationNoVideoBackgroundShort: "Registration point-cloud task: no original video background",
          sequenceNoVideoBackground: "No original RGB background frame was found for this sequence. The page can still show tracks, heatmaps, and structured evidence.",
          sequenceNoVideoBackgroundShort: "No original background frame for this sequence",
          mediaKindVideo: "Original video background",
          mediaKindRegistration: "Point-cloud registration playback",
          mediaKindTrackOnly: "Track-only playback",
          comparisonRequiresBackground: "Four-panel comparison requires original video background frames.",
          playbackPrefix: "Playback",
          visibleTracks: "visible tracks",
          methodHeader: "Method",
          roleHeader: "Role",
          anomalyTypeHeader: "Anomaly type",
          hitsHeader: "Hits@K",
          totalHeader: "Total",
          recallHeader: "Recall@K",
          meanScoreHeader: "Mean positive score",
          sampleHeader: "Sample",
          typeHeader: "Type",
          scoreHeader: "Score",
          rankHeader: "Rank",
          framesHeader: "Frames",
          sourceHeader: "Source",
          familyHeader: "Family",
          learningHeader: "Learning",
          statusHeader: "Status",
          trackRankTitle: "High-Risk Tracks",
          explainTitle: "Why Anomaly",
          groupInsightTitle: "Group Relations",
          selectedTrack: "Selected track",
          anomalyLabel: "Label",
          anomalyScore: "Anomaly score",
          anomalyTypeLabel: "Anomaly type",
          frameRangeLabel: "Label frames",
          motionLengthLabel: "Path length",
          avgSpeedLabel: "Mean speed",
          displacementLabel: "Displacement",
          currentNeighborsLabel: "Current neighbors",
          centroidRadiusLabel: "Group radius",
          syntheticLabel: "Synthetic anomaly",
          normalLabel: "Normal / unlabeled",
          methodSummaryTitle: "Algorithm integration status",
          integratedStatus: "Integrated final result",
          proposedStatus: "Current proposed method",
          baselineStatus: "Comparison baseline",
          noTrackSelected: "No explainable track is selected.",
          truePositive: "True Positive",
          falsePositive: "False Positive",
          falseNegative: "False Negative",
          taskIndividual: "Individual",
          taskGroup: "Group",
          view_comparison: "four-panel comparison",
          view_single: "single view",
          layer_tracks: "tracks",
          layer_both: "heat + tracks",
          layer_heatmap: "heatmap",
          methodFlowTitle: "Method Flow",
          flowStepPrepare: "Prepare",
          flowStepFeatures: "Build features",
          flowStepIndividual: "Individual branch",
          flowStepGroup: "Group branch",
          flowStepFusion: "Fuse score",
          flowReadoutTask: "Task",
          flowReadoutMethod: "Method",
          flowReadoutScore: "Current track score",
          methodFlowDone: "Done",
          methodFlowActive: "Active",
          methodFlowPending: "Pending",
          submoduleTitle: "Individual submodules",
          submodulePrefix: "Submodule",
          submoduleRoute: "Route",
          submoduleSpeed: "Speed",
          submoduleShape: "Shape",
          noSubmoduleData: "No submodule evidence for this track.",
          timelineTitle: "Event timeline",
          timelineGt: "Ground truth segments",
          timelinePred: "Predicted segments",
          noTimeline: "No timeline data",
          compSInd: "Individual score",
          compSGrp: "Group score",
          compSEvent: "Event score",
          compSFused: "Fused score",
          compNoData: "No score decomposition data for this track."
        }}
      }};
      Object.assign(translations.zh, {{
        documentTitle: "FusionTrack 最终结果看板",
        title: "FusionTrack 最终结果看板",
        subtitle: "多方法、多模态异常检测实验展示",
        language: "语言",
        task: "任务",
        method: "方法",
        sequence: "序列",
        helpButton: "说明",
        helpTitle: "网页说明",
        closeHelp: "关闭",
        protocolTitle: "异常协议",
        protocolNote: "当前标签来自规则化 synthetic anomaly injection；背景帧仍是原始视频，异常主要体现在轨迹、热力和群体关系层。",
        cardMethods: "方法数",
        cardLabels: "总标签数",
        cardPositives: "总异常数",
        cardAuroc: "当前 AUROC",
        registrationSuccessRate: "当前成功率",
        registrationPairCount: "配准样本数",
        registrationFailedCount: "失败/跳过数",
        sequenceSampleCount: "当前序列样本数",
        sequenceAnomalyCount: "当前序列异常/高误差数",
        sequenceFrameRange: "当前序列帧范围",
        sequenceVisualizedTracks: "可视化轨迹数",
        analysisTitle: "实验分析",
        tabLeaderboard: "方法排名",
        tabTypes: "异常类型分析",
        tabCases: "典型案例",
        tabMethods: "算法接入",
        tabDataFlow: "数据流审计",
        interactivePlayback: "动态可视化",
        frame: "帧",
        play: "播放",
        pause: "暂停",
        viewComparison: "四画面对比",
        viewSingle: "单画面模式",
        singleLayerLabel: "单画面图层",
        panelOriginal: "原视频",
        panelHeatmap: "热力图",
        panelTracks: "轨迹",
        panelBoth: "热力 + 轨迹",
        layerTracks: "轨迹",
        layerBoth: "热力 + 轨迹",
        layerHeatmap: "热力图",
        heatOpacityLabel: "热力透明度",
        timeWindowLabel: "时间窗口",
        playSpeedLabel: "播放速度",
        noPlayback: "当前任务没有可播放轨迹。",
        backgroundLoading: "\u80cc\u666f\u5e27\u52a0\u8f7d\u4e2d...",
        backgroundLoadFailed: "\u80cc\u666f\u5e27\u52a0\u8f7d\u5931\u8d25\uff0c\u8bf7\u68c0\u67e5\u9759\u6001\u7f51\u9875 assets \u662f\u5426\u5df2\u540c\u6b65\u53d1\u5e03\u3002",
        registrationNoVideoBackground: "Registration \u662f\u70b9\u4e91\u914d\u51c6\u8bca\u65ad\u4efb\u52a1\uff0c\u6ca1\u6709 VT-Tiny-MOT \u539f\u59cb\u89c6\u9891\u80cc\u666f\uff1b\u8bf7\u5728\u4e0b\u65b9\u914d\u51c6\u8bc1\u636e\u67e5\u770b\u6e90\u70b9\u4e91\u3001\u53c2\u8003\u70b9\u4e91\u548c\u5bf9\u9f50\u7ed3\u679c\u3002",
        registrationNoVideoBackgroundShort: "\u914d\u51c6\u70b9\u4e91\u4efb\u52a1\uff0c\u65e0\u539f\u59cb\u89c6\u9891\u80cc\u666f",
        registrationPlaybackTitle: "\u70b9\u4e91\u914d\u51c6\u52a8\u6001\u89c6\u56fe",
        registrationPlaybackNote: "\u52a8\u6001\u5c55\u793a\u6e90\u70b9\u4e91\u3001\u53c2\u8003\u70b9\u4e91\u548c\u4f30\u8ba1\u5bf9\u9f50\u7ed3\u679c",
        registrationEvidenceTitle: "\u914d\u51c6\u8bc1\u636e",
        registrationSelectedPair: "\u5f53\u524d\u70b9\u4e91\u5bf9",
        registrationMethod: "\u914d\u51c6\u65b9\u6cd5",
        registrationPairStatus: "\u914d\u51c6\u72b6\u6001",
        registrationSourceCloud: "\u6e90\u70b9\u4e91",
        registrationReferenceCloud: "\u53c2\u8003\u70b9\u4e91",
        registrationAlignedCloud: "\u5bf9\u9f50\u7ed3\u679c",
        registrationNoPointCloud: "\u5f53\u524d\u6837\u672c\u6682\u65e0\u70b9\u4e91\u5750\u6807\uff0c\u6b63\u5728\u5c55\u793a\u8f7b\u91cf\u8bca\u65ad\u5360\u4f4d\u56fe\u3002",
        sequenceNoVideoBackground: "\u5f53\u524d\u5e8f\u5217\u6ca1\u6709\u627e\u5230\u53ef\u7528\u7684\u539f\u59cb RGB \u80cc\u666f\u5e27\uff0c\u53ea\u80fd\u5c55\u793a\u8f68\u8ff9\u3001\u70ed\u529b\u548c\u7ed3\u6784\u5316\u8bc1\u636e\u3002",
        sequenceNoVideoBackgroundShort: "\u5f53\u524d\u5e8f\u5217\u65e0\u539f\u59cb\u80cc\u666f\u5e27",
        mediaKindVideo: "\u539f\u59cb\u89c6\u9891\u80cc\u666f\u56de\u653e",
        mediaKindRegistration: "\u70b9\u4e91\u914d\u51c6\u56de\u653e",
        mediaKindTrackOnly: "\u65e0\u80cc\u666f\u8f68\u8ff9\u56de\u653e",
        comparisonRequiresBackground: "\u56db\u753b\u9762\u5bf9\u6bd4\u9700\u8981\u539f\u59cb\u89c6\u9891\u80cc\u666f\u5e27\u3002",
        playbackPrefix: "可视化",
        visibleTracks: "条轨迹",
        methodHeader: "方法",
        roleHeader: "角色",
        anomalyTypeHeader: "异常类型",
        hitsHeader: "命中@K",
        totalHeader: "总数",
        recallHeader: "召回@K",
        meanScoreHeader: "正样本平均分",
        sampleHeader: "样本",
        typeHeader: "类型",
        scoreHeader: "分数",
        rankHeader: "排名",
        framesHeader: "帧范围",
        sourceHeader: "来源",
        familyHeader: "方法族",
        learningHeader: "学习类型",
        statusHeader: "状态",
        trackRankTitle: "当前高风险轨迹",
        explainTitle: "为什么判为异常",
        groupInsightTitle: "群体/配准证据",
        selectedTrack: "选中轨迹",
        anomalyLabel: "标签",
        anomalyScore: "异常分数",
        anomalyTypeLabel: "异常类型",
        frameRangeLabel: "标签帧段",
        motionLengthLabel: "轨迹长度",
        avgSpeedLabel: "平均速度",
        displacementLabel: "首尾位移",
        currentNeighborsLabel: "当前邻近对象",
        centroidRadiusLabel: "群体半径",
        syntheticLabel: "规则注入异常",
        normalLabel: "正常/未标注异常",
        methodSummaryTitle: "算法接入状态",
        integratedStatus: "已接入最终结果",
        proposedStatus: "当前主方法",
        baselineStatus: "对比基线",
        noTrackSelected: "当前没有可解释轨迹。",
        truePositive: "正确检出",
        falsePositive: "误报/高风险",
        falseNegative: "漏报/失败",
        taskIndividual: "Individual",
        taskGroup: "Group",
        taskRegistration: "Registration",
        view_registration: "\u70b9\u4e91\u914d\u51c6\u89c6\u56fe",
        view_comparison: "四画面对比",
        view_single: "单画面模式",
        layer_tracks: "轨迹",
        layer_both: "热力 + 轨迹",
        layer_heatmap: "热力图",
        methodFlowTitle: "方法流程",
        flowStepPrepare: "准备输入",
        flowStepFeatures: "特征构建",
        flowStepIndividual: "单目标分支",
        flowStepGroup: "群体分支",
        flowStepFusion: "融合得分",
        flowStepRegistrationPair: "点云配准对",
        flowStepRegistrationFeature: "几何匹配/变换估计",
        flowStepRegistrationMetric: "误差度量",
        flowStepRegistrationScore: "配准风险得分",
        flowReadoutTask: "任务",
        flowReadoutMethod: "方法",
        flowReadoutScore: "当前轨迹分数",
        methodFlowDone: "已完成",
        methodFlowActive: "进行中",
        methodFlowPending: "待执行",
        submoduleTitle: "子模块证据",
        submodulePrefix: "子模块",
        submoduleRoute: "轨迹/路由",
        submoduleSpeed: "速度",
        submoduleShape: "形状",
        noSubmoduleData: "当前轨迹暂无单目标子模块证据。",
        timelineTitle: "事件级时间线",
        timelineGt: "真实异常段",
        timelinePred: "模型预测段",
        noTimeline: "无可展示时间线",
        compSInd: "个体分数",
        compSGrp: "群体分数",
        compSEvent: "事件分数",
        compSFused: "融合分数",
        compNoData: "当前轨迹无分数分解数据。",
        registrationProtocolTitle: "Registration",
        registrationProtocolText: "配准任务展示非学习基线的旋转误差、平移误差、Chamfer、耗时和成功率；它不是合成异常标签任务，而是系统几何对齐能力的独立证据。",
        registrationMetricRotation: "旋转误差",
        registrationMetricTranslation: "平移误差",
        registrationMetricChamfer: "Chamfer",
        registrationMetricRuntime: "耗时",
        registrationMetricSuccess: "是否成功",
        registrationSuccess: "成功",
        registrationFailed: "失败/跳过",
        registrationNoMetric: "暂无配准误差字段；请先用 registration_adapter 生成带误差字段的 manifest 和 score rows。",
        dataFlowSequences: "可播放序列",
        dataFlowTracks: "可视化轨迹",
        dataFlowFrames: "帧跨度",
        dataFlowBackgrounds: "背景帧资源",
        dataFlowTaskAudit: "任务审计",
        dataFlowSequenceAudit: "序列数据审计",
        dataFlowPoints: "轨迹点",
        dataFlowScoreCoverage: "分数覆盖",
        dataFlowLabelCoverage: "标签覆盖",
        dataFlowPointCoverage: "融合点覆盖",
        dataFlowRgbCoverage: "RGB 覆盖",
        dataFlowThermalCoverage: "热成像覆盖",
        dataFlowMissingModalities: "缺失模态点",
        dataFlowBackgroundStatus: "背景状态",
        dataFlowAvgOffset: "平均模态偏移",
        dataFlowMaxOffset: "最大模态偏移",
        dataFlowOk: "完整",
        dataFlowPartial: "部分缺失",
        dataFlowMissing: "缺失",
        dataFlowNoTracks: "无轨迹",
        dataFlowNoBackground: "无背景帧",
        provenanceTitle: "运行来源审计",
        provenanceMode: "运行模式",
        provenanceGeneratedAt: "生成时间",
        provenanceDataset: "数据集",
        provenanceDatasetStatus: "数据状态",
        provenanceDatasetFingerprint: "数据指纹",
        provenanceDatasetManifest: "数据 manifest",
        provenanceFinalResults: "最终结果目录",
        provenanceLabelFiles: "标签文件",
        provenanceScoreRoots: "分数搜索目录",
        provenanceFusedJsonl: "融合轨迹",
        provenanceRegistration: "配准 manifest",
        provenanceParameters: "构建参数",
        provenanceMissing: "未提供",
        groupEventTitle: "群体事件聚合",
        groupEventTracks: "涉及轨迹",
        groupEventDuration: "持续帧",
        groupEventCenter: "事件中心",
        noGroupEvents: "当前序列没有可聚合的群体事件。",
        registration3DTitle: "配准 3D 投影诊断",
        registration3DNote: "蓝色为 source，绿色为 reference，红色为 estimated aligned；当前是由 benchmark 误差生成的轻量投影预览。",
        chartNoData: "暂无逐帧曲线数据"
      }});
      Object.assign(translations.en, {{
        tabDataFlow: "Data Flow Audit",
        registrationPairCount: "Registration samples",
        registrationFailedCount: "Failed/skipped",
        flowStepRegistrationPair: "Point-cloud pair",
        flowStepRegistrationFeature: "Match / estimate transform",
        flowStepRegistrationMetric: "Measure error",
        flowStepRegistrationScore: "Registration risk score",
        registrationProtocolTitle: "Registration",
        registrationProtocolText: "Registration shows rotation error, translation error, Chamfer, runtime, and success rate for non-learning baselines. It is a geometry-alignment evidence module, not a synthetic anomaly-label task.",
        registrationMetricRotation: "Rotation error",
        registrationMetricTranslation: "Translation error",
        registrationMetricChamfer: "Chamfer",
        registrationMetricRuntime: "Runtime",
        registrationMetricSuccess: "Success",
        registrationSuccess: "Success",
        registrationFailed: "Failed/skipped",
        registrationNoMetric: "No registration metrics are available. Generate manifest and score rows with registration_adapter first.",
        registrationPlaybackTitle: "Point-cloud registration view",
        registrationPlaybackNote: "Dynamic source, reference, and estimated aligned point clouds",
        registrationEvidenceTitle: "Registration evidence",
        registrationSelectedPair: "Selected point-cloud pair",
        registrationMethod: "Registration method",
        registrationPairStatus: "Pair status",
        registrationSourceCloud: "Source cloud",
        registrationReferenceCloud: "Reference cloud",
        registrationAlignedCloud: "Aligned result",
        registrationNoPointCloud: "No point-cloud coordinates are available for this sample; showing a lightweight diagnostic placeholder.",
        view_registration: "Point-cloud registration view",
        dataFlowSequences: "Playable sequences",
        dataFlowTracks: "Visualized tracks",
        dataFlowFrames: "Frame span",
        dataFlowBackgrounds: "Background assets",
        dataFlowTaskAudit: "Task audit",
        dataFlowSequenceAudit: "Sequence data audit",
        dataFlowPoints: "Trajectory points",
        dataFlowScoreCoverage: "Score coverage",
        dataFlowLabelCoverage: "Label coverage",
        dataFlowPointCoverage: "Fused point coverage",
        dataFlowRgbCoverage: "RGB coverage",
        dataFlowThermalCoverage: "Thermal coverage",
        dataFlowMissingModalities: "Missing modality points",
        dataFlowBackgroundStatus: "Background status",
        dataFlowAvgOffset: "Avg modal offset",
        dataFlowMaxOffset: "Max modal offset",
        dataFlowOk: "Complete",
        dataFlowPartial: "Partial",
        dataFlowMissing: "Missing",
        dataFlowNoTracks: "No tracks",
        dataFlowNoBackground: "No background frames",
        provenanceTitle: "Run Provenance Audit",
        provenanceMode: "Run mode",
        provenanceGeneratedAt: "Generated at",
        provenanceDataset: "Dataset",
        provenanceDatasetStatus: "Dataset status",
        provenanceDatasetFingerprint: "Dataset fingerprint",
        provenanceDatasetManifest: "Dataset manifest",
        provenanceFinalResults: "Final results root",
        provenanceLabelFiles: "Label files",
        provenanceScoreRoots: "Score search roots",
        provenanceFusedJsonl: "Fused trajectories",
        provenanceRegistration: "Registration manifest",
        provenanceParameters: "Build parameters",
        provenanceMissing: "Not provided",
        groupEventTitle: "Group event aggregation",
        groupEventTracks: "Tracks",
        groupEventDuration: "Duration",
        groupEventCenter: "Event center",
        noGroupEvents: "No aggregatable group events in this sequence.",
        registration3DTitle: "Registration 3D projection",
        registration3DNote: "Blue is source, green is reference, red is estimated aligned. This is a lightweight projection preview generated from benchmark error fields.",
        chartNoData: "No frame-level curve data"
      }});
      Object.assign(translations.zh, {{
        eventThresholdLabel: "事件阈值",
        windowEventTitle: "当前窗口事件证据",
        windowEventEmpty: "当前时间窗口没有命中超过阈值的事件证据。",
        windowEventRange: "窗口范围",
        windowEventPeak: "峰值事件分数",
        windowEventReason: "主导原因",
        windowEventFrames: "命中帧数",
        windowEventComponents: "分量证据",
        windowEventSource: "证据来源",
        windowEventSourceModel: "逐帧模型输出",
        windowEventSourceSegment: "事件段/协议段回退"
      }});
      Object.assign(translations.en, {{
        eventThresholdLabel: "Event threshold",
        windowEventTitle: "Current-window event evidence",
        windowEventEmpty: "No event evidence exceeds the threshold in the current time window.",
        windowEventRange: "Window range",
        windowEventPeak: "Peak event score",
        windowEventReason: "Dominant reason",
        windowEventFrames: "Hit frames",
        windowEventComponents: "Component evidence",
        windowEventSource: "Evidence source",
        windowEventSourceModel: "Frame-level model output",
        windowEventSourceSegment: "Event/protocol segment fallback"
      }});
      const backgroundCache = new Map();
      const backgroundFailures = new Set();
      const state = {{
        language: localStorage.getItem("fusiontrack.finalDashboard.language") || "zh",
        task: "{html.escape(initial_task)}",
        method: "{html.escape(initial_method)}",
        caseType: "true_positive",
        sequence: "{html.escape(initial_sequence)}",
        submodule: "route",
        frame: -1,
        playing: false,
        viewMode: "comparison",
        layer: "both",
        heatOpacity: 0.64,
        heatWindow: 36,
        eventThreshold: 0.0,
        playSpeed: 1.0,
        selectedSampleId: "",
        image: null,
        imageKey: null,
        imageStatus: "idle",
        timer: null
      }};
      const rawSavedSpeed = Number(localStorage.getItem("fusiontrack.finalDashboard.playSpeed"));
      if (Number.isFinite(rawSavedSpeed)) {{
        state.playSpeed = Math.max(0.2, Math.min(3, rawSavedSpeed));
      }}

      const anomalyDescriptions = {{
        zh: {{}}, /*
          route_shift: "单条轨迹整体偏移，用来模拟目标偏离常规航线。",
          speed_spike: "目标速度突然增大，轨迹点间距离异常放大。",
          stop_or_slowdown: "目标在中后段突然停住或明显减速。",
          jump: "轨迹中某一帧发生突跳。",
          shape_warp: "轨迹形状被拉伸或压缩。",
          modal_offset: "RGB/thermal 等模态中心发生偏移，模拟多模态不一致。",
          leave_group: "对象偏离群体中心或群体运动区域。",
          against_motion: "对象运动方向与群体趋势相反。",
          neighbor_replacement: "对象轨迹被邻近对象替代。",
          population_change: "群体数量发生变化，当前协议通过复制对象模拟。",
          dispersion_change: "群体离散程度异常变大。",
          split_merge: "群体出现分裂或合并式变化。"
        }},
        */ en: {{
          route_shift: "The whole trajectory is shifted away from its regular route.",
          speed_spike: "Motion speed suddenly increases with enlarged point-to-point distance.",
          stop_or_slowdown: "The object stops or slows down in the later segment.",
          jump: "One frame in the trajectory has an abrupt jump.",
          shape_warp: "The trajectory shape is stretched or compressed.",
          modal_offset: "RGB/thermal centers are shifted to simulate multimodal inconsistency.",
          leave_group: "An object moves away from the group center or motion region.",
          against_motion: "An object moves against the group trend.",
          neighbor_replacement: "An object's trajectory is replaced by a neighboring object.",
          population_change: "The group population changes; this protocol simulates it by copying an object.",
          dispersion_change: "The group's spatial dispersion increases abnormally.",
          split_merge: "The group shows a split or merge pattern."
        }}
      }};
      Object.assign(anomalyDescriptions.zh, {{
        route_shift: "单条轨迹整体偏移，用来模拟目标偏离常规航线。",
        speed_spike: "目标速度突然增大，轨迹点间距离异常放大。",
        stop_or_slowdown: "目标在中后段突然停住或明显减速。",
        jump: "轨迹中某一帧发生突跳。",
        shape_warp: "轨迹形状被拉伸或压缩。",
        modal_offset: "RGB/thermal 等模态中心发生偏移，模拟多模态不一致。",
        leave_group: "对象偏离群体中心或群体运动区域。",
        against_motion: "对象运动方向与群体趋势相反。",
        neighbor_replacement: "对象轨迹被邻近对象替代。",
        population_change: "群体数量发生变化，当前协议通过复制对象模拟。",
        dispersion_change: "群体离散程度异常变大。",
        split_merge: "群体出现分裂或合并式变化。"
      }});

      function taskData() {{ return dashboard.tasks[state.task]; }}
      function fmt(value) {{ return Number(value || 0).toFixed(3); }}
      function pct(value) {{ return `${{Math.round(Number(value || 0) * 100)}}%`; }}
      function finiteNumber(value, fallback = 0) {{
        const number = Number(value);
        return Number.isFinite(number) ? number : fallback;
      }}
      function esc(value) {{ return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}})[ch]); }}
      function methodsForTask(task) {{ return task.leaderboard.map(row => row.method); }}
      function sequences() {{ return Object.keys(playbackData); }}
      function sequenceHasTaskData(sequence, taskName) {{
        const data = playbackData[sequence];
        if (!data) {{
          return false;
        }}
        const stats = (data.stats_by_task || {{}})[taskName];
        if (stats && Number(stats.sequence_sample_count || 0) > 0) {{
          return true;
        }}
        return (data.tracks || []).some(track => {{
          const scores = ((track.task_scores || {{}})[taskName]) || {{}};
          const label = ((track.task_labels || {{}})[taskName]) || {{}};
          const hasScore = Object.values(scores).some(value => Number(value || 0) !== 0);
          const hasLabel = Number(label.label || 0) === 1;
          return hasScore || hasLabel;
        }});
      }}
      function sequencesForTask() {{
        const names = sequences();
        const filtered = names.filter(sequence => sequenceHasTaskData(sequence, state.task));
        const selected = filtered.length ? filtered : names;
        return [...selected].sort((a, b) => compareSequencesForTask(a, b, state.task));
      }}
      function currentPlayback() {{
        const names = sequencesForTask();
        return playbackData[state.sequence] || playbackData[names[0]] || null;
      }}
      function clamp(value, min, max) {{ return Math.max(min, Math.min(max, value)); }}
      function t(key) {{ return (translations[state.language] || translations.zh)[key] || translations.en[key] || key; }}
      function isRegistrationTask(taskName = state.task) {{ return taskName === "registration"; }}
      function trackScoresForTask(track, taskName) {{
        return ((track.task_scores || {{}})[taskName]) || (taskName === "individual" ? track.method_scores : {{}}) || {{}};
      }}
      function trackLabelForTask(track, taskName) {{
        return ((track.task_labels || {{}})[taskName]) || {{
          label: track.label || 0,
          anomaly_type: track.anomaly_type || "normal",
          frame_start: track.frame_start,
          frame_end: track.frame_end
        }};
      }}
      function trackScores(track) {{ return trackScoresForTask(track, state.task); }}
      function trackLabel(track) {{ return trackLabelForTask(track, state.task); }}
      function trackScore(track) {{ return Number((trackScores(track) || {{}})[state.method] || 0); }}
      function trackLabelValue(track) {{ return Number(trackLabel(track).label || 0); }}
      function compareSequencesForTask(a, b, taskName) {{
        const dataA = playbackData[a] || {{}};
        const dataB = playbackData[b] || {{}};
        const statsA = (dataA.stats_by_task || {{}})[taskName] || {{}};
        const statsB = (dataB.stats_by_task || {{}})[taskName] || {{}};
        const positiveA = (dataA.tracks || []).filter(track => Number(trackLabelForTask(track, taskName).label || 0) === 1).length;
        const positiveB = (dataB.tracks || []).filter(track => Number(trackLabelForTask(track, taskName).label || 0) === 1).length;
        if ((positiveA > 0) !== (positiveB > 0)) {{
          return positiveA > 0 ? -1 : 1;
        }}
        const anomaliesA = Number(statsA.sequence_anomaly_count || 0);
        const anomaliesB = Number(statsB.sequence_anomaly_count || 0);
        if (anomaliesA !== anomaliesB) {{
          return anomaliesB - anomaliesA;
        }}
        const maxScoreA = Math.max(...(dataA.tracks || []).map(track => Math.max(0, ...Object.values(trackScoresForTask(track, taskName)).map(value => Number(value || 0)))), 0);
        const maxScoreB = Math.max(...(dataB.tracks || []).map(track => Math.max(0, ...Object.values(trackScoresForTask(track, taskName)).map(value => Number(value || 0)))), 0);
        if (maxScoreA !== maxScoreB) {{
          return maxScoreB - maxScoreA;
        }}
        return sequences().indexOf(a) - sequences().indexOf(b);
      }}

      function anomalyDescription(typeName) {{
        return (anomalyDescriptions[state.language] || anomalyDescriptions.zh)[typeName] || typeName;
      }}

      function anomalyTypeSummary(task) {{
        const byType = new Map();
        for (const row of task.anomaly_type_rows || []) {{
          const typeName = String(row.anomaly_type || "anomaly");
          const current = byType.get(typeName) || 0;
          byType.set(typeName, Math.max(current, Number(row.total_positive || 0)));
        }}
        return [...byType.entries()].sort((a, b) => b[1] - a[1]);
      }}

      function activeTrack(taskSpecific) {{
        const data = currentPlayback();
        const ranked = data ? rankedTracks(data) : [];
        return ranked.find(track => track.sample_id === state.selectedSampleId) || ranked[0] || null;
      }}

      function selectedTrackDecomposition(track) {{
        if (!track) {{
          return null;
        }}
        const decompRows = (track.task_score_decomposition || {{}})[state.task];
        const selected = decompRows && decompRows[state.method];
        if (selected) {{
          return {{
            S_ind: Number(selected.S_ind || 0),
            S_grp: Number(selected.S_grp || 0),
            S_event: Number(selected.S_event || 0),
            S_fused: Number(selected.S_fused || 0),
            alpha: Number(selected.alpha || 0.65),
          }};
        }}
        const fallback = Object.values(decompRows || {{}})[0];
        if (!fallback) {{
          return null;
        }}
        return {{
          S_ind: Number(fallback.S_ind || 0),
          S_grp: Number(fallback.S_grp || 0),
          S_event: Number(fallback.S_event || 0),
          S_fused: Number(fallback.S_fused || 0),
          alpha: Number(fallback.alpha || 0.65),
        }};
      }}

      function selectedTrackScoreComponents(track) {{
        if (!track) {{
          return {{}};
        }}
        const compRows = (track.task_score_components || {{}})[state.task] || {{}};
        return compRows[state.method] || Object.values(compRows || {{}})[0] || {{}};
      }}

      function frameEventScoresForWindow(row, frame = state.frame, window = state.heatWindow, threshold = state.eventThreshold) {{
        const rawRows = Array.isArray(row?.frame_event_scores) ? row.frame_event_scores : [];
        const end = Number.isFinite(Number(frame)) ? Number(frame) : 0;
        const span = Math.max(0, Number(window) || 0);
        const start = end - span;
        const floor = Number(threshold) || 0;
        return rawRows
          .map(item => {{
            const frameValue = Number(item.frame ?? item.frame_id);
            const score = Number(item.score || 0);
            const componentScores = Object.fromEntries(
              Object.entries(item.component_scores || {{}})
                .map(([key, value]) => [key, Number(value)])
                .filter(([, value]) => Number.isFinite(value))
            );
            return {{
              frame: frameValue,
              score,
              dominant_reason: String(item.dominant_reason || item.reason || "event"),
              component_scores: componentScores,
              source: item.source ? String(item.source) : ""
            }};
          }})
          .filter(item => Number.isFinite(item.frame) && Number.isFinite(item.score))
          .filter(item => item.frame >= start && item.frame <= end && item.score > floor)
          .sort((a, b) => b.score - a.score || a.frame - b.frame);
      }}

      function windowEventSummary(row, frame = state.frame, window = state.heatWindow, threshold = state.eventThreshold) {{
        const rawRows = Array.isArray(row?.frame_event_scores) ? row.frame_event_scores : [];
        const end = Number.isFinite(Number(frame)) ? Number(frame) : 0;
        const span = Math.max(0, Number(window) || 0);
        const start = end - span;
        const rows = frameEventScoresForWindow(row, end, span, threshold);
        if (!rows.length) {{
          return {{
            hasFrameScores: rawRows.length > 0,
            rows: [],
            rangeStart: start,
            rangeEnd: end,
            componentScores: {{}}
          }};
        }}
        const peak = rows.reduce((best, item) => item.score > best.score ? item : best, rows[0]);
        const componentScores = {{}};
        rows.forEach(item => {{
          Object.entries(item.component_scores || {{}}).forEach(([key, value]) => {{
            componentScores[key] = Math.max(Number(componentScores[key] || 0), Number(value || 0));
          }});
        }});
        return {{
          hasFrameScores: true,
          rows,
          rangeStart: start,
          rangeEnd: end,
          peakScore: peak.score,
          dominantReason: peak.dominant_reason,
          peakFrame: peak.frame,
          componentScores
        }};
      }}

      function windowSegmentSummary(track, row, frame = state.frame, window = state.heatWindow) {{
        const end = Number.isFinite(Number(frame)) ? Number(frame) : 0;
        const span = Math.max(0, Number(window) || 0);
        const start = end - span;
        const rowSegments = Array.isArray(row?.event_segments) ? row.event_segments : [];
        const taskSegments = (((track || {{}}).task_segments || {{}})[state.task] || []).filter(item => Number(item.label || 0) === 1);
        const sourceSegments = rowSegments.length ? rowSegments : taskSegments;
        const rows = sourceSegments
          .map(item => {{
            const frameStart = Number(item.frame_start ?? item.start ?? 0);
            const frameEnd = Number(item.frame_end ?? item.end ?? frameStart);
            const score = Number(item.score ?? item.event_score ?? 1);
            return {{
              frame_start: frameStart,
              frame_end: frameEnd,
              score,
              dominant_reason: String(item.dominant_reason || item.reason || item.anomaly_type || item.label || "event"),
              component_scores: Object.fromEntries(
                Object.entries(item.component_scores || {{}})
                  .map(([key, value]) => [key, Number(value)])
                  .filter(([, value]) => Number.isFinite(value))
              )
            }};
          }})
          .filter(item => Number.isFinite(item.frame_start) && Number.isFinite(item.frame_end))
          .filter(item => item.frame_end >= start && item.frame_start <= end)
          .sort((a, b) => b.score - a.score || a.frame_start - b.frame_start);
        if (!rows.length) {{
          return {{
            hasSegmentScores: sourceSegments.length > 0,
            rows: [],
            rangeStart: start,
            rangeEnd: end,
            componentScores: {{}}
          }};
        }}
        const peak = rows.reduce((best, item) => item.score > best.score ? item : best, rows[0]);
        const componentScores = {{}};
        rows.forEach(item => {{
          Object.entries(item.component_scores || {{}}).forEach(([key, value]) => {{
            componentScores[key] = Math.max(Number(componentScores[key] || 0), Number(value || 0));
          }});
        }});
        return {{
          hasSegmentScores: true,
          rows,
          rangeStart: start,
          rangeEnd: end,
          peakScore: peak.score,
          dominantReason: peak.dominant_reason,
          peakFrame: `${{peak.frame_start}}-${{peak.frame_end}}`,
          componentScores,
          fallback: true
        }};
      }}

      function renderWindowEventEvidence(track) {{
        const row = selectedTrackScoreComponents(track);
        let summary = windowEventSummary(row);
        let sourceLabel = t("windowEventSourceModel");
        if (!summary.hasFrameScores) {{
          const segmentSummary = windowSegmentSummary(track, row);
          if (segmentSummary.hasSegmentScores || segmentSummary.rows.length) {{
            summary = segmentSummary;
            sourceLabel = t("windowEventSourceSegment");
          }}
        }}
        if (!summary.hasFrameScores && !summary.hasSegmentScores && state.task !== "group") {{
          return "";
        }}
        if (!summary.rows.length) {{
          return `
            <div class="event-card window-event-card">
              <strong>${{t("windowEventTitle")}}</strong>
              <div class="subtle">${{t("windowEventRange")}}: ${{summary.rangeStart}}-${{summary.rangeEnd}}</div>
              <div class="subtle">${{t("windowEventSource")}}: ${{sourceLabel}}</div>
              <div class="subtle">${{t("windowEventEmpty")}}</div>
            </div>
          `;
        }}
        const components = Object.entries(summary.componentScores || {{}})
          .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
          .slice(0, 5);
        const maxComponent = Math.max(1e-6, ...components.map(([, value]) => Number(value || 0)));
        const componentRows = components.length ? components.map(([key, value]) => `
          <div class="decomp-row">
            <span>${{esc(key)}}</span>
            <span class="decomp-track"><span class="decomp-fill" style="width: ${{Math.max(2, Math.min(100, Math.round((Number(value || 0) / maxComponent) * 100)))}}%;"></span></span>
            <strong>${{fmt(value)}}</strong>
          </div>
        `).join("") : `<div class="subtle">${{t("windowEventComponents")}}: -</div>`;
        return `
          <div class="event-card window-event-card">
            <strong>${{t("windowEventTitle")}}</strong>
            <div>${{t("windowEventRange")}}: ${{summary.rangeStart}}-${{summary.rangeEnd}}</div>
            <div>${{t("windowEventSource")}}: <strong>${{sourceLabel}}</strong></div>
            <div>${{t("windowEventPeak")}}: <strong>${{fmt(summary.peakScore)}}</strong> @ ${{t("frame")}} ${{summary.peakFrame}}</div>
            <div>${{t("windowEventReason")}}: <strong>${{esc(summary.dominantReason || "event")}}</strong></div>
            <div>${{t("windowEventFrames")}}: ${{summary.rows.length}}</div>
            <div class="decomp-bar">${{componentRows}}</div>
          </div>
        `;
      }}

      function renderMethodFlow(track) {{
        const task = taskData();
        const decomp = selectedTrackDecomposition(track);
        const labels = {{
          prepare: t("flowStepPrepare"),
          features: t("flowStepFeatures"),
          individual: t("flowStepIndividual"),
          group: t("flowStepGroup"),
          fusion: t("flowStepFusion"),
        }};
        const steps = isRegistrationTask() ? [
          {{ text: t("flowStepRegistrationPair"), state: "done" }},
          {{ text: t("flowStepRegistrationFeature"), state: "done" }},
          {{ text: t("flowStepRegistrationMetric"), state: track ? "active" : "pending" }},
          {{ text: t("flowStepRegistrationScore") + " (" + (track ? fmt(trackScore(track)) : "0.000") + ")", state: decomp ? "done" : "pending" }},
        ] : [
          {{ text: labels.prepare, state: task?.data_root ? "done" : "done" }},
          {{ text: labels.features, state: decomp ? "done" : "pending" }},
          {{ text: labels.individual, state: state.task === "individual" ? "active" : "pending" }},
          {{ text: labels.group, state: state.task === "group" ? "active" : "pending" }},
          {{ text: labels.fusion + " (" + (track ? fmt(trackScore(track)) : "0.000") + ")", state: decomp ? "done" : "pending" }},
        ];
        const statusLabel = {{
          done: t("methodFlowDone"),
          active: t("methodFlowActive"),
          pending: t("methodFlowPending"),
        }};
        methodFlowPanel.innerHTML = steps
          .map(item => `<div class="flow-step ${{item.state}}"><span class="flow-step-text">${{item.text}}</span> <span class="flow-step-state">${{statusLabel[item.state]}}</span></div>`)
          .join("");
        const taskLabel = isRegistrationTask() ? t("taskRegistration") : state.task === "group" ? t("taskGroup") : t("taskIndividual");
        const scoreText = track ? fmt(trackScore(track)) : t("noTrackSelected");
        flowReadout.textContent = `${{t("flowReadoutTask")}}: ${{taskLabel}} / ${{t("flowReadoutMethod")}}: ${{state.method}} / ${{t("flowReadoutScore")}}: ${{scoreText}}`;
      }}

      function submoduleFeatureValue(track, stats, kind) {{
        if (!track) {{
          return 0;
        }}
        const components = selectedTrackScoreComponents(track).component_scores || {{}};
        const candidates = {{
          route: {{
            route_shift: Number(components.route_score || 0),
            shape_shift: Number(components.route_shape_score || 0),
            normal: 0,
          }},
          speed: {{
            speed_spike: Number(components.speed_score || 0),
            stop_or_slowdown: Number(components.speed_slowdown_score || 0),
            jump: Number(components.jump_score || 0),
          }},
          shape: {{
            shape_warp: Number(components.shape_score || 0),
            modal_offset: Number(components.modal_offset_score || 0),
          }},
        }};
        if (kind === "route") {{
          return Math.max(...Object.values(candidates.route || {{ normal: 0 }}), stats.length > 0 ? Math.min(stats.length / 100, 1) : 0);
        }}
        if (kind === "speed") {{
          const base = stats.avgSpeed || 0;
          return Math.max(...Object.values(candidates.speed || {{ normal: 0 }}), Math.min(base / 5, 1));
        }}
        if (kind === "shape") {{
          const ratio = stats.displacement > 0 ? Math.min(stats.length / stats.displacement, 5) / 5 : 0;
          return Math.max(...Object.values(candidates.shape || {{ normal: 0 }}), Math.min(ratio, 1));
        }}
        return 0;
      }}

      function metricValue(row, key) {{
        const value = row ? row[key] : null;
        return value === null || value === undefined || value === "" ? null : Number(value);
      }}

      function projectPoint3d(point, width = 260, height = 150) {{
        const x = Number(point?.[0] || 0);
        const y = Number(point?.[1] || 0);
        const z = Number(point?.[2] || 0);
        return {{
          x: width / 2 + x * 62 + z * 22,
          y: height / 2 - y * 48 + z * 16,
        }};
      }}

      function sampledPointCloud(points, limit = 320) {{
        const clean = (points || []).filter(point => Array.isArray(point) && point.length >= 2);
        if (clean.length <= limit) {{
          return clean;
        }}
        const stride = Math.ceil(clean.length / limit);
        return clean.filter((_, index) => index % stride === 0).slice(0, limit);
      }}

      function registrationPointGroups(track) {{
        const row = selectedTrackScoreComponents(track);
        const preview = row.registration_points || {{}};
        const groups = [
          {{ key: "source", label: t("registrationSourceCloud"), points: sampledPointCloud(preview.source || []), color: "#3b82f6" }},
          {{ key: "reference", label: t("registrationReferenceCloud"), points: sampledPointCloud(preview.reference || []), color: "#22c55e" }},
          {{ key: "aligned", label: t("registrationAlignedCloud"), points: sampledPointCloud(preview.aligned || []), color: "#ef4444" }},
        ];
        if (groups.some(group => group.points.length)) {{
          return groups;
        }}
        const fallback = (track?.points || []).map((point, index) => [
          (Number(point.x || 0) - 480) / 180,
          (Number(point.y || 0) - 306) / 180,
          (index - Math.max(1, (track?.points || []).length) / 2) * 0.12,
        ]);
        if (!fallback.length) {{
          return [];
        }}
        const scoreOffset = Math.min(0.22, Math.max(0.04, trackScore(track) * 0.05));
        return [
          {{ key: "source", label: t("registrationSourceCloud"), points: fallback.map(point => [point[0] - 0.12, point[1], point[2]]), color: "#3b82f6" }},
          {{ key: "reference", label: t("registrationReferenceCloud"), points: fallback.map(point => [point[0] + 0.12, point[1] + 0.08, point[2] + 0.04]), color: "#22c55e" }},
          {{ key: "aligned", label: t("registrationAlignedCloud"), points: fallback.map(point => [point[0] + 0.12 + scoreOffset, point[1] + 0.08, point[2] + 0.04]), color: "#ef4444" }},
        ];
      }}

      function projectRegistrationPoint(point, angle, width, height) {{
        const x = Number(point?.[0] || 0);
        const y = Number(point?.[1] || 0);
        const z = Number(point?.[2] || 0);
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const rotatedX = x * cos + z * sin;
        const rotatedZ = z * cos - x * sin;
        const scale = Math.min(width, height) * 0.30;
        const perspective = 1 / Math.max(0.62, 1 + rotatedZ * 0.18);
        return {{
          x: width / 2 + rotatedX * scale * perspective,
          y: height / 2 - y * scale * perspective + rotatedZ * 18,
          depth: rotatedZ,
          radius: Math.max(2.2, 4.6 * perspective),
        }};
      }}

      function renderRegistrationPointCloud(track) {{
        const row = selectedTrackScoreComponents(track);
        const preview = row.registration_points || {{}};
        const groups = [
          ["source", preview.source || [], "#2563eb"],
          ["reference", preview.reference || [], "#16a34a"],
          ["aligned", preview.aligned || [], "#dc2626"],
        ];
        const hasPoints = groups.some(([, points]) => points.length);
        if (!hasPoints) {{
          return "";
        }}
        const dots = groups.map(([name, points, color]) => points.map(point => {{
          const projected = projectPoint3d(point);
          return `<circle cx="${{projected.x.toFixed(1)}}" cy="${{projected.y.toFixed(1)}}" r="3.5" fill="${{color}}" opacity="0.82"><title>${{name}}</title></circle>`;
        }}).join("")).join("");
        return `
          <div class="registration-preview">
            <strong>${{t("registration3DTitle")}}</strong>
            <svg viewBox="0 0 260 150" role="img" aria-label="${{t("registration3DTitle")}}">
              <line x1="24" y1="126" x2="230" y2="126" stroke="#e2e8f0" />
              <line x1="42" y1="136" x2="42" y2="24" stroke="#e2e8f0" />
              <line x1="42" y1="126" x2="78" y2="92" stroke="#e2e8f0" />
              ${{dots}}
            </svg>
            <div class="legend-row">
              <span><i class="legend-dot" style="background:#2563eb"></i>source</span>
              <span><i class="legend-dot" style="background:#16a34a"></i>reference</span>
              <span><i class="legend-dot" style="background:#dc2626"></i>aligned</span>
            </div>
            <div class="subtle">${{t("registration3DNote")}}</div>
          </div>
        `;
      }}

      function renderRegistrationEvidence(track) {{
        const row = selectedTrackScoreComponents(track);
        const hasMetrics = row && (
          row.rotation_error_deg !== null && row.rotation_error_deg !== undefined ||
          row.translation_error !== null && row.translation_error !== undefined ||
          row.chamfer_distance !== null && row.chamfer_distance !== undefined ||
          row.runtime_sec !== null && row.runtime_sec !== undefined
        );
        if (!hasMetrics) {{
          submodulePanel.textContent = t("registrationNoMetric");
          return;
        }}
        const success = row.success === true || row.success === "true" || row.success === 1 || row.success === "1";
        const skipped = row.skipped === true || row.skipped === "true" || row.skipped === 1 || row.skipped === "1";
        const rows = [
          [t("registrationMetricRotation"), metricValue(row, "rotation_error_deg"), "deg"],
          [t("registrationMetricTranslation"), metricValue(row, "translation_error"), ""],
          [t("registrationMetricChamfer"), metricValue(row, "chamfer_distance"), ""],
          [t("registrationMetricRuntime"), metricValue(row, "runtime_sec"), "s"],
        ];
        submodulePanel.innerHTML = `
          <div class="explain-metrics">
            ${{rows.map(([label, value, unit]) => `
              <div class="explain-metric"><span>${{label}}</span><strong>${{value === null || Number.isNaN(value) ? "-" : `${{fmt(value)}}${{unit ? " " + unit : ""}}`}}</strong></div>
            `).join("")}}
            <div class="explain-metric"><span>${{t("registrationMetricSuccess")}}</span><strong>${{success && !skipped ? t("registrationSuccess") : t("registrationFailed")}}</strong></div>
            <div class="explain-metric"><span>${{t("anomalyScore")}}</span><strong>${{fmt(track ? trackScore(track) : 0)}}</strong></div>
          </div>
          ${{renderRegistrationPointCloud(track)}}
        `;
      }}

      function submoduleCurve(track, kind) {{
        const points = (track?.points || []).slice().sort((a, b) => Number(a.frame) - Number(b.frame));
        if (points.length < 2) {{
          return [];
        }}
        if (kind === "speed") {{
          return points.slice(1).map((point, index) => {{
            const previous = points[index];
            const frameDelta = Math.max(1, Number(point.frame) - Number(previous.frame));
            return {{
              frame: Number(point.frame),
              value: Math.hypot(Number(point.x) - Number(previous.x), Number(point.y) - Number(previous.y)) / frameDelta,
            }};
          }});
        }}
        const first = points[0];
        const last = points[points.length - 1];
        const dx = Number(last.x) - Number(first.x);
        const dy = Number(last.y) - Number(first.y);
        const base = Math.max(1e-6, Math.hypot(dx, dy));
        if (kind === "route") {{
          return points.map(point => {{
            const value = Math.abs(dy * Number(point.x) - dx * Number(point.y) + Number(last.x) * Number(first.y) - Number(last.y) * Number(first.x)) / base;
            return {{ frame: Number(point.frame), value }};
          }});
        }}
        return points.slice(1, -1).map((point, index) => {{
          const previous = points[index];
          const next = points[index + 2];
          const ax = Number(point.x) - Number(previous.x);
          const ay = Number(point.y) - Number(previous.y);
          const bx = Number(next.x) - Number(point.x);
          const by = Number(next.y) - Number(point.y);
          const denom = Math.max(1e-6, Math.hypot(ax, ay) * Math.hypot(bx, by));
          const angle = Math.acos(clamp((ax * bx + ay * by) / denom, -1, 1));
          return {{ frame: Number(point.frame), value: angle }};
        }});
      }}

      function renderMiniChart(curve) {{
        if (!curve.length) {{
          return `<div class="subtle">${{t("chartNoData")}}</div>`;
        }}
        const width = 320;
        const height = 96;
        const minFrame = Math.min(...curve.map(item => item.frame));
        const maxFrame = Math.max(...curve.map(item => item.frame));
        const maxValue = Math.max(1e-6, ...curve.map(item => Number(item.value || 0)));
        const points = curve.map(item => {{
          const x = 10 + ((item.frame - minFrame) / Math.max(1, maxFrame - minFrame)) * (width - 20);
          const y = height - 12 - (Number(item.value || 0) / maxValue) * (height - 24);
          return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
        }}).join(" ");
        return `
          <svg class="mini-chart" viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
            <polyline points="${{points}}" fill="none" stroke="#0f766e" stroke-width="2.2" />
            <line x1="10" y1="${{height - 12}}" x2="${{width - 10}}" y2="${{height - 12}}" stroke="#e2e8f0" />
          </svg>
        `;
      }}

      function renderSubmoduleTrack(track) {{
        if (isRegistrationTask()) {{
          submoduleSwitch.hidden = true;
          renderRegistrationEvidence(track);
          return;
        }}
        submoduleSwitch.hidden = false;
        const stats = trajectoryStats(track || {{}});  
        const score = submoduleFeatureValue(track, stats, state.submodule);
        const label = track ? trackLabel(track) : {{ anomaly_type: "normal" }};
        const source = selectedTrackScoreComponents(track).used_sources || "";
        const kindLabel = {{
          route: t("submoduleRoute"),
          speed: t("submoduleSpeed"),
          shape: t("submoduleShape"),
        }};
        const curve = submoduleCurve(track, state.submodule);
        submodulePanel.innerHTML = `
          <div class="explain-metric">
            <span>${{t("submodulePrefix")}}</span>
            <strong>${{kindLabel[state.submodule] || state.submodule}}</strong>
          </div>
          <div class="explain-metric">
            <span>${{t("anomalyTypeLabel")}}</span>
            <strong>${{esc(label.anomaly_type || "normal")}}</strong>
          </div>
          <div class="explain-reason">${{source ? `Used source: ${{source}}` : t("noSubmoduleData")}}</div>
          <div class="explain-reason">Proxy score: ${{fmt(score)}} (dynamic evidence)</div>
          ${{renderMiniChart(curve)}}
        `;
      }}

      function renderCompositionBars(track) {{
        if (isRegistrationTask()) {{
          const row = selectedTrackScoreComponents(track);
          const rows = [
            [t("registrationMetricRotation"), metricValue(row, "rotation_error_deg")],
            [t("registrationMetricTranslation"), metricValue(row, "translation_error")],
            [t("registrationMetricChamfer"), metricValue(row, "chamfer_distance")],
            [t("registrationMetricRuntime"), metricValue(row, "runtime_sec")],
            [t("compSFused"), track ? trackScore(track) : 0],
          ].filter(([, value]) => value !== null && !Number.isNaN(value));
          if (!rows.length) {{
            scoreCompositionPanel.textContent = t("compNoData");
            return;
          }}
          const max = Math.max(1e-6, ...rows.map(([, value]) => Math.abs(Number(value || 0))));
          scoreCompositionPanel.innerHTML = rows
            .map(([name, value]) => `
              <div class="decomp-row">
                <span>${{name}}</span>
                <span class="decomp-track"><span class="decomp-fill" style="width: ${{Math.max(2, Math.min(100, Math.round((Math.abs(Number(value || 0)) / max) * 100)))}}%;"></span></span>
                <strong>${{fmt(value)}}</strong>
              </div>
            `).join("");
          return;
        }}
        const decomp = selectedTrackDecomposition(track);
        if (!decomp) {{
          scoreCompositionPanel.textContent = t("compNoData");
          return;
        }}
        const rows = [
          ["S_ind", decomp.S_ind, t("compSInd")],
          ["S_grp", decomp.S_grp, t("compSGrp")],
          ["S_event", decomp.S_event, t("compSEvent")],
          ["S_fused", decomp.S_fused, t("compSFused")],
        ];
        const max = Math.max(1e-6, ...rows.map(([, value]) => Number(value || 0)));
        scoreCompositionPanel.innerHTML = rows
          .map(([, value, name]) => `
            <div class="decomp-row">
              <span>${{name}}</span>
              <span class="decomp-track"><span class="decomp-fill" style="width: ${{Math.max(0, Math.min(100, Math.round((Number(value || 0) / max) * 100)))}}%;"></span></span>
              <strong>${{fmt(value)}}</strong>
            </div>
          `).join("");
      }}

      function parseRangeToPixels(frameStart, frameEnd, rangeStart, rangeEnd) {{
        if (frameEnd <= frameStart || rangeEnd <= rangeStart) {{
          return {{ left: 0, width: 0 }};
        }}
        const left = Math.max(0, ((Number(frameStart) - rangeStart) / (rangeEnd - rangeStart)) * 100);
        const right = Math.min(100, ((Number(frameEnd) - rangeStart) / (rangeEnd - rangeStart)) * 100);
        return {{
          left: Math.max(0, left),
          width: Math.max(0, Math.min(100, right - left)),
        }};
      }}

      function aggregateGroupEvents(data) {{
        if (state.task !== "group" || !data) {{
          return [];
        }}
        const grouped = new Map();
        for (const track of data.tracks || []) {{
          const segments = (((track || {{}}).task_segments || {{}})[state.task] || []).filter(item => Number(item.label || 0) === 1);
          for (const segment of segments) {{
            const typeName = String(segment.anomaly_type || "group_event");
            const start = Number(segment.frame_start || 0);
            const end = Number(segment.frame_end || start);
            const bucketStart = Math.round(start / 8) * 8;
            const bucketEnd = Math.round(end / 8) * 8;
            const key = `${{typeName}}:${{bucketStart}}:${{bucketEnd}}`;
            const current = grouped.get(key) || {{
              anomaly_type: typeName,
              frame_start: start,
              frame_end: end,
              tracks: new Set(),
              points: [],
            }};
            current.frame_start = Math.min(current.frame_start, start);
            current.frame_end = Math.max(current.frame_end, end);
            current.tracks.add(track.track_id);
            for (const point of track.points || []) {{
              const frame = Number(point.frame);
              if (frame >= start && frame <= end) {{
                current.points.push(point);
              }}
            }}
            grouped.set(key, current);
          }}
        }}
        return [...grouped.values()].map(item => {{
          const center = item.points.length ? {{
            x: item.points.reduce((sum, point) => sum + Number(point.x), 0) / item.points.length,
            y: item.points.reduce((sum, point) => sum + Number(point.y), 0) / item.points.length,
          }} : null;
          return {{
            anomaly_type: item.anomaly_type,
            frame_start: item.frame_start,
            frame_end: item.frame_end,
            track_count: item.tracks.size,
            center,
          }};
        }}).sort((a, b) => b.track_count - a.track_count || a.frame_start - b.frame_start).slice(0, 6);
      }}

      function renderGroupEventCards(data) {{
        const events = aggregateGroupEvents(data);
        if (!events.length) {{
          return `<div class="subtle">${{t("noGroupEvents")}}</div>`;
        }}
        return `
          <div class="event-card-grid">
            ${{events.map(event => `
              <div class="event-card">
                <strong>${{esc(event.anomaly_type)}}</strong>
                <div>${{event.frame_start}}-${{event.frame_end}} / ${{t("groupEventDuration")}} ${{Math.max(0, event.frame_end - event.frame_start)}}</div>
                <div>${{t("groupEventTracks")}}: ${{event.track_count}}</div>
                <div>${{t("groupEventCenter")}}: ${{event.center ? `${{fmt(event.center.x)}}, ${{fmt(event.center.y)}}` : "-"}}</div>
              </div>
            `).join("")}}
          </div>
        `;
      }}

      function renderEventTimeline(track) {{
        if (isRegistrationTask()) {{
          eventTimelinePanel.textContent = t("registrationProtocolText");
          return;
        }}
        const data = currentPlayback();
        if (!data) {{
          eventTimelinePanel.textContent = t("noTimeline");
          return;
        }}
        const range = data.frame_range || [0, 0];
        const totalStart = Number(range[0] || 0);
        const totalEnd = Number(range[1] || totalStart);
        const duration = Math.max(1, totalEnd - totalStart);
        const label = track ? trackLabel(track) : {{ label: 0 }};
        const gtSegments = (((track || {{}}).task_segments || {{}})[state.task] || []).filter(item => Number(item.label || 0) === 1);
        const predSegments = [];
        if (track) {{
          const row = selectedTrackScoreComponents(track);
          const used = Number(row.event_score || 0);
          const gtStart = Number(label.frame_start || totalStart);
          const gtEnd = Number(label.frame_end || totalEnd);
          const eventFrames = Array.isArray(row.event_segments) ? row.event_segments : [];
          const derivedEventFrames = eventFrames.length ? eventFrames : eventSegmentsFromFrameScores(row.frame_event_scores);
          if (used > 0 && derivedEventFrames.length) {{
            derivedEventFrames.forEach(item => {{
              const start = Number(item.frame_start || gtStart);
              const end = Number(item.frame_end || gtEnd);
              if (end > start) {{
                predSegments.push({{
                  frame_start: start,
                  frame_end: end,
                  label: "event",
                }});
              }}
            }});
          }} else if (used > 0 && gtEnd > gtStart) {{
            predSegments.push({{
              frame_start: gtStart,
              frame_end: gtEnd,
              label: "event",
            }});
          }} else if (label.label === 1 && gtEnd > gtStart) {{
            predSegments.push({{
              frame_start: gtStart,
              frame_end: gtEnd,
              label: "event",
            }});
          }}
        }}
        const gtRow = gtSegments.map(item => {{
          const rangeValue = parseRangeToPixels(item.frame_start, item.frame_end, totalStart, totalEnd);
          return `<span class="timeline-segment gt" style="left:${{rangeValue.left}}%;width:${{rangeValue.width}}%"></span>`;
        }}).join("");
        const predRow = predSegments.map(item => {{
          const rangeValue = parseRangeToPixels(item.frame_start, item.frame_end, totalStart, totalEnd);
          return `<span class="timeline-segment pred" style="left:${{rangeValue.left}}%;width:${{rangeValue.width}}%"></span>`;
        }}).join("");
        eventTimelinePanel.innerHTML = `
          <div class="timeline">
            <div class="timeline-item">
              <div class="timeline-label">${{t("timelineGt")}} (${{
                gtSegments.length
              }})</div>
              <div class="timeline-strip">${{gtRow || `<span class="subtle">${{t("noTimeline")}}</span>`}}</div>
            </div>
            <div class="timeline-item">
              <div class="timeline-label">${{t("timelinePred")}} (${{
                predSegments.length
              }})</div>
              <div class="timeline-strip">${{predRow || `<span class="subtle">${{t("noTimeline")}}</span>`}}</div>
            </div>
          </div>
          ${{state.task === "group" ? `<h4>${{t("groupEventTitle")}}</h4>${{renderGroupEventCards(data)}}` : ""}}
        `;
      }}

      function eventSegmentsFromFrameScores(frameScores) {{
        if (!Array.isArray(frameScores)) {{
          return [];
        }}
        const segments = [];
        let current = null;
        frameScores
          .map(item => ({{
            frame: Number(item.frame ?? item.frame_id),
            score: Number(item.score || 0)
          }}))
          .filter(item => Number.isFinite(item.frame))
          .sort((a, b) => a.frame - b.frame)
          .forEach(item => {{
            if (item.score <= 0) {{
              if (current) {{
                segments.push(current);
                current = null;
              }}
              return;
            }}
            if (!current) {{
              current = {{ frame_start: item.frame, frame_end: item.frame, label: "event" }};
              return;
            }}
            if (item.frame <= current.frame_end + 1) {{
              current.frame_end = item.frame;
            }} else {{
              segments.push(current);
              current = {{ frame_start: item.frame, frame_end: item.frame, label: "event" }};
            }}
          }});
        if (current) {{
          segments.push(current);
        }}
        return segments;
      }}

      function renderProtocolOverview() {{
        const labels = {{ individual: t("taskIndividual"), group: t("taskGroup"), registration: t("taskRegistration") }};
        for (const [taskName, target] of [["individual", individualProtocol], ["group", groupProtocol], ["registration", registrationProtocol]]) {{
          const task = dashboard.tasks[taskName];
          if (!task || !target) {{
            if (target) {{
              target.hidden = true;
            }}
            continue;
          }}
          target.hidden = false;
          if (isRegistrationTask(taskName)) {{
            const rows = task.leaderboard || [];
            const best = rows[0] || {{}};
            target.innerHTML = `
              <h3>${{t("registrationProtocolTitle")}}</h3>
              <div class="subtle">${{t("registrationProtocolText")}}</div>
              <div class="type-cloud">
                <span class="type-chip">${{t("cardMethods")}} <strong>${{rows.length}}</strong></span>
                <span class="type-chip">${{t("registrationSuccessRate")}} <strong>${{fmt(best.success_rate || best.auroc || 0)}}</strong></span>
                <span class="type-chip">${{t("registrationMetricChamfer")}} <strong>${{fmt(best.chamfer_distance_mean || 0)}}</strong></span>
              </div>
            `;
            continue;
          }}
          const typeItems = anomalyTypeSummary(task).map(([typeName, count]) => `
            <span class="type-chip" title="${{esc(anomalyDescription(typeName))}}">${{esc(typeName)}} <strong>${{count}}</strong></span>
          `).join("");
          target.innerHTML = `
            <h3>${{esc(labels[taskName] || taskName)}}</h3>
            <div class="subtle">${{t("cardLabels")}} ${{task.num_labels}} / ${{t("cardPositives")}} ${{task.num_positive}}</div>
            <div class="type-cloud">${{typeItems}}</div>
          `;
        }}
      }}

      function renderHelp() {{
        const individualTypes = anomalyTypeSummary(dashboard.tasks.individual || {{}}).map(([typeName]) => `<li><strong>${{esc(typeName)}}</strong>: ${{esc(anomalyDescription(typeName))}}</li>`).join("");
        const groupTypes = anomalyTypeSummary(dashboard.tasks.group || {{}}).map(([typeName]) => `<li><strong>${{esc(typeName)}}</strong>: ${{esc(anomalyDescription(typeName))}}</li>`).join("");
        if (state.language === "zh") {{
          helpBody.innerHTML = `
            <section class="help-section">
              <h3>页面内容</h3>
              <p>顶部指标卡展示当前任务的方法数、标签数、异常数和当前方法指标。中间动态可视化提供原视频、热力图、轨迹、热力+轨迹四画面对比，也可以切换成单画面模式。</p>
            </section>
            <section class="help-section">
              <h3>异常来源</h3>
              <p>Individual 和 Group 任务使用 synthetic anomaly injection protocol。VT-Tiny-MOT 本身没有这些异常标签，我们在清洗后的正常轨迹或群体窗口上按规则注入异常。背景帧仍是原始视频，异常主要体现在轨迹坐标、多模态中心偏移和群体关系变化上。</p>
            </section>
            <section class="help-section">
              <h3>Registration 任务</h3>
              <p>Registration 是配准模块的独立诊断入口，用来展示非学习配准基线的旋转误差、平移误差、Chamfer、耗时和成功率。它不是合成异常标签任务，而是论文系统中几何对齐能力的证据层。</p>
            </section>
            <section class="help-section">
              <h3>Individual 异常类型</h3>
              <ul>${{individualTypes}}</ul>
            </section>
            <section class="help-section">
              <h3>Group 异常类型</h3>
              <ul>${{groupTypes}}</ul>
            </section>
            <section class="help-section">
              <h3>如何解读</h3>
              <p>热力越强表示当前方法给出的异常分数越高；红色轨迹表示当前任务标签下的正样本。下方解释面板会显示选中轨迹的分数、标签、帧段、运动长度、邻近关系或配准误差。</p>
            </section>
          `;
          return;
        }}
        helpBody.innerHTML = state.language === "zh" ? `
          <section class="help-section">
            <h3>页面内容</h3>
            <p>顶部指标卡展示当前任务的方法数、标签总数、异常正样本数和当前方法 AUROC。中间动态可视化提供原视频、热力图、轨迹、热力+轨迹四画面对比，也可以切换成单画面模式。</p>
          </section>
          <section class="help-section">
            <h3>异常来源</h3>
            <p>当前实验使用 synthetic anomaly injection protocol：原始 VT-Tiny-MOT 本身没有这些异常标签，我们在清洗后的正常轨迹或群体窗口上按规则注入异常。背景帧仍是原始视频，异常主要体现在轨迹坐标、多模态中心偏移和群体关系变化上。</p>
          </section>
          <section class="help-section">
            <h3>Individual 异常类型</h3>
            <ul>${{individualTypes}}</ul>
          </section>
          <section class="help-section">
            <h3>Group 异常类型</h3>
            <ul>${{groupTypes}}</ul>
          </section>
          <section class="help-section">
            <h3>如何解读</h3>
            <p>热力越强表示当前方法给出的异常分数越高；红色轨迹表示该任务标签下的正样本。下方“为什么判为异常”会显示选中轨迹的分数、标签、帧段、运动长度和邻近关系。</p>
          </section>
        ` : `
          <section class="help-section">
            <h3>Page contents</h3>
            <p>The top cards show method count, total labels, positive anomalies, and AUROC for the selected method. The playback area compares original video, heatmap, tracks, and heat+tracks, with a single-view mode for focused inspection.</p>
          </section>
          <section class="help-section">
            <h3>Anomaly source</h3>
            <p>The experiment uses a synthetic anomaly injection protocol. VT-Tiny-MOT does not provide these anomaly labels; rules are injected on cleaned normal trajectories or group windows. Original frames remain unchanged, while anomalies appear in trajectories, multimodal offsets, and group relations.</p>
          </section>
          <section class="help-section">
            <h3>Individual anomaly types</h3>
            <ul>${{individualTypes}}</ul>
          </section>
          <section class="help-section">
            <h3>Group anomaly types</h3>
            <ul>${{groupTypes}}</ul>
          </section>
          <section class="help-section">
            <h3>How to read it</h3>
            <p>Stronger heat means a higher anomaly score from the selected method. Red tracks are positive labels for the current task. The explanation panel shows score, label, frame range, motion length, and neighborhood evidence for the selected track.</p>
          </section>
        `;
      }}

      function methodStatus(row) {{
        if (row.is_our_method) {{
          return t("proposedStatus");
        }}
        return t("baselineStatus");
      }}

      function renderMethodStatus() {{
        const task = taskData();
        const rows = task.leaderboard || [];
        const proposed = rows.filter(row => row.is_our_method).length;
        const baselines = rows.length - proposed;
        methodSummary.innerHTML = [
          [t("cardMethods"), rows.length],
          [t("proposedStatus"), proposed],
          [t("baselineStatus"), baselines]
        ].map(([label, value]) => `<div class="method-summary-item"><span>${{label}}</span><strong>${{value}}</strong></div>`).join("");
        if (isRegistrationTask()) {{
          methodStatusTable.innerHTML = `
            <thead><tr><th>${{t("methodHeader")}}</th><th>${{t("familyHeader")}}</th><th class="metric">${{t("registrationSuccessRate")}}</th><th class="metric">${{t("registrationMetricRotation")}}</th><th class="metric">${{t("registrationMetricTranslation")}}</th><th class="metric">${{t("registrationMetricChamfer")}}</th><th class="metric">${{t("registrationMetricRuntime")}}</th></tr></thead>
            <tbody>${{rows.map(row => `
              <tr>
                <td><strong>${{esc(row.method)}}</strong></td>
                <td>${{esc(row.method_family || "")}}</td>
                <td class="metric">${{fmt(row.success_rate || row.auroc || 0)}}</td>
                <td class="metric">${{fmt(row.rotation_error_deg_mean || 0)}}</td>
                <td class="metric">${{fmt(row.translation_error_mean || 0)}}</td>
                <td class="metric">${{fmt(row.chamfer_distance_mean || 0)}}</td>
                <td class="metric">${{fmt(row.runtime_sec_mean || 0)}}</td>
              </tr>
            `).join("")}}</tbody>
          `;
          return;
        }}
        methodStatusTable.innerHTML = `
          <thead><tr><th>${{t("methodHeader")}}</th><th>${{t("sourceHeader")}}</th><th>${{t("familyHeader")}}</th><th>${{t("learningHeader")}}</th><th>${{t("statusHeader")}}</th><th class="metric">AUROC</th></tr></thead>
          <tbody>${{rows.map(row => `
            <tr>
              <td><strong>${{esc(row.method)}}</strong></td>
              <td>${{esc(row.owner || "")}}</td>
              <td>${{esc(row.method_family || "")}}</td>
              <td>${{esc(row.learning_type || "")}}</td>
              <td>${{methodStatus(row)}} · ${{t("integratedStatus")}}</td>
              <td class="metric">${{fmt(row.auroc)}}</td>
            </tr>
          `).join("")}}</tbody>
        `;
      }}

      function rankedTracks(data) {{
        return [...(data?.tracks || [])].sort((a, b) => trackScore(b) - trackScore(a)).slice(0, 80);
      }}

      function ensureSelectedTrack(data, ranked) {{
        if (!data || !ranked.length) {{
          state.selectedSampleId = "";
          return null;
        }}
        const current = ranked.find(track => track.sample_id === state.selectedSampleId);
        if (current) {{
          return current;
        }}
        const labeled = ranked.find(track => trackLabelValue(track) === 1);
        const selected = labeled || ranked[0];
        state.selectedSampleId = selected.sample_id;
        return selected;
      }}

      function selectedTrack(data) {{
        return (data?.tracks || []).find(track => track.sample_id === state.selectedSampleId) || null;
      }}

      function pointAtFrame(track, frame) {{
        const points = activePoints(track, frame);
        return points[points.length - 1] || null;
      }}

      function trajectoryStats(track) {{
        const points = track.points || [];
        let length = 0;
        for (let index = 1; index < points.length; index += 1) {{
          const previous = points[index - 1];
          const current = points[index];
          length += Math.hypot(Number(current.x) - Number(previous.x), Number(current.y) - Number(previous.y));
        }}
        const first = points[0] || {{ x: 0, y: 0, frame: 0 }};
        const last = points[points.length - 1] || first;
        const frameSpan = Math.max(1, Number(last.frame || 0) - Number(first.frame || 0));
        return {{
          length,
          avgSpeed: length / frameSpan,
          displacement: Math.hypot(Number(last.x) - Number(first.x), Number(last.y) - Number(first.y))
        }};
      }}

      function groupFrameStats(data, selected) {{
        const frame = Number(state.frame);
        const points = (data?.tracks || []).map(track => ({{ track, point: pointAtFrame(track, frame) }})).filter(item => item.point);
        if (!points.length) {{
          return {{ neighbors: [], radius: 0, centroid: null }};
        }}
        const centroid = {{
          x: points.reduce((sum, item) => sum + Number(item.point.x), 0) / points.length,
          y: points.reduce((sum, item) => sum + Number(item.point.y), 0) / points.length
        }};
        const radius = points.reduce((sum, item) => sum + Math.hypot(Number(item.point.x) - centroid.x, Number(item.point.y) - centroid.y), 0) / points.length;
        const selectedPoint = selected ? pointAtFrame(selected, frame) : null;
        const neighbors = selectedPoint
          ? points
              .filter(item => item.track.sample_id !== selected.sample_id)
              .map(item => ({{ track: item.track, distance: Math.hypot(Number(item.point.x) - selectedPoint.x, Number(item.point.y) - selectedPoint.y) }}))
              .sort((a, b) => a.distance - b.distance)
              .slice(0, 3)
          : [];
        return {{ neighbors, radius, centroid }};
      }}

      function explanationReason(track, stats, groupStats) {{
        if (isRegistrationTask()) {{
          const row = selectedTrackScoreComponents(track);
          const rotation = metricValue(row, "rotation_error_deg");
          const translation = metricValue(row, "translation_error");
          const chamfer = metricValue(row, "chamfer_distance");
          const runtime = metricValue(row, "runtime_sec");
          const success = row.success === true || row.success === "true" || row.success === 1 || row.success === "1";
          return state.language === "zh"
            ? `配准证据：旋转误差 ${{rotation === null ? "-" : fmt(rotation)}}，平移误差 ${{translation === null ? "-" : fmt(translation)}}，Chamfer ${{chamfer === null ? "-" : fmt(chamfer)}}，耗时 ${{runtime === null ? "-" : fmt(runtime)}}s，状态 ${{success ? t("registrationSuccess") : t("registrationFailed")}}。`
            : `Registration evidence: rotation error ${{rotation === null ? "-" : fmt(rotation)}}, translation error ${{translation === null ? "-" : fmt(translation)}}, Chamfer ${{chamfer === null ? "-" : fmt(chamfer)}}, runtime ${{runtime === null ? "-" : fmt(runtime)}}s, status ${{success ? t("registrationSuccess") : t("registrationFailed")}}.`;
        }}
        const label = trackLabel(track);
        const typeName = String(label.anomaly_type || "normal");
        const base = typeName !== "normal" ? anomalyDescription(typeName) : (state.language === "zh" ? "当前轨迹未被协议标为正样本，若分数较高则属于候选误报或模型认为的高风险轨迹。" : "This track is not labeled positive by the protocol; a high score means a candidate false positive or high-risk model output.");
        const motion = state.language === "zh"
          ? `运动证据：轨迹长度 ${{fmt(stats.length)}}，平均速度 ${{fmt(stats.avgSpeed)}}，首尾位移 ${{fmt(stats.displacement)}}。`
          : `Motion evidence: path length ${{fmt(stats.length)}}, mean speed ${{fmt(stats.avgSpeed)}}, displacement ${{fmt(stats.displacement)}}.`;
        const group = state.task === "group"
          ? (state.language === "zh" ? `群体证据：当前帧邻近对象 ${{groupStats.neighbors.length}} 个，群体半径 ${{fmt(groupStats.radius)}}。` : `Group evidence: ${{groupStats.neighbors.length}} nearby objects at the current frame, group radius ${{fmt(groupStats.radius)}}.`)
          : "";
        return [base, motion, group].filter(Boolean).join(" ");
      }}

      function renderTrackInsights(ranked) {{
        const data = currentPlayback();
        const track = ensureSelectedTrack(data, ranked);
        renderTrackRankList(ranked);
        if (!data || !track) {{
          explanationPanel.textContent = t("noTrackSelected");
          groupInsightPanel.textContent = isRegistrationTask() ? t("registrationNoMetric") : state.task === "group" ? t("noTrackSelected") : (state.language === "zh" ? "切换到 Group 任务可查看群体中心、半径和邻近关系。" : "Switch to Group to inspect centroid, radius, and neighborhood relations.");
          return;
        }}
        const label = trackLabel(track);
        const stats = trajectoryStats(track);
        const groupStats = groupFrameStats(data, track);
        const windowEventEvidence = renderWindowEventEvidence(track);
        explanationPanel.innerHTML = `
          <div><strong>${{t("selectedTrack")}}</strong> ${{esc(track.sequence)}} / ${{esc(track.track_id)}} <span class="badge ${{trackLabelValue(track) === 1 ? "our" : "baseline"}}">${{trackLabelValue(track) === 1 ? t("syntheticLabel") : t("normalLabel")}}</span></div>
          <div class="explain-metrics">
            <div class="explain-metric"><span>${{t("anomalyScore")}}</span><strong>${{fmt(trackScore(track))}}</strong></div>
            <div class="explain-metric"><span>${{t("anomalyTypeLabel")}}</span><strong>${{esc(label.anomaly_type || "normal")}}</strong></div>
            <div class="explain-metric"><span>${{t("frameRangeLabel")}}</span><strong>${{label.frame_start ?? "-"}}-${{label.frame_end ?? "-"}}</strong></div>
            <div class="explain-metric"><span>${{t("motionLengthLabel")}}</span><strong>${{fmt(stats.length)}}</strong></div>
            <div class="explain-metric"><span>${{t("avgSpeedLabel")}}</span><strong>${{fmt(stats.avgSpeed)}}</strong></div>
            <div class="explain-metric"><span>${{t("displacementLabel")}}</span><strong>${{fmt(stats.displacement)}}</strong></div>
          </div>
          <div class="explain-reason">${{esc(explanationReason(track, stats, groupStats))}}</div>
          ${{windowEventEvidence}}
        `;
        if (isRegistrationTask()) {{
          renderRegistrationEvidence(track);
          groupInsightPanel.innerHTML = submodulePanel.innerHTML;
        }} else if (state.task === "group") {{
          groupInsightPanel.innerHTML = `
            <div class="explain-metrics">
              <div class="explain-metric"><span>${{t("currentNeighborsLabel")}}</span><strong>${{groupStats.neighbors.length}}</strong></div>
              <div class="explain-metric"><span>${{t("centroidRadiusLabel")}}</span><strong>${{fmt(groupStats.radius)}}</strong></div>
            </div>
            <div class="explain-reason">${{groupStats.neighbors.map(item => `${{esc(item.track.track_id)}} · ${{fmt(item.distance)}}`).join("<br>") || (state.language === "zh" ? "当前帧没有可计算邻近对象。" : "No nearby objects are available at the current frame.")}}</div>
          `;
        }} else {{
          groupInsightPanel.textContent = state.language === "zh" ? "Individual 任务主要解释单轨迹运动和多模态偏移；切换到 Group 后会显示群体中心、半径和邻近对象。" : "Individual focuses on single-track motion and multimodal offsets. Switch to Group for centroid, radius, and neighbors.";
        }}
        renderMethodFlow(track);
        renderSubmoduleTrack(track);
        renderCompositionBars(track);
        renderEventTimeline(track);
      }}

      function renderTrackRankList(ranked) {{
        trackRankList.innerHTML = ranked.slice(0, 8).map((track, index) => {{
          const label = trackLabel(track);
          return `
            <button type="button" class="track-rank-item${{track.sample_id === state.selectedSampleId ? " active" : ""}}" data-sample="${{esc(track.sample_id)}}">
              <span><strong>#${{index + 1}} ${{esc(track.sequence)}} / ${{esc(track.track_id)}}</strong><br><span class="subtle">${{esc(label.anomaly_type || "normal")}}</span></span>
              <span class="score">${{fmt(trackScore(track))}}</span>
            </button>
          `;
        }}).join("");
        Array.from(trackRankList.querySelectorAll("[data-sample]")).forEach(button => {{
          button.addEventListener("click", () => {{
            state.selectedSampleId = button.dataset.sample || "";
            drawPlayback();
          }});
        }});
      }}

      function setTaskOptions() {{
        const labels = {{ individual: t("taskIndividual"), group: t("taskGroup"), registration: t("taskRegistration") }};
        taskSelector.innerHTML = Object.keys(dashboard.tasks).map(task =>
          `<option value="${{esc(task)}}"${{task === state.task ? " selected" : ""}}>${{esc(labels[task] || task)}}</option>`
        ).join("");
      }}

      function applyLanguage(language) {{
        state.language = translations[language] ? language : "zh";
        languageSelector.value = state.language;
        document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
        document.title = t("documentTitle");
        localStorage.setItem("fusiontrack.finalDashboard.language", state.language);
        document.querySelectorAll("[data-i18n]").forEach(element => {{
          element.textContent = t(element.dataset.i18n);
        }});
        setTaskOptions();
      }}

      function setMethodOptions() {{
        const task = taskData();
        methodSelector.innerHTML = methodsForTask(task).map(method =>
          `<option value="${{esc(method)}}"${{method === state.method ? " selected" : ""}}>${{esc(method)}}</option>`
        ).join("");
      }}

      function setSequenceOptions() {{
        const names = sequencesForTask();
        if (!names.length) {{
          sequenceSelector.innerHTML = "";
          sequenceSelector.disabled = true;
          playToggle.disabled = true;
          frameSlider.disabled = true;
          return;
        }}
        if (!state.sequence || !names.includes(state.sequence)) {{
          state.sequence = names[0];
          state.image = null;
          state.imageKey = null;
          state.imageStatus = "idle";
        }}
        sequenceSelector.disabled = false;
        playToggle.disabled = false;
        frameSlider.disabled = false;
        sequenceSelector.innerHTML = names.map(sequence =>
          `<option value="${{esc(sequence)}}"${{sequence === state.sequence ? " selected" : ""}}>${{esc(sequence)}}</option>`
        ).join("");
      }}

      function renderCards() {{
        const task = taskData();
        const current = task.methods[state.method];
        const metrics = current.metrics;
        const metricLabel = isRegistrationTask() ? t("registrationSuccessRate") : t("cardAuroc");
        const metricValue = isRegistrationTask() ? (metrics.success_rate ?? metrics.auroc ?? 0) : metrics.auroc;
        const labelCount = isRegistrationTask() ? Math.round(Number(metrics.num_score_rows || 0)) : task.num_labels;
        const positiveCount = isRegistrationTask() ? Math.round(Number(metrics.num_failed_pairs || 0)) : task.num_positive;
        cards.innerHTML = [
          [t("cardMethods"), Object.keys(task.methods).length],
          [isRegistrationTask() ? t("registrationPairCount") : t("cardLabels"), labelCount],
          [isRegistrationTask() ? t("registrationFailedCount") : t("cardPositives"), positiveCount],
          [metricLabel, fmt(metricValue)]
        ].map(([label, value]) => `<div class="card"><div>${{label}}</div><div class="value">${{value}}</div></div>`).join("");
      }}

      function renderProvenanceAudit() {{
        const provenance = dashboard.provenance || {{}};
        const dataset = provenance.dataset || {{}};
        const inputs = provenance.inputs || {{}};
        const parameters = provenance.parameters || {{}};
        const missing = t("provenanceMissing");
        const datasetText = [dataset.name, Array.isArray(dataset.splits) ? dataset.splits.join("/") : ""]
          .filter(Boolean)
          .join(" / ") || missing;
        const labelsText = [inputs.individual_label_file, inputs.group_label_file]
          .filter(Boolean)
          .join(" / ") || missing;
        const roots = Array.isArray(inputs.score_search_roots) ? inputs.score_search_roots.filter(Boolean) : [];
        const rootCount = Number(inputs.score_search_root_count || roots.length || 0);
        const rootsText = roots.length ? `${{rootCount}}: ${{roots.map(item => esc(item)).join(", ")}}` : String(rootCount);
        const parameterText = Object.keys(parameters).length
          ? Object.entries(parameters).map(([key, value]) => `${{key}}=${{value}}`).join(", ")
          : missing;
        const items = [
          [t("provenanceMode"), provenance.mode || missing],
          [t("provenanceGeneratedAt"), provenance.generated_at_utc || missing],
          [t("provenanceDataset"), datasetText],
          [t("provenanceDatasetStatus"), dataset.status || missing],
          [t("provenanceDatasetFingerprint"), dataset.fingerprint || missing],
          [t("provenanceDatasetManifest"), dataset.manifest || missing],
          [t("provenanceFinalResults"), inputs.final_results_root || missing],
          [t("provenanceLabelFiles"), labelsText],
          [t("provenanceScoreRoots"), rootsText],
          [t("provenanceFusedJsonl"), inputs.fused_jsonl || missing],
          [t("provenanceRegistration"), inputs.registration_manifest || missing],
          [t("provenanceParameters"), parameterText],
        ];
        return `
          <div id="provenancePanel" class="provenance-panel">
            <h3>${{t("provenanceTitle")}}</h3>
            <div class="provenance-grid">
              ${{items.map(([label, value]) => `
                <div class="provenance-item">
                  <span>${{label}}</span>
                  <strong>${{esc(value)}}</strong>
                </div>
              `).join("")}}
            </div>
          </div>
        `;
      }}

      function renderDataFlowAudit() {{
        const sequenceNames = sequences();
        const allTracks = sequenceNames.flatMap(name => playbackData[name]?.tracks || []);
        const backgroundCount = sequenceNames.reduce((sum, name) => sum + ((playbackData[name]?.background_frames || []).length), 0);
        const audits = sequenceNames.map(name => playbackData[name]?.modality_audit || {{}});
        const pointCount = audits.reduce((sum, audit) => sum + Number(audit.point_count || 0), 0);
        const fusedPointCount = audits.reduce((sum, audit) => sum + Number(audit.fused_point_count || 0), 0);
        const rgbPointCount = audits.reduce((sum, audit) => sum + Number(audit.rgb_point_count || 0), 0);
        const thermalPointCount = audits.reduce((sum, audit) => sum + Number(audit.thermal_point_count || 0), 0);
        const missingModalities = audits.reduce((sum, audit) => sum + Number(audit.missing_rgb_points || 0) + Number(audit.missing_thermal_points || 0), 0);
        const frameStarts = sequenceNames.map(name => Number(playbackData[name]?.frame_range?.[0] || 0));
        const frameEnds = sequenceNames.map(name => Number(playbackData[name]?.frame_range?.[1] || 0));
        const frameText = sequenceNames.length ? `${{Math.min(...frameStarts)}}-${{Math.max(...frameEnds)}}` : "-";
        const cards = [
          [t("dataFlowSequences"), sequenceNames.length],
          [t("dataFlowTracks"), allTracks.length],
          [t("dataFlowFrames"), frameText],
          [t("dataFlowBackgrounds"), backgroundCount || t("dataFlowNoBackground")],
          [t("dataFlowPointCoverage"), pct(pointCount ? fusedPointCount / pointCount : 0)],
          [t("dataFlowRgbCoverage"), pct(pointCount ? rgbPointCount / pointCount : 0)],
          [t("dataFlowThermalCoverage"), pct(pointCount ? thermalPointCount / pointCount : 0)],
          [t("dataFlowMissingModalities"), missingModalities],
        ];
        const statusMeta = (status) => {{
          if (status === "ok") {{
            return {{ label: t("dataFlowOk"), cls: "ok" }};
          }}
          if (status === "no_tracks") {{
            return {{ label: t("dataFlowNoTracks"), cls: "missing" }};
          }}
          if (status === "missing_background") {{
            return {{ label: t("dataFlowMissing"), cls: "missing" }};
          }}
          return {{ label: t("dataFlowPartial"), cls: "partial" }};
        }};
        const sequenceRows = sequenceNames.map(name => {{
          const data = playbackData[name] || {{}};
          const audit = data.modality_audit || {{}};
          const status = statusMeta(audit.status || "missing_background");
          const missing = Number(audit.missing_rgb_points || 0) + Number(audit.missing_thermal_points || 0);
          return `
            <tr>
              <td><strong>${{esc(name)}}</strong></td>
              <td class="metric">${{audit.trajectory_count ?? (data.tracks || []).length}}</td>
              <td class="metric">${{audit.point_count ?? 0}}</td>
              <td class="metric">${{pct(audit.rgb_coverage || 0)}}</td>
              <td class="metric">${{pct(audit.thermal_coverage || 0)}}</td>
              <td class="metric">${{missing}}</td>
              <td class="metric">${{audit.background_frame_count || 0}}</td>
              <td class="metric">${{fmt(audit.modal_offset_mean || 0)}}</td>
              <td><span class="status-pill ${{status.cls}}">${{status.label}}</span></td>
            </tr>
          `;
        }}).join("");
        const taskRows = Object.entries(dashboard.tasks || {{}}).map(([taskName, task]) => {{
          const scored = allTracks.filter(track => Object.keys(trackScoresForTask(track, taskName) || {{}}).length > 0).length;
          const labeled = allTracks.filter(track => Number(trackLabelForTask(track, taskName).num_windows || 0) > 0 || Number(trackLabelForTask(track, taskName).label || 0) === 1).length;
          const label = taskName === "registration" ? t("taskRegistration") : taskName === "group" ? t("taskGroup") : t("taskIndividual");
          return `
            <tr>
              <td><strong>${{esc(label)}}</strong></td>
              <td class="metric">${{task.num_labels || 0}}</td>
              <td class="metric">${{task.num_positive || 0}}</td>
              <td class="metric">${{Object.keys(task.methods || {{}}).length}}</td>
              <td class="metric">${{scored}}</td>
              <td class="metric">${{labeled}}</td>
            </tr>
          `;
        }}).join("");
        dataFlowPanel.innerHTML = `
          ${{renderProvenanceAudit()}}
          ${{cards.map(([label, value]) => `<div class="data-flow-card"><span>${{label}}</span><strong>${{value}}</strong></div>`).join("")}}
          <div class="table-scroll" style="grid-column: 1 / -1;">
            <table class="leaderboard">
              <thead><tr><th>${{t("dataFlowSequenceAudit")}}</th><th class="metric">${{t("dataFlowTracks")}}</th><th class="metric">${{t("dataFlowPoints")}}</th><th class="metric">${{t("dataFlowRgbCoverage")}}</th><th class="metric">${{t("dataFlowThermalCoverage")}}</th><th class="metric">${{t("dataFlowMissingModalities")}}</th><th class="metric">${{t("dataFlowBackgrounds")}}</th><th class="metric">${{t("dataFlowAvgOffset")}}</th><th>${{t("dataFlowBackgroundStatus")}}</th></tr></thead>
              <tbody>${{sequenceRows}}</tbody>
            </table>
          </div>
          <div class="table-scroll" style="grid-column: 1 / -1;">
            <table class="leaderboard">
              <thead><tr><th>${{t("dataFlowTaskAudit")}}</th><th class="metric">${{t("cardLabels")}}</th><th class="metric">${{t("cardPositives")}}</th><th class="metric">${{t("cardMethods")}}</th><th class="metric">${{t("dataFlowScoreCoverage")}}</th><th class="metric">${{t("dataFlowLabelCoverage")}}</th></tr></thead>
              <tbody>${{taskRows}}</tbody>
            </table>
          </div>
        `;
      }}

      function renderSequenceStats() {{
        const data = currentPlayback();
        if (!data) {{
          sequenceStats.innerHTML = "";
          return;
        }}
        const stats = (data.stats_by_task || {{}})[state.task] || data.stats || {{}};
        const frameStart = stats.frame_start ?? data.frame_range?.[0] ?? 0;
        const frameEnd = stats.frame_end ?? data.frame_range?.[1] ?? frameStart;
        const rows = [
          [t("sequenceSampleCount"), stats.sequence_sample_count ?? data.tracks.length],
          [t("sequenceAnomalyCount"), stats.sequence_anomaly_count ?? data.tracks.filter(track => trackLabelValue(track) === 1).length],
          [t("sequenceFrameRange"), `${{frameStart}}-${{frameEnd}}`],
          [t("sequenceVisualizedTracks"), stats.visualized_tracks ?? data.tracks.length]
        ];
        sequenceStats.innerHTML = rows.map(([label, value]) => `
          <div class="sequence-stat"><span>${{label}}</span><strong>${{value}}</strong></div>
        `).join("");
      }}

      function renderLeaderboard() {{
        const rows = taskData().leaderboard.slice(0, 8);
        leaderboardTable.innerHTML = `
          <thead><tr><th>${{t("methodHeader")}}</th><th>${{t("roleHeader")}}</th><th class="metric">AUROC</th><th class="metric">AUPRC</th><th class="metric">F1</th><th class="metric">P@100</th><th class="metric">R@100</th></tr></thead>
          <tbody>${{rows.map(row => `
            <tr>
              <td><strong>${{esc(row.method)}}</strong><br><span class="badge ${{row.is_our_method ? "our" : "baseline"}}">${{esc(row.owner || "")}}</span></td>
              <td>${{esc(row.role || row.method_family || "")}}</td>
              <td class="metric">${{fmt(row.auroc)}}</td>
              <td class="metric">${{fmt(row.auprc)}}</td>
              <td class="metric">${{fmt(row.f1)}}</td>
              <td class="metric">${{fmt(row.precision_at_k)}}</td>
              <td class="metric">${{fmt(row.recall_at_k)}}</td>
            </tr>`).join("")}}</tbody>
        `;
      }}

      function renderTypeTable() {{
        if (isRegistrationTask()) {{
          typeTable.innerHTML = `
            <thead><tr><th>${{t("methodHeader")}}</th><th class="metric">${{t("registrationSuccessRate")}}</th><th class="metric">${{t("registrationMetricRotation")}}</th><th class="metric">${{t("registrationMetricTranslation")}}</th><th class="metric">${{t("registrationMetricChamfer")}}</th></tr></thead>
            <tbody>${{(taskData().leaderboard || []).map(row => `
              <tr><td>${{esc(row.method)}}</td><td class="metric">${{fmt(row.success_rate || row.auroc || 0)}}</td><td class="metric">${{fmt(row.rotation_error_deg_mean || 0)}}</td><td class="metric">${{fmt(row.translation_error_mean || 0)}}</td><td class="metric">${{fmt(row.chamfer_distance_mean || 0)}}</td></tr>
            `).join("")}}</tbody>
          `;
          return;
        }}
        const rows = taskData().anomaly_type_rows.filter(row => row.method === state.method);
        typeTable.innerHTML = `
          <thead><tr><th>${{t("anomalyTypeHeader")}}</th><th class="metric">${{t("hitsHeader")}}</th><th class="metric">${{t("totalHeader")}}</th><th class="metric">${{t("recallHeader")}}</th><th class="metric">${{t("meanScoreHeader")}}</th></tr></thead>
          <tbody>${{rows.map(row => `
            <tr><td>${{esc(row.anomaly_type)}}</td><td class="metric">${{row.hits_at_k}}</td><td class="metric">${{row.total_positive}}</td><td class="metric">${{fmt(row.recall_at_k)}}</td><td class="metric">${{fmt(row.mean_positive_score)}}</td></tr>
          `).join("")}}</tbody>
        `;
      }}

      function renderCases() {{
        const rows = ((taskData().case_rows[state.method] || {{}})[state.caseType] || []);
        caseTable.innerHTML = `
          <thead><tr><th>${{t("sampleHeader")}}</th><th>${{t("typeHeader")}}</th><th class="metric">${{t("scoreHeader")}}</th><th class="metric">${{t("rankHeader")}}</th><th>${{t("framesHeader")}}</th></tr></thead>
          <tbody>${{rows.map(row => `
            <tr><td><strong>${{esc(row.sequence)}} / ${{esc(row.track_id)}}</strong><br><span class="subtle">${{esc(row.sample_id)}}</span></td><td>${{esc(row.anomaly_type)}}</td><td class="metric">${{fmt(row.score)}}</td><td class="metric">${{row.rank}}</td><td>${{row.frame_start}}-${{row.frame_end}}</td></tr>
          `).join("")}}</tbody>
        `;
      }}

      function renderCaseTabs() {{
        const labels = isRegistrationTask() ? {{
          true_positive: t("registrationSuccess"),
          false_positive: t("falsePositive"),
          false_negative: t("registrationFailed")
        }} : {{
          true_positive: t("truePositive"),
          false_positive: t("falsePositive"),
          false_negative: t("falseNegative")
        }};
        caseTabs.forEach(button => {{
          button.textContent = labels[button.dataset.case] || button.dataset.case;
        }});
      }}

      function setAnalysisPanel(panel) {{
        analysisTabs.forEach(tab => tab.classList.toggle("active", tab.dataset.panel === panel));
        analysisPanels.forEach(item => {{
          item.hidden = item.dataset.analysisPanel !== panel;
        }});
      }}

      function resetFrameForSequence() {{
        const data = currentPlayback();
        if (!data) {{
          state.frame = 0;
          frameSlider.min = 0;
          frameSlider.max = 0;
          frameSlider.value = 0;
          frameBadge.textContent = "0 / 0";
          return;
        }}
        const start = Number(data.frame_range?.[0] || 0);
        const end = Number(data.frame_range?.[1] || start);
        if (state.frame < start || state.frame > end) {{
          state.frame = Math.round(start + (end - start) * 0.35);
        }}
        frameSlider.min = start;
        frameSlider.max = end;
        frameSlider.value = state.frame;
        frameBadge.textContent = `${{state.frame}} / ${{end}}`;
      }}

      function backgroundForFrame(data, frame) {{
        const frames = data.background_frames || [];
        if (!frames.length) {{
          return data.background ? {{ frame: data.frame_range?.[0] || 0, src: data.background }} : null;
        }}
        let selected = frames[0];
        for (const item of frames) {{
          if (Number(item.frame) <= frame) {{
            selected = item;
          }} else {{
            break;
          }}
        }}
        return selected;
      }}

      function backgroundFallbackForFrame(data, background) {{
        if (!data || !background) {{
          return null;
        }}
        const src = background.fallback_src || data.background;
        if (!src || src === background.src) {{
          return null;
        }}
        return {{ frame: data.frame_range?.[0] || background.frame || 0, src }};
      }}

      function backgroundCandidatesForFrame(data, frame) {{
        const primary = backgroundForFrame(data, frame);
        const fallback = backgroundFallbackForFrame(data, primary);
        const seen = new Set();
        return [primary, fallback].filter(item => {{
          if (!item || !item.src || seen.has(item.src)) {{
            return false;
          }}
          seen.add(item.src);
          return true;
        }});
      }}

      function hasVideoBackground(data) {{
        return Boolean(data && (data.background || (data.background_frames || []).length));
      }}

      function playbackMediaKind(data) {{
        const media = (data && data.media) || {{}};
        if (media.kind) {{
          return media.kind;
        }}
        if (hasVideoBackground(data)) {{
          return "original_video_background";
        }}
        return isRegistrationTask() ? "registration_point_cloud" : "track_only_missing_background";
      }}

      function playbackMediaLabel(data) {{
        const media = (data && data.media) || {{}};
        const fallbackKeys = {{
          original_video_background: "mediaKindVideo",
          registration_point_cloud: "mediaKindRegistration",
          track_only_missing_background: "mediaKindTrackOnly"
        }};
        return t(media.label_key || fallbackKeys[playbackMediaKind(data)] || "mediaKindTrackOnly");
      }}

      function playbackMediaNoticeKey(data) {{
        const media = (data && data.media) || {{}};
        if (media.explanation_key) {{
          return media.explanation_key;
        }}
        return playbackMediaKind(data) === "registration_point_cloud"
          ? "registrationNoVideoBackground"
          : "sequenceNoVideoBackground";
      }}

      function playbackUsesOriginalVideo(data) {{
        return playbackMediaKind(data) === "original_video_background" && hasVideoBackground(data) && !isRegistrationTask();
      }}

      function canvasPlaceholderText(data) {{
        if (playbackMediaKind(data) === "registration_point_cloud") {{
          return t("registrationNoVideoBackgroundShort");
        }}
        if (!hasVideoBackground(data)) {{
          return t("sequenceNoVideoBackgroundShort");
        }}
        return state.imageStatus === "failed" ? t("backgroundLoadFailed") : t("backgroundLoading");
      }}

      function updateBackgroundNotice(data) {{
        if (!backgroundNotice) {{
          return;
        }}
        if (!data || playbackUsesOriginalVideo(data)) {{
          backgroundNotice.hidden = true;
          backgroundNotice.textContent = "";
          return;
        }}
        backgroundNotice.textContent = t(playbackMediaNoticeKey(data));
        backgroundNotice.hidden = false;
      }}

      function syncPlaybackModeForData(data) {{
        const canCompare = playbackUsesOriginalVideo(data);
        if (!canCompare && state.viewMode === "comparison") {{
          state.viewMode = "single";
        }}
        viewModeButtons.forEach(button => {{
          const isComparison = button.dataset.viewMode === "comparison";
          button.disabled = isComparison && !canCompare;
          button.title = isComparison && !canCompare ? t("comparisonRequiresBackground") : "";
          button.classList.toggle("active", button.dataset.viewMode === state.viewMode);
        }});
      }}

      function ensureBackground(data, frame) {{
        const candidates = backgroundCandidatesForFrame(data, frame).filter(background => {{
          const key = `${{data.sequence}}:${{background.src}}`;
          return !backgroundFailures.has(key);
        }});
        if (!candidates.length) {{
          state.image = null;
          state.imageKey = null;
          state.imageStatus = hasVideoBackground(data) ? "failed" : "idle";
          return;
        }}
        const cached = candidates.find(background => backgroundCache.has(`${{data.sequence}}:${{background.src}}`));
        if (cached) {{
          const cachedKey = `${{data.sequence}}:${{cached.src}}`;
          state.image = backgroundCache.get(cachedKey);
          state.imageKey = cachedKey;
          state.imageStatus = "loaded";
          return;
        }}
        const background = candidates[0];
        const key = `${{data.sequence}}:${{background.src}}`;
        if (state.imageKey === key) {{
          return;
        }}
        state.imageKey = key;
        state.imageStatus = "loading";
        const image = new Image();
        image.onload = () => {{
          backgroundCache.set(key, image);
          if (state.imageKey === key) {{
            state.image = image;
            state.imageStatus = "loaded";
            drawPlayback();
          }}
        }};
        image.onerror = () => {{
          backgroundFailures.add(key);
          if (state.imageKey === key) {{
            state.image = null;
            state.imageKey = null;
            ensureBackground(data, frame);
            if (state.imageStatus === "failed" || state.image) {{
              drawPlayback();
            }}
          }}
        }};
        image.src = background.src;
      }}

      function activePoints(track, frame) {{
        const points = track.points.filter(point => Number(point.frame) <= frame);
        return points.length ? points : track.points.slice(0, 1);
      }}

      function heatPoints(track, frame) {{
        const start = frame - state.heatWindow;
        const points = track.points.filter(point => Number(point.frame) <= frame && Number(point.frame) >= start);
        const visible = points.length ? points : activePoints(track, frame).slice(-1);
        if (visible.length <= 12) {{
          return visible;
        }}
        const stride = Math.ceil(visible.length / 12);
        return visible.filter((_, index) => index % stride === 0).slice(-12);
      }}

      function setCanvasSize(targetCanvas, data) {{
        const width = Number(data.size?.width || 960);
        const height = Number(data.size?.height || 612);
        if (targetCanvas.width !== width) {{
          targetCanvas.width = width;
        }}
        if (targetCanvas.height !== height) {{
          targetCanvas.height = height;
        }}
      }}

      function clearPlaybackCanvases() {{
        Object.values(canvases).forEach(targetCanvas => {{
          if (!targetCanvas) {{
            return;
          }}
          const targetCtx = targetCanvas.getContext("2d");
          targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
        }});
      }}

      function setRegistrationCanvasSize(targetCanvas) {{
        if (!targetCanvas) {{
          return;
        }}
        if (targetCanvas.width !== 960) {{
          targetCanvas.width = 960;
        }}
        if (targetCanvas.height !== 520) {{
          targetCanvas.height = 520;
        }}
      }}

      function drawRegistrationAxes(targetCtx, targetCanvas) {{
        const width = targetCanvas.width;
        const height = targetCanvas.height;
        targetCtx.save();
        targetCtx.strokeStyle = "rgba(148, 163, 184, 0.22)";
        targetCtx.lineWidth = 1;
        for (let x = width * 0.18; x <= width * 0.82; x += width * 0.08) {{
          targetCtx.beginPath();
          targetCtx.moveTo(x, height * 0.18);
          targetCtx.lineTo(x, height * 0.82);
          targetCtx.stroke();
        }}
        for (let y = height * 0.2; y <= height * 0.82; y += height * 0.09) {{
          targetCtx.beginPath();
          targetCtx.moveTo(width * 0.14, y);
          targetCtx.lineTo(width * 0.86, y);
          targetCtx.stroke();
        }}
        targetCtx.strokeStyle = "rgba(226, 232, 240, 0.38)";
        targetCtx.beginPath();
        targetCtx.ellipse(width / 2, height / 2 + 42, width * 0.28, height * 0.16, 0, 0, Math.PI * 2);
        targetCtx.stroke();
        targetCtx.restore();
      }}

      function renderRegistrationPlaybackSummary(track) {{
        if (!registrationPlaybackSummary) {{
          return;
        }}
        if (!track) {{
          registrationPlaybackSummary.textContent = t("noTrackSelected");
          return;
        }}
        const row = selectedTrackScoreComponents(track);
        const success = row.success === true || row.success === "true" || row.success === 1 || row.success === "1";
        const skipped = row.skipped === true || row.skipped === "true" || row.skipped === 1 || row.skipped === "1";
        const metrics = [
          [t("registrationSelectedPair"), `${{esc(track.sequence)}} / ${{esc(track.track_id)}}`],
          [t("registrationMethod"), esc(state.method)],
          [t("registrationMetricRotation"), metricValue(row, "rotation_error_deg") === null ? "-" : `${{fmt(metricValue(row, "rotation_error_deg"))}} deg`],
          [t("registrationMetricTranslation"), metricValue(row, "translation_error") === null ? "-" : fmt(metricValue(row, "translation_error"))],
          [t("registrationMetricChamfer"), metricValue(row, "chamfer_distance") === null ? "-" : fmt(metricValue(row, "chamfer_distance"))],
          [t("registrationMetricRuntime"), metricValue(row, "runtime_sec") === null ? "-" : `${{fmt(metricValue(row, "runtime_sec"))}} s`],
          [t("registrationPairStatus"), success && !skipped ? t("registrationSuccess") : t("registrationFailed")],
          [t("anomalyScore"), fmt(trackScore(track))],
        ];
        registrationPlaybackSummary.innerHTML = `
          <div class="explain-metrics">
            ${{metrics.map(([label, value]) => `<div class="explain-metric"><span>${{label}}</span><strong>${{value}}</strong></div>`).join("")}}
          </div>
          <div class="explain-reason">${{t("registrationNoVideoBackground")}}</div>
        `;
      }}

      function drawRegistrationPlayback(data, ranked, maxScore) {{
        const targetCanvas = canvases.registration;
        if (!targetCanvas) {{
          return;
        }}
        setRegistrationCanvasSize(targetCanvas);
        const targetCtx = targetCanvas.getContext("2d");
        const width = targetCanvas.width;
        const height = targetCanvas.height;
        targetCtx.fillStyle = "#0f172a";
        targetCtx.fillRect(0, 0, width, height);
        drawRegistrationAxes(targetCtx, targetCanvas);
        const track = ensureSelectedTrack(data, ranked);
        renderRegistrationPlaybackSummary(track);
        const groups = registrationPointGroups(track);
        if (!groups.length || !groups.some(group => group.points.length)) {{
          drawCanvasPlaceholder(targetCtx, targetCanvas, t("registrationNoPointCloud"));
          return;
        }}
        const angle = Number(state.frame || 0) * 0.045 + (state.playing ? 0.12 : 0);
        const projected = [];
        groups.forEach(group => {{
          group.points.forEach(point => {{
            projected.push({{ ...projectRegistrationPoint(point, angle, width, height), group }});
          }});
        }});
        projected.sort((a, b) => a.depth - b.depth);
        targetCtx.save();
        for (const item of projected) {{
          targetCtx.fillStyle = item.group.color;
          targetCtx.globalAlpha = item.group.key === "aligned" ? 0.88 : 0.72;
          targetCtx.beginPath();
          targetCtx.arc(item.x, item.y, item.radius, 0, Math.PI * 2);
          targetCtx.fill();
        }}
        targetCtx.globalAlpha = 1;
        targetCtx.fillStyle = "#e2e8f0";
        targetCtx.font = "700 15px Arial, sans-serif";
        targetCtx.fillText(`${{t("registrationPlaybackTitle")}} · ${{state.method}}`, 22, 30);
        targetCtx.restore();
      }}

      function setViewModeVisibility(data = currentPlayback()) {{
        setPlaybackSurfaceForTask(data);
      }}

      function setPlaybackSurfaceForTask(data = currentPlayback()) {{
        const registration = isRegistrationTask();
        if (modeSwitch) {{
          modeSwitch.hidden = registration;
        }}
        if (heatControlsPanel) {{
          heatControlsPanel.hidden = registration;
        }}
        if (registrationPlaybackView) {{
          registrationPlaybackView.hidden = !registration;
        }}
        if (registration) {{
          comparisonView.hidden = true;
          singleView.hidden = true;
          singleLayerSwitch.hidden = true;
          viewModeButtons.forEach(button => {{
            button.disabled = true;
            button.classList.remove("active");
            button.title = t("registrationNoVideoBackgroundShort");
          }});
          return;
        }}
        syncPlaybackModeForData(data);
        const comparison = state.viewMode === "comparison" && playbackUsesOriginalVideo(data);
        comparisonView.hidden = !comparison;
        singleView.hidden = comparison;
        singleLayerSwitch.hidden = comparison;
      }}

      function drawCanvasBase(targetCtx, targetCanvas, data, layer) {{
        targetCtx.fillStyle = state.image ? "#e2e8f0" : "#0f172a";
        targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
        if (state.image) {{
          targetCtx.drawImage(state.image, 0, 0, targetCanvas.width, targetCanvas.height);
        }} else {{
          drawCanvasPlaceholder(targetCtx, targetCanvas, canvasPlaceholderText(data));
        }}
        if (layer === "heatmap" || layer === "both") {{
          targetCtx.save();
          targetCtx.fillStyle = layer === "heatmap" ? "rgba(4, 9, 18, 0.18)" : "rgba(4, 9, 18, 0.06)";
          targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
          targetCtx.restore();
        }}
      }}

      function drawCanvasPlaceholder(targetCtx, targetCanvas, text) {{
        targetCtx.save();
        targetCtx.strokeStyle = "rgba(148, 163, 184, 0.18)";
        targetCtx.lineWidth = 1;
        const step = Math.max(36, Math.round(targetCanvas.width / 18));
        for (let x = -targetCanvas.height; x < targetCanvas.width; x += step) {{
          targetCtx.beginPath();
          targetCtx.moveTo(x, targetCanvas.height);
          targetCtx.lineTo(x + targetCanvas.height, 0);
          targetCtx.stroke();
        }}
        targetCtx.fillStyle = "#e2e8f0";
        targetCtx.font = `${{Math.max(16, Math.round(targetCanvas.width / 42))}}px Arial, sans-serif`;
        targetCtx.textAlign = "center";
        targetCtx.textBaseline = "middle";
        const maxWidth = targetCanvas.width * 0.82;
        const words = String(text || "").split(/\\s+/);
        const lines = [];
        let current = "";
        for (const word of words) {{
          const next = current ? `${{current}} ${{word}}` : word;
          if (targetCtx.measureText(next).width > maxWidth && current) {{
            lines.push(current);
            current = word;
          }} else {{
            current = next;
          }}
        }}
        if (current) {{
          lines.push(current);
        }}
        const compactLines = lines.length ? lines.slice(0, 3) : [String(text || "")];
        const lineHeight = Math.max(22, Math.round(targetCanvas.width / 34));
        const startY = targetCanvas.height / 2 - ((compactLines.length - 1) * lineHeight) / 2;
        compactLines.forEach((line, index) => {{
          targetCtx.fillText(line, targetCanvas.width / 2, startY + index * lineHeight, maxWidth);
        }});
        targetCtx.restore();
      }}

      function drawHeatmap(targetCtx, targetCanvas, data, ranked, maxScore) {{
        const heatCanvas = document.createElement("canvas");
        heatCanvas.width = targetCanvas.width;
        heatCanvas.height = targetCanvas.height;
        const heatCtx = heatCanvas.getContext("2d");
        heatCtx.clearRect(0, 0, heatCanvas.width, heatCanvas.height);
        heatCtx.globalCompositeOperation = "lighter";
        const currentFrame = Number(state.frame);
        for (const track of ranked.slice(0, 55)) {{
          const score = trackScore(track);
          const scoreRatio = clamp(score / maxScore, 0, 1);
          const labelBoost = trackLabelValue(track) === 1 ? 0.22 : 0;
          for (const point of heatPoints(track, currentFrame)) {{
            const age = Math.max(0, currentFrame - Number(point.frame));
            const recency = clamp(1 - age / Math.max(1, state.heatWindow), 0.18, 1);
            const strength = clamp((0.16 + 0.72 * scoreRatio + labelBoost) * recency, 0.08, 1);
            const radius = 14 + 26 * strength;
            const gradient = heatCtx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius);
            gradient.addColorStop(0, `rgba(255, 43, 85, ${{0.30 * strength}})`);
            gradient.addColorStop(0.28, `rgba(255, 179, 0, ${{0.23 * strength}})`);
            gradient.addColorStop(0.62, `rgba(45, 212, 191, ${{0.15 * strength}})`);
            gradient.addColorStop(1, "rgba(45, 212, 191, 0)");
            heatCtx.fillStyle = gradient;
            heatCtx.beginPath();
            heatCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            heatCtx.fill();
          }}
        }}
        targetCtx.save();
        targetCtx.globalAlpha = state.heatOpacity;
        targetCtx.globalCompositeOperation = "screen";
        targetCtx.drawImage(heatCanvas, 0, 0);
        targetCtx.restore();
      }}

      function drawGroupRelations(targetCtx, data, ranked) {{
        if (state.task !== "group") {{
          return;
        }}
        const frame = Number(state.frame);
        const visible = ranked.slice(0, 35).map(track => ({{ track, point: pointAtFrame(track, frame) }})).filter(item => item.point);
        if (visible.length < 2) {{
          return;
        }}
        const centroid = {{
          x: visible.reduce((sum, item) => sum + Number(item.point.x), 0) / visible.length,
          y: visible.reduce((sum, item) => sum + Number(item.point.y), 0) / visible.length
        }};
        const radius = visible.reduce((sum, item) => sum + Math.hypot(Number(item.point.x) - centroid.x, Number(item.point.y) - centroid.y), 0) / visible.length;
        targetCtx.save();
        targetCtx.strokeStyle = "rgba(20, 184, 166, 0.32)";
        targetCtx.lineWidth = 1.2;
        targetCtx.setLineDash([5, 7]);
        targetCtx.beginPath();
        targetCtx.arc(centroid.x, centroid.y, Math.max(8, radius), 0, Math.PI * 2);
        targetCtx.stroke();
        targetCtx.setLineDash([]);
        for (const item of visible) {{
          targetCtx.strokeStyle = item.track.sample_id === state.selectedSampleId ? "rgba(20, 184, 166, 0.72)" : "rgba(20, 184, 166, 0.18)";
          targetCtx.beginPath();
          targetCtx.moveTo(centroid.x, centroid.y);
          targetCtx.lineTo(item.point.x, item.point.y);
          targetCtx.stroke();
        }}
        targetCtx.fillStyle = "#14b8a6";
        targetCtx.beginPath();
        targetCtx.arc(centroid.x, centroid.y, 4.8, 0, Math.PI * 2);
        targetCtx.fill();
        targetCtx.restore();
      }}

      function drawTracks(targetCtx, ranked, maxScore) {{
        for (const track of ranked) {{
          const points = activePoints(track, state.frame);
          if (!points.length) {{
            continue;
          }}
          const score = trackScore(track);
          const isAnomaly = trackLabelValue(track) === 1;
          const isSelected = track.sample_id === state.selectedSampleId;
          const ratio = clamp(score / maxScore, 0, 1);
          targetCtx.strokeStyle = isSelected ? "rgba(20, 184, 166, 0.98)" : isAnomaly ? "rgba(239, 68, 68, 0.95)" : `rgba(37, 99, 235, ${{0.25 + 0.45 * ratio}})`;
          targetCtx.lineWidth = isSelected ? 3.4 : isAnomaly ? 2.6 : 0.9 + 2.0 * ratio;
          targetCtx.beginPath();
          points.forEach((point, index) => index ? targetCtx.lineTo(point.x, point.y) : targetCtx.moveTo(point.x, point.y));
          targetCtx.stroke();
          const last = points[points.length - 1];
          if (last) {{
            targetCtx.fillStyle = isSelected ? "#14b8a6" : isAnomaly ? "#ef4444" : "#f59e0b";
            targetCtx.beginPath();
            targetCtx.arc(last.x, last.y, isSelected ? 6.2 : isAnomaly ? 4.8 : 2.7, 0, Math.PI * 2);
            targetCtx.fill();
          }}
        }}
      }}

      function drawCanvasLayer(targetCanvas, layer, data, ranked, maxScore) {{
        if (!targetCanvas) {{
          return;
        }}
        setCanvasSize(targetCanvas, data);
        const targetCtx = targetCanvas.getContext("2d");
        drawCanvasBase(targetCtx, targetCanvas, data, layer);
        if (layer === "original") {{
          return;
        }}
        if (layer === "heatmap" || layer === "both") {{
          drawHeatmap(targetCtx, targetCanvas, data, ranked, maxScore);
        }}
        if (layer === "tracks" || layer === "both") {{
          drawGroupRelations(targetCtx, data, ranked);
          drawTracks(targetCtx, ranked, maxScore);
        }}
      }}

      function drawComparisonView(data, ranked, maxScore) {{
        drawCanvasLayer(canvases.original, "original", data, ranked, maxScore);
        drawCanvasLayer(canvases.heatmap, "heatmap", data, ranked, maxScore);
        drawCanvasLayer(canvases.tracks, "tracks", data, ranked, maxScore);
        drawCanvasLayer(canvases.both, "both", data, ranked, maxScore);
      }}

      function drawSingleView(data, ranked, maxScore) {{
        drawCanvasLayer(canvases.single, state.layer, data, ranked, maxScore);
      }}

      function drawPlayback() {{
        const names = sequencesForTask();
        if (!names.length) {{
          playbackReadout.textContent = t("noPlayback");
          updateBackgroundNotice(null);
          clearPlaybackCanvases();
          return;
        }}
        const data = currentPlayback();
        if (!data) {{
          return;
        }}
        resetFrameForSequence();
        const scores = data.tracks.map(track => trackScore(track));
        const maxScore = Math.max(...scores, 1e-6);
        const ranked = rankedTracks(data);
        ensureSelectedTrack(data, ranked);
        const speedText = `x${{state.playSpeed.toFixed(1)}}`;
        setPlaybackSurfaceForTask(data);
        updateBackgroundNotice(data);
        if (isRegistrationTask()) {{
          state.image = null;
          state.imageKey = null;
          state.imageStatus = "idle";
          drawRegistrationPlayback(data, ranked, maxScore);
        }} else {{
          ensureBackground(data, state.frame);
        }}
        if (!isRegistrationTask() && state.viewMode === "comparison" && playbackUsesOriginalVideo(data)) {{
          drawComparisonView(data, ranked, maxScore);
        }} else if (!isRegistrationTask()) {{
          drawSingleView(data, ranked, maxScore);
        }}
        renderTrackInsights(ranked);
        const viewLabel = isRegistrationTask()
          ? t("view_registration")
          : state.viewMode === "comparison" ? t("view_comparison") : `${{t("view_single")}} - ${{t(`layer_${{state.layer}}`)}}`;
        playbackReadout.textContent = `${{t("playbackPrefix")}} / ${{data.sequence}} / ${{state.method}} / ${{t("frame")}} ${{state.frame}} / ${{t("play")}} ${{speedText}} / ${{viewLabel}} / ${{playbackMediaLabel(data)}} / ${{ranked.length}} ${{t("visibleTracks")}}`;
      }}

      function stopPlayback() {{
        state.playing = false;
        playToggle.textContent = t("play");
        playToggle.classList.remove("active");
        if (state.timer) {{
          window.clearInterval(state.timer);
          state.timer = null;
        }}
      }}

      function updatePlaySpeedDisplay() {{
        const speed = Math.max(0.2, Math.min(3, Number(state.playSpeed) || 1));
        state.playSpeed = speed;
        if (playSpeed) {{
          playSpeed.value = String(Math.round(speed * 100));
        }}
        if (playSpeedReadout) {{
          playSpeedReadout.textContent = `${{speed.toFixed(1)}}x`;
        }}
        localStorage.setItem("fusiontrack.finalDashboard.playSpeed", String(speed));
      }}

      function updateEventThresholdDisplay() {{
        const threshold = clamp(Number(state.eventThreshold) || 0, 0, 1);
        state.eventThreshold = threshold;
        if (eventThreshold) {{
          eventThreshold.value = String(Math.round(threshold * 100));
        }}
        if (eventThresholdReadout) {{
          eventThresholdReadout.textContent = threshold.toFixed(2);
        }}
      }}

      function startPlayback() {{
        if (!currentPlayback()) {{
          return;
        }}
        state.playing = true;
        playToggle.textContent = t("pause");
        playToggle.classList.add("active");
        const speed = Math.max(0.2, Number(state.playSpeed || 1));
        state.timer = window.setInterval(() => {{
          const data = currentPlayback();
          const start = Number(data.frame_range?.[0] || 0);
          const end = Number(data.frame_range?.[1] || start);
          state.frame = state.frame >= end ? start : state.frame + 1;
          frameSlider.value = state.frame;
          drawPlayback();
        }}, 120 / speed);
      }}

      function pickTrackFromCanvas(event, targetCanvas) {{
        const data = currentPlayback();
        if (!data || !targetCanvas) {{
          return;
        }}
        const rect = targetCanvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * (targetCanvas.width / Math.max(1, rect.width));
        const y = (event.clientY - rect.top) * (targetCanvas.height / Math.max(1, rect.height));
        let best = null;
        let bestDistance = Infinity;
        for (const track of data.tracks || []) {{
          const point = pointAtFrame(track, state.frame);
          if (!point) {{
            continue;
          }}
          const distance = Math.hypot(Number(point.x) - x, Number(point.y) - y);
          if (distance < bestDistance) {{
            bestDistance = distance;
            best = track;
          }}
        }}
        if (best && bestDistance <= 28) {{
          state.selectedSampleId = best.sample_id;
          drawPlayback();
        }}
      }}

      function renderMethodView() {{
        setTaskOptions();
        setMethodOptions();
        setSequenceOptions();
        playToggle.textContent = state.playing ? t("pause") : t("play");
        renderCards();
        renderSequenceStats();
        renderCaseTabs();
        renderLeaderboard();
        renderTypeTable();
        renderCases();
        renderMethodStatus();
        renderDataFlowAudit();
        renderProtocolOverview();
        renderHelp();
        drawPlayback();
      }}

      languageSelector.addEventListener("change", () => {{
        applyLanguage(languageSelector.value);
        renderMethodView();
      }});
      taskSelector.addEventListener("change", () => {{
        state.task = taskSelector.value;
        state.method = methodsForTask(taskData())[0] || "";
        state.viewMode = isRegistrationTask() ? "single" : "comparison";
        state.sequence = "";
        state.image = null;
        state.imageKey = null;
        state.imageStatus = "idle";
        state.frame = -1;
        state.selectedSampleId = "";
        stopPlayback();
        renderMethodView();
      }});
      methodSelector.addEventListener("change", () => {{
        state.method = methodSelector.value;
        state.selectedSampleId = "";
        renderMethodView();
      }});
      caseTabs.forEach(button => button.addEventListener("click", () => {{
        state.caseType = button.dataset.case;
        caseTabs.forEach(tab => tab.classList.toggle("active", tab === button));
        renderCases();
      }}));
      analysisTabs.forEach(button => button.addEventListener("click", () => {{
        setAnalysisPanel(button.dataset.panel);
      }}));
      sequenceSelector.addEventListener("change", () => {{
        state.sequence = sequenceSelector.value;
        state.image = null;
        state.imageKey = null;
        state.imageStatus = "idle";
        state.frame = -1;
        state.selectedSampleId = "";
        resetFrameForSequence();
        renderSequenceStats();
        drawPlayback();
      }});
      frameSlider.addEventListener("input", () => {{
        state.frame = Number(frameSlider.value);
        drawPlayback();
      }});
      heatOpacity.addEventListener("input", () => {{
        state.heatOpacity = Number(heatOpacity.value) / 100;
        drawPlayback();
      }});
      heatWindow.addEventListener("input", () => {{
        state.heatWindow = Number(heatWindow.value);
        drawPlayback();
      }});
      eventThreshold.addEventListener("input", () => {{
        state.eventThreshold = Number(eventThreshold.value || 0) / 100;
        updateEventThresholdDisplay();
        drawPlayback();
      }});
      playSpeed.addEventListener("input", () => {{
        state.playSpeed = Number(playSpeed.value || 100) / 100;
        updatePlaySpeedDisplay();
        if (state.playing) {{
          stopPlayback();
          startPlayback();
        }}
        drawPlayback();
      }});
      viewModeButtons.forEach(button => button.addEventListener("click", () => {{
        if (button.disabled) {{
          return;
        }}
        const data = currentPlayback();
        if (button.dataset.viewMode === "comparison" && !playbackUsesOriginalVideo(data)) {{
          return;
        }}
        state.viewMode = button.dataset.viewMode || "comparison";
        setViewModeVisibility(data);
        drawPlayback();
      }}));
      layerButtons.forEach(button => button.addEventListener("click", () => {{
        state.layer = button.dataset.layer;
        layerButtons.forEach(item => item.classList.toggle("active", item === button));
        drawPlayback();
      }}));
      submoduleTabs.forEach(button => button.addEventListener("click", () => {{
        state.submodule = button.dataset.submodule || "route";
        submoduleTabs.forEach(item => item.classList.toggle("active", item === button));
        const track = activeTrack(true);
        renderSubmoduleTrack(track);
        renderCompositionBars(track);
      }}));
      playToggle.addEventListener("click", () => {{
        state.playing ? stopPlayback() : startPlayback();
      }});
      [canvases.heatmap, canvases.tracks, canvases.both, canvases.single].forEach(targetCanvas => {{
        if (targetCanvas) {{
          targetCanvas.addEventListener("click", event => pickTrackFromCanvas(event, targetCanvas));
        }}
      }});
      helpButton.addEventListener("click", () => {{
        renderHelp();
        if (typeof helpDialog.showModal === "function") {{
          helpDialog.showModal();
        }} else {{
          helpDialog.setAttribute("open", "open");
        }}
      }});
      helpClose.addEventListener("click", () => {{
        helpDialog.close();
      }});
      applyLanguage(state.language);
      updatePlaySpeedDisplay();
      updateEventThresholdDisplay();
      setAnalysisPanel("leaderboard");
      renderMethodView();
    }})();
  </script>
</body>
</html>
"""
