from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

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
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    dashboard_data = dashboard.to_public_dict()
    playback_payloads = {}
    if fused_jsonl is not None:
        playback_payloads = _build_playback_payloads(
            dashboard=dashboard,
            fused_jsonl=Path(fused_jsonl),
            data_root=Path(data_root) if data_root is not None else Path("data") / "VT-Tiny-MOT",
            assets_dir=assets_dir,
            top_sequences=top_sequences,
        )
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
                    method_name: {
                        "score": round(float(method_rows.get("score", 0.0) or 0.0), 6),
                        "used_sources": str(method_rows.get("used_sources", "")),
                        "source": str(method_rows.get("source", "")),
                        "component_scores": method_rows.get("component_scores", {}),
                        "metadata": method_rows.get("metadata", {}),
                    }
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
            stats_by_task[task_name] = {
                "sequence_sample_count": len(sequence_labels),
                "sequence_anomaly_count": sum(1 for row in sequence_labels if int(row.get("label", 0) or 0) == 1),
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
        payloads[sequence] = {
            "sequence": sequence,
            "background": f"assets/{background_asset.name}" if background_asset else None,
            "background_frames": [
                {"frame": int(item["frame"]), "src": f"assets/{item['path'].name}"}
                for item in background_frames
            ],
            "size": {"width": width, "height": height},
            "frame_range": [frame_start, frame_end],
            "stats": default_stats,
            "stats_by_task": stats_by_task,
            "tracks": tracks,
        }
    return payloads


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
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
        return _coerce_float(metadata.get("individual_raw_score"), 0.0)
    if key_prefix == "group":
        return _coerce_float(metadata.get("group_raw_score"), 0.0)
    return fallback_weight * _coerce_float(row.get("score"), 0.0)


def _score_decomposition(row: dict[str, Any]) -> dict[str, float]:
    components = row.get("component_scores") if isinstance(row.get("component_scores"), dict) else {}
    used_sources = str(row.get("used_sources", ""))
    has_individual = "individual" in used_sources
    has_group = "group" in used_sources
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    individual_raw = _coerce_float(metadata.get("individual_raw_score", 0.0), 0.0)
    group_raw = _coerce_float(metadata.get("group_raw_score", 0.0), 0.0)
    alpha = _coerce_float(
        metadata.get("alpha", 0.65) if isinstance(metadata.get("alpha"), (int, float)) else 0.65,
        0.65,
    )
    fused = _coerce_float(row.get("score", 0.0), 0.0)
    ind = 0.0
    grp = 0.0
    evt = _coerce_float(row.get("event_score", 0.0), 0.0)

    if has_individual and has_group:
        ind = _normalize_weighted_score(row, "individual", fallback_weight=alpha)
        grp = _normalize_weighted_score(row, "group", fallback_weight=(1 - alpha))
        if not ind and individual_raw:
            ind = individual_raw
        if not grp and group_raw:
            grp = group_raw
    elif has_individual:
        ind = _normalize_weighted_score(row, "individual", fallback_weight=1.0)
    elif has_group:
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
    .layer-switch {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .layer-button {{ min-height: 44px; padding: 7px 12px; }}
    .layer-button.active {{ background: #111827; border-color: #111827; color: white; }}
    .layer-switch[hidden], .comparison-grid[hidden], .single-view[hidden] {{ display: none; }}
    .heat-controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
    .heat-controls label {{ min-width: 180px; max-width: 260px; }}
    .sequence-stats {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 8px; }}
    .sequence-stat {{ border: 1px solid #e1e7ef; border-radius: 7px; padding: 8px 10px; background: white; }}
    .sequence-stat span {{ display: block; color: #64748b; font-size: 12px; }}
    .sequence-stat strong {{ display: block; margin-top: 3px; font-size: 17px; }}
    #frameBadge {{ color: #475569; font-size: 13px; font-variant-numeric: tabular-nums; }}
    .canvas-shell {{ background: #111827; border-radius: 8px; padding: 10px; }}
    .comparison-grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; }}
    .video-panel {{ min-width: 0; margin: 0; border: 1px solid #1f2937; border-radius: 8px; padding: 9px; background: #0f172a; }}
    .video-panel figcaption {{ display: flex; align-items: center; min-height: 24px; margin: 0 0 7px; color: #f8fafc; font-size: 12px; font-weight: 800; }}
    canvas {{ display: block; width: 100%; height: auto; background: #e2e8f0; border-radius: 6px; }}
    section {{ margin-top: 16px; }}
    .analysis-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .analysis-tab.active {{ background: #111827; border-color: #111827; color: white; }}
    .analysis-panel-block[hidden] {{ display: none; }}
    .table-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; border: 1px solid #eef2f7; border-radius: 8px; }}
    .table-scroll table {{ background: white; }}
    .help-button {{ background: #0f766e; border-color: #0f766e; color: white; }}
    .help-button:hover {{ background: #115e59; border-color: #115e59; }}
    .protocol-strip {{ display: grid; grid-template-columns: minmax(260px, 0.9fr) repeat(2, minmax(260px, 1fr)); gap: 12px; margin: 0 0 16px; }}
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
      .toolbar {{ display: grid; width: 100%; justify-content: stretch; }}
      .toolbar label {{ width: 100%; }}
      .section-heading {{ display: grid; }}
      .section-heading .subtle {{ text-align: left; }}
      .control-surface {{ padding: 10px; }}
      .mode-switch button, .layer-switch button {{ flex: 1 1 140px; }}
      .protocol-strip, .insight-grid, .method-summary {{ grid-template-columns: 1fr; }}
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
        </div>
        <div id="sequenceStats" class="sequence-stats"></div>
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
    </section>
    <dialog id="helpDialog">
      <div class="help-dialog-head">
        <h2 data-i18n="helpTitle">网页说明</h2>
        <button type="button" class="secondary-button" id="helpClose" data-i18n="closeHelp">关闭</button>
      </div>
      <div class="help-dialog-body" id="helpBody"></div>
    </dialog>
  </main>
  <script id="dashboardData" type="application/json">{{dashboard_json}}</script>
  <script id="playbackData" type="application/json">{{playback_json}}</script>
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
      const leaderboardTable = document.getElementById("leaderboardTable");
      const typeTable = document.getElementById("typeTable");
      const caseTable = document.getElementById("caseTable");
      const methodSummary = document.getElementById("methodSummary");
      const methodStatusTable = document.getElementById("methodStatusTable");
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
        single: document.getElementById("singleCanvas")
      }};
      const comparisonView = document.getElementById("comparisonView");
      const singleView = document.getElementById("singleView");
      const singleLayerSwitch = document.getElementById("singleLayerSwitch");
      const playbackReadout = document.getElementById("playbackReadout");
      const sequenceSelector = document.getElementById("sequenceSelector");
      const playToggle = document.getElementById("playToggle");
      const frameSlider = document.getElementById("frameSlider");
      const frameBadge = document.getElementById("frameBadge");
      const heatOpacity = document.getElementById("heatOpacity");
      const heatWindow = document.getElementById("heatWindow");
      const sequenceStats = document.getElementById("sequenceStats");
      const viewModeButtons = Array.from(document.querySelectorAll(".view-mode-button"));
      const layerButtons = Array.from(document.querySelectorAll(".layer-button"));
      const translations = {{
        zh: {{
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
        en: {{
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
          noPlayback: "Playback is not available for the current task.",
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
      const backgroundCache = new Map();
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
        selectedSampleId: "",
        image: null,
        imageKey: null,
        timer: null
      }};

      const anomalyDescriptions = {{
        zh: {{
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
        en: {{
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

      function taskData() {{ return dashboard.tasks[state.task]; }}
      function fmt(value) {{ return Number(value || 0).toFixed(3); }}
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
          const hasLabel = Number(label.num_windows || 0) > 0 || Number(label.label || 0) === 1;
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

      function renderMethodFlow(track) {{
        const task = taskData();
        const labels = {{
          prepare: t("flowStepPrepare"),
          features: t("flowStepFeatures"),
          individual: t("flowStepIndividual"),
          group: t("flowStepGroup"),
          fusion: t("flowStepFusion"),
        }};
        const decomp = selectedTrackDecomposition(track);
        const steps = [
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
        const taskLabel = state.task === "group" ? t("taskGroup") : t("taskIndividual");
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

      function renderSubmoduleTrack(track) {{
        const stats = trajectoryStats(track || {{}});  
        const score = submoduleFeatureValue(track, stats, state.submodule);
        const label = track ? trackLabel(track) : {{ anomaly_type: "normal" }};
        const source = selectedTrackScoreComponents(track).used_sources || "";
        const kindLabel = {{
          route: t("submoduleRoute"),
          speed: t("submoduleSpeed"),
          shape: t("submoduleShape"),
        }};
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
        `;
      }}

      function renderCompositionBars(track) {{
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
              <span class="decomp-track"><span class="decomp-fill" style="width: ${{Math.max(0, Math.min(100, Math.round((Number(value || 0) / max) * 100))}}%;"></span></span>
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

      function renderEventTimeline(track) {{
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
          if (used > 0 && eventFrames.length) {{
            eventFrames.forEach(item => {{
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
        `;
      }}

      function renderProtocolOverview() {{
        const labels = {{ individual: t("taskIndividual"), group: t("taskGroup") }};
        for (const [taskName, target] of [["individual", individualProtocol], ["group", groupProtocol]]) {{
          const task = dashboard.tasks[taskName];
          if (!task || !target) {{
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
          groupInsightPanel.textContent = state.task === "group" ? t("noTrackSelected") : (state.language === "zh" ? "切换到 Group 任务可查看群体中心、半径和邻近关系。" : "Switch to Group to inspect centroid, radius, and neighborhood relations.");
          return;
        }}
        const label = trackLabel(track);
        const stats = trajectoryStats(track);
        const groupStats = groupFrameStats(data, track);
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
        `;
        if (state.task === "group") {{
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
        const labels = {{ individual: t("taskIndividual"), group: t("taskGroup") }};
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
        cards.innerHTML = [
          [t("cardMethods"), Object.keys(task.methods).length],
          [t("cardLabels"), task.num_labels],
          [t("cardPositives"), task.num_positive],
          [t("cardAuroc"), fmt(metrics.auroc)]
        ].map(([label, value]) => `<div class="card"><div>${{label}}</div><div class="value">${{value}}</div></div>`).join("");
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
        const labels = {{
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

      function ensureBackground(data, frame) {{
        const background = backgroundForFrame(data, frame);
        if (!background || !background.src) {{
          state.image = null;
          state.imageKey = null;
          return;
        }}
        const key = `${{data.sequence}}:${{background.src}}`;
        if (state.imageKey === key) {{
          return;
        }}
        state.imageKey = key;
        if (backgroundCache.has(key)) {{
          state.image = backgroundCache.get(key);
          return;
        }}
        const image = new Image();
        image.onload = () => {{
          backgroundCache.set(key, image);
          if (state.imageKey === key) {{
            state.image = image;
            drawPlayback();
          }}
        }};
        image.onerror = () => {{
          if (state.imageKey === key) {{
            state.image = null;
            drawPlayback();
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

      function setViewModeVisibility() {{
        const comparison = state.viewMode === "comparison";
        comparisonView.hidden = !comparison;
        singleView.hidden = comparison;
        singleLayerSwitch.hidden = comparison;
        viewModeButtons.forEach(button => {{
          button.classList.toggle("active", button.dataset.viewMode === state.viewMode);
        }});
      }}

      function drawCanvasBase(targetCtx, targetCanvas, data, layer) {{
        targetCtx.fillStyle = "#e2e8f0";
        targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
        if (state.image) {{
          targetCtx.drawImage(state.image, 0, 0, targetCanvas.width, targetCanvas.height);
        }}
        if (layer === "heatmap" || layer === "both") {{
          targetCtx.save();
          targetCtx.fillStyle = layer === "heatmap" ? "rgba(4, 9, 18, 0.18)" : "rgba(4, 9, 18, 0.06)";
          targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
          targetCtx.restore();
        }}
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
        setViewModeVisibility();
        const names = sequencesForTask();
        if (!names.length) {{
          playbackReadout.textContent = t("noPlayback");
          clearPlaybackCanvases();
          return;
        }}
        const data = currentPlayback();
        if (!data) {{
          return;
        }}
        resetFrameForSequence();
        ensureBackground(data, state.frame);
        const scores = data.tracks.map(track => trackScore(track));
        const maxScore = Math.max(...scores, 1e-6);
        const ranked = rankedTracks(data);
        ensureSelectedTrack(data, ranked);
        if (state.viewMode === "comparison") {{
          drawComparisonView(data, ranked, maxScore);
        }} else {{
          drawSingleView(data, ranked, maxScore);
        }}
        renderTrackInsights(ranked);
        const viewLabel = state.viewMode === "comparison" ? t("view_comparison") : `${{t("view_single")}} - ${{t(`layer_${{state.layer}}`)}}`;
        playbackReadout.textContent = `${{t("playbackPrefix")}} / ${{data.sequence}} / ${{state.method}} / ${{t("frame")}} ${{state.frame}} / ${{viewLabel}} / ${{ranked.length}} ${{t("visibleTracks")}}`;
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

      function startPlayback() {{
        if (!currentPlayback()) {{
          return;
        }}
        state.playing = true;
        playToggle.textContent = t("pause");
        playToggle.classList.add("active");
        state.timer = window.setInterval(() => {{
          const data = currentPlayback();
          const start = Number(data.frame_range?.[0] || 0);
          const end = Number(data.frame_range?.[1] || start);
          state.frame = state.frame >= end ? start : state.frame + 1;
          frameSlider.value = state.frame;
          drawPlayback();
        }}, 90);
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
        state.sequence = "";
        state.image = null;
        state.imageKey = null;
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
      viewModeButtons.forEach(button => button.addEventListener("click", () => {{
        state.viewMode = button.dataset.viewMode || "comparison";
        setViewModeVisibility();
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
      setAnalysisPanel("leaderboard");
      renderMethodView();
    }})();
  </script>
</body>
</html>
"""
