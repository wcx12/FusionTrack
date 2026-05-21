from __future__ import annotations

import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_scores(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def _positive_ground_truth(labels_by_sample: dict[str, list[dict[str, Any]]], sample_id: str) -> list[dict[str, Any]]:
    labels = []
    for row in labels_by_sample.get(sample_id, []):
        if int(row.get("label", 0) or 0) != 1:
            continue
        labels.append(
            {
                "frame_start": int(row.get("frame_start", 0) or 0),
                "frame_end": int(row.get("frame_end", 0) or 0),
                "label": 1,
                "anomaly_type": str(row.get("anomaly_type", "")),
                "injection_seed": int(row.get("injection_seed", 0) or 0),
            }
        )
    return labels


def _metric_label(key: str) -> str:
    labels = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "f1": "F1",
        "precision_at_k": "Precision@K",
        "recall_at_k": "Recall@K",
    }
    return labels.get(key, key.replace("_", " ").title())


def _format_metric_value(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _compact_experiment_context(experiment_context: dict[str, Any]) -> dict[str, Any]:
    if not experiment_context:
        return {}
    return {
        "method_name": experiment_context.get("method_name", ""),
        "task": experiment_context.get("task", ""),
        "split": experiment_context.get("split", ""),
        "seed": experiment_context.get("seed"),
        "metrics": experiment_context.get("metrics", {}) or {},
        "summary": experiment_context.get("summary", {}) or {},
    }


def _score_for(trajectory: dict[str, Any], scores: dict[str, dict[str, Any]]) -> float:
    return float(scores.get(trajectory["sample_id"], {}).get("score", 0.0))


def _trajectory_points(trajectory: dict[str, Any]) -> list[tuple[float, float]]:
    points = []
    for point in trajectory.get("points", []):
        fused = point.get("fused")
        if fused and fused.get("center_xy"):
            points.append((float(fused["center_xy"][0]), float(fused["center_xy"][1])))
    return points


def _trajectory_frame_points(trajectory: dict[str, Any]) -> list[tuple[int, float, float]]:
    points = []
    for point in trajectory.get("points", []):
        fused = point.get("fused")
        if fused and fused.get("center_xy") and point.get("frame_id") is not None:
            points.append((int(point["frame_id"]), float(fused["center_xy"][0]), float(fused["center_xy"][1])))
    return points


def _trajectory_confidence(trajectory: dict[str, Any]) -> float:
    values = [
        float(point["fused"].get("confidence", 0.0))
        for point in trajectory.get("points", [])
        if point.get("fused")
    ]
    return sum(values) / len(values) if values else 0.0


def _find_background_image(data_root: Path, trajectories: list[dict[str, Any]]) -> Path | None:
    for trajectory in trajectories:
        for point in trajectory.get("points", []):
            rgb = point.get("rgb")
            if not rgb:
                continue
            rgb_file = rgb.get("file")
            if not rgb_file:
                continue
            relative = Path(str(rgb_file))
            for candidate in (
                data_root / relative,
                data_root / "test2017" / relative,
                data_root / "train2017" / relative,
            ):
                if candidate.exists():
                    return candidate
    return None


def _resolve_rgb_file(data_root: Path, rgb_file: str) -> Path | None:
    relative = Path(str(rgb_file))
    for candidate in (
        data_root / relative,
        data_root / "test2017" / relative,
        data_root / "train2017" / relative,
    ):
        if candidate.exists():
            return candidate
    return None


def _background_frame_candidates(
    data_root: Path,
    trajectories: list[dict[str, Any]],
) -> list[tuple[int, Path]]:
    by_frame: dict[int, Path] = {}
    for trajectory in trajectories:
        for point in trajectory.get("points", []):
            frame_id = point.get("frame_id")
            rgb = point.get("rgb")
            if frame_id is None or not rgb or not rgb.get("file"):
                continue
            image_path = _resolve_rgb_file(data_root, str(rgb["file"]))
            if image_path is not None and int(frame_id) not in by_frame:
                by_frame[int(frame_id)] = image_path
    return sorted(by_frame.items(), key=lambda item: item[0])


def _select_background_frames(
    candidates: list[tuple[int, Path]],
    max_frames: int = 72,
) -> list[tuple[int, Path]]:
    if len(candidates) <= max_frames:
        return candidates
    if max_frames <= 1:
        return [candidates[0]]
    selected_indices = {
        round(index * (len(candidates) - 1) / (max_frames - 1))
        for index in range(max_frames)
    }
    return [candidates[index] for index in sorted(selected_indices)]


def _apply_image_coordinate_axes(ax: Any, background_size: tuple[int, int] | None) -> None:
    if background_size is not None:
        width, height = background_size
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    else:
        ax.invert_yaxis()


def _try_draw_background(ax: Any, data_root: Path, trajectories: list[dict[str, Any]]) -> tuple[int, int] | None:
    background = _find_background_image(data_root, trajectories)
    if background is None:
        return None
    try:
        image = plt.imread(background)
        height, width = image.shape[:2]
        ax.imshow(image, extent=(0, width, height, 0), origin="upper")
        return (int(width), int(height))
    except Exception:
        return None


def _image_size(image_path: Path) -> tuple[int, int] | None:
    try:
        image = plt.imread(image_path)
        height, width = image.shape[:2]
        return int(width), int(height)
    except Exception:
        return None


def _copy_background_asset(
    sequence: str,
    trajectories: list[dict[str, Any]],
    data_root: Path,
    assets_dir: Path,
) -> tuple[Path | None, tuple[int, int] | None, list[dict[str, Any]]]:
    candidates = _background_frame_candidates(data_root, trajectories)
    background = candidates[0][1] if candidates else _find_background_image(data_root, trajectories)
    if background is None:
        return None, None, []
    suffix = background.suffix.lower() or ".jpg"
    if suffix == ".jpeg":
        suffix = ".jpg"
    output_path = assets_dir / f"background_{_safe_name(sequence)}{suffix}"
    shutil.copy2(background, output_path)
    frame_assets = []
    for frame_id, frame_path in _select_background_frames(candidates):
        frame_suffix = frame_path.suffix.lower() or ".jpg"
        if frame_suffix == ".jpeg":
            frame_suffix = ".jpg"
        frame_output = assets_dir / f"background_{_safe_name(sequence)}_{frame_id:06d}{frame_suffix}"
        shutil.copy2(frame_path, frame_output)
        frame_assets.append({"frame": frame_id, "path": frame_output})
    return output_path, _image_size(output_path), frame_assets


def _fallback_scene_size(trajectories: list[dict[str, Any]]) -> tuple[int, int]:
    xs: list[float] = []
    ys: list[float] = []
    for trajectory in trajectories:
        for _, x, y in _trajectory_frame_points(trajectory):
            xs.append(x)
            ys.append(y)
    width = max(int(max(xs, default=640) + 32), 640)
    height = max(int(max(ys, default=512) + 32), 512)
    return width, height


def _build_playback_payload(
    sequence: str,
    trajectories: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    background_asset: Path | None,
    background_frames: list[dict[str, Any]],
    scene_size: tuple[int, int] | None,
    labels_by_sample: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    labels_by_sample = labels_by_sample or {}
    ranked = sorted(trajectories, key=lambda item: _score_for(item, scores), reverse=True)
    frame_ids: list[int] = []
    tracks = []
    frame_mass: dict[int, float] = defaultdict(float)
    ground_truth_segments = []
    max_score = max([_score_for(item, scores) for item in ranked] or [0.0])
    for trajectory in ranked[:80]:
        points = []
        frame_points = _trajectory_frame_points(trajectory)
        if not frame_points:
            continue
        score = _score_for(trajectory, scores)
        per_point = score / max(len(frame_points), 1)
        for frame_id, x, y in frame_points:
            frame_ids.append(frame_id)
            frame_mass[frame_id] += per_point
            fused_point = next(
                (
                    point.get("fused", {})
                    for point in trajectory.get("points", [])
                    if point.get("frame_id") == frame_id and point.get("fused")
                ),
                {},
            )
            points.append(
                {
                    "frame": frame_id,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "confidence": round(float(fused_point.get("confidence", 0.0)), 4),
                }
            )
        row = scores.get(trajectory["sample_id"], {})
        confidences = [point["confidence"] for point in points]
        modal_offsets = [item[1] for item in _modal_offsets(trajectory)]
        path_length = 0.0
        speeds = []
        for previous, current in zip(frame_points, frame_points[1:]):
            previous_frame, previous_x, previous_y = previous
            current_frame, current_x, current_y = current
            distance = ((current_x - previous_x) ** 2 + (current_y - previous_y) ** 2) ** 0.5
            path_length += distance
            frame_delta = max(current_frame - previous_frame, 1)
            speeds.append(distance / frame_delta)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        confidence_drop = max(confidences) - min(confidences) if confidences else 0.0
        max_modal_offset = max(modal_offsets) if modal_offsets else 0.0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        source = row.get("used_sources", "")
        source_coverage = 1.0 if "individual" in source and "group" in source else 0.58 if source else 0.0
        reason_breakdown = [
            {"key": "score", "label": "Overall score", "value": round(score / max(max_score, 1e-6), 6)},
            {"key": "source", "label": "Source coverage", "value": round(source_coverage, 6)},
            {"key": "motion", "label": "Motion burst", "value": round(max_speed / (max_speed + 6.0), 6) if max_speed else 0.0},
            {
                "key": "modal",
                "label": "Modal offset",
                "value": round(max_modal_offset / (max_modal_offset + 8.0), 6) if max_modal_offset else 0.0,
            },
            {"key": "confidence", "label": "Confidence drop", "value": round(min(confidence_drop * 2.0, 1.0), 6)},
        ]
        ground_truth = _positive_ground_truth(labels_by_sample, str(trajectory["sample_id"]))
        for label in ground_truth:
            ground_truth_segments.append(
                {
                    "sample_id": str(trajectory["sample_id"]),
                    "track_id": str(trajectory["track_id"]),
                    "frame_start": label["frame_start"],
                    "frame_end": label["frame_end"],
                    "anomaly_type": label["anomaly_type"],
                }
            )
        tracks.append(
            {
                "sequence": sequence,
                "sample_id": trajectory["sample_id"],
                "track_id": str(trajectory["track_id"]),
                "category": trajectory.get("category_name", "") or "",
                "score": round(score, 6),
                "score_ratio": round(score / max(max_score, 1e-6), 6),
                "source": row.get("used_sources", ""),
                "first_frame": frame_points[0][0],
                "last_frame": frame_points[-1][0],
                "duration": frame_points[-1][0] - frame_points[0][0] + 1,
                "avg_confidence": round(avg_confidence, 6),
                "confidence_drop": round(confidence_drop, 6),
                "max_modal_offset": round(max_modal_offset, 6),
                "path_length": round(path_length, 6),
                "avg_speed": round(avg_speed, 6),
                "max_speed": round(max_speed, 6),
                "reason_breakdown": reason_breakdown,
                "ground_truth": ground_truth,
                "points": points,
            }
        )
    width, height = scene_size or _fallback_scene_size(trajectories)
    frame_start = min(frame_ids) if frame_ids else 0
    frame_end = max(frame_ids) if frame_ids else 0
    timeline = [
        {"frame": frame_id, "mass": round(frame_mass[frame_id], 6)}
        for frame_id in sorted(frame_mass)
    ]
    return {
        "sequence": sequence,
        "background": f"assets/{background_asset.name}" if background_asset is not None else None,
        "background_frames": [
            {"frame": int(item["frame"]), "src": f"assets/{item['path'].name}"}
            for item in background_frames
        ],
        "size": {"width": width, "height": height},
        "frame_range": [frame_start, frame_end],
        "tracks": tracks,
        "ground_truth_segments": ground_truth_segments,
        "timeline": timeline,
    }


def _write_playback_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")


def _draw_sequence_plot(
    sequence: str,
    trajectories: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    output_path: Path,
    data_root: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    background_size = _try_draw_background(ax, data_root, trajectories)
    max_score = max([_score_for(item, scores) for item in trajectories] or [1.0])
    max_score = max(max_score, 1e-6)
    ranked = sorted(trajectories, key=lambda item: _score_for(item, scores), reverse=True)
    for rank, trajectory in enumerate(ranked):
        points = _trajectory_points(trajectory)
        if len(points) < 2:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        score = _score_for(trajectory, scores)
        color_value = min(score / max_score, 1.0)
        ax.plot(xs, ys, linewidth=1.0 + 3.5 * color_value, alpha=0.72)
        ax.scatter(xs[-1], ys[-1], s=18 + 90 * color_value, alpha=0.86)
        if rank < 12:
            ax.text(xs[-1], ys[-1], str(trajectory["track_id"]), fontsize=7)
    ax.set_title(f"{sequence} fused trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _apply_image_coordinate_axes(ax, background_size)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_heatmap(
    sequence: str,
    trajectories: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    output_path: Path,
    data_root: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    background_size = _try_draw_background(ax, data_root, trajectories)
    xs: list[float] = []
    ys: list[float] = []
    weights: list[float] = []
    for trajectory in trajectories:
        points = _trajectory_points(trajectory)
        if not points:
            continue
        weight = _score_for(trajectory, scores) / max(len(points), 1)
        for x, y in points:
            xs.append(x)
            ys.append(y)
            weights.append(weight)
    if xs:
        histogram_range = None
        if background_size is not None:
            width, height = background_size
            histogram_range = [[0, width], [0, height]]
        mass, x_edges, y_edges = np.histogram2d(xs, ys, bins=48, weights=weights, range=histogram_range)
        positive = mass[mass > 0]
        if positive.size:
            visible_mass = np.ma.masked_where(mass.T <= 0, mass.T)
            if positive.size > 12:
                visible_mass = np.ma.masked_less(visible_mass, float(np.quantile(positive, 0.35)))
            cmap = plt.get_cmap("inferno").copy()
            cmap.set_bad((0, 0, 0, 0))
            heat = ax.pcolormesh(x_edges, y_edges, visible_mass, cmap=cmap, alpha=0.72, shading="auto")
            fig.colorbar(heat, ax=ax, fraction=0.035, pad=0.02, label="anomaly mass")
    ax.set_title(f"{sequence} anomaly heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _apply_image_coordinate_axes(ax, background_size)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_timeline(
    sequence: str,
    trajectories: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    frame_mass: dict[int, float] = defaultdict(float)
    for trajectory in trajectories:
        frame_points = _trajectory_frame_points(trajectory)
        if not frame_points:
            continue
        per_point = _score_for(trajectory, scores) / max(len(frame_points), 1)
        for frame_id, _, _ in frame_points:
            frame_mass[frame_id] += per_point
    fig, ax = plt.subplots(figsize=(10, 3.5))
    if frame_mass:
        frames = sorted(frame_mass)
        values = [frame_mass[frame] for frame in frames]
        ax.plot(frames, values, color="#2563eb", linewidth=1.6)
        ax.fill_between(frames, values, color="#93c5fd", alpha=0.45)
    ax.set_title(f"{sequence} anomaly timeline")
    ax.set_xlabel("frame")
    ax.set_ylabel("anomaly mass")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _modal_offsets(trajectory: dict[str, Any]) -> list[tuple[int, float, float]]:
    values = []
    for point in trajectory.get("points", []):
        frame_id = point.get("frame_id")
        if frame_id is None:
            continue
        fused = point.get("fused") or {}
        confidence = float(fused.get("confidence", 0.0))
        offset = None
        if point.get("modal"):
            offset = point["modal"].get("offset_distance")
        if offset is None:
            offset = fused.get("component_scores", {}).get("modal_offset_distance")
        if offset is not None:
            values.append((int(frame_id), float(offset), confidence))
    return values


def _draw_modal_consistency(
    sequence: str,
    trajectories: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    ranked = sorted(trajectories, key=lambda item: _score_for(item, scores), reverse=True)[:8]
    fig, ax1 = plt.subplots(figsize=(10, 4.2))
    ax2 = ax1.twinx()
    has_modal_line = False
    for trajectory in ranked:
        values = _modal_offsets(trajectory)
        if not values:
            continue
        frames = [item[0] for item in values]
        offsets = [item[1] for item in values]
        confidences = [item[2] for item in values]
        label = str(trajectory["track_id"])
        ax1.plot(frames, offsets, linewidth=1.4, alpha=0.75, label=label)
        ax2.plot(frames, confidences, linewidth=1.0, alpha=0.25, linestyle="--")
        has_modal_line = True
    ax1.set_title(f"{sequence} modal consistency")
    ax1.set_xlabel("frame")
    ax1.set_ylabel("RGB/Thermal offset")
    ax2.set_ylabel("fusion confidence")
    ax1.grid(True, alpha=0.25)
    if has_modal_line:
        ax1.legend(fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_visual_report(
    fused_jsonl: str | Path,
    final_scores_csv: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    top_sequences: int = 5,
    experiment_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fused_jsonl = Path(fused_jsonl)
    final_scores_csv = Path(final_scores_csv)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    trajectories = _load_jsonl(fused_jsonl)
    scores = _load_scores(final_scores_csv)
    experiment_context = experiment_context or {}
    labels_by_sample = experiment_context.get("labels_by_sample", {}) or {}
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_sequence[str(trajectory["sequence"])].append(trajectory)

    scored_rows = sorted(scores.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)
    sequence_mass = []
    for sequence, items in by_sequence.items():
        mass = sum(_score_for(item, scores) for item in items)
        sequence_mass.append((sequence, mass))
    sequence_mass.sort(key=lambda item: item[1], reverse=True)
    selected_sequences: list[str] = []

    def include_sequence(sequence: str) -> None:
        if sequence in by_sequence and sequence not in selected_sequences:
            selected_sequences.append(sequence)

    for sequence, _ in sequence_mass[:top_sequences]:
        include_sequence(sequence)
    for row in scored_rows[:10]:
        include_sequence(str(row.get("sequence", "")))

    sequence_assets = []
    playback_payloads: dict[str, dict[str, Any]] = {}
    for sequence in selected_sequences:
        safe_sequence = _safe_name(sequence)
        trajectory_path = assets_dir / f"trajectory_{safe_sequence}.png"
        heatmap_path = assets_dir / f"heatmap_{safe_sequence}.png"
        timeline_path = assets_dir / f"timeline_{safe_sequence}.png"
        modal_path = assets_dir / f"modal_{safe_sequence}.png"
        playback_path = assets_dir / f"playback_{safe_sequence}.json"
        background_asset, background_size, background_frames = _copy_background_asset(
            sequence,
            by_sequence[sequence],
            data_root,
            assets_dir,
        )
        playback_payload = _build_playback_payload(
            sequence,
            by_sequence[sequence],
            scores,
            background_asset,
            background_frames,
            background_size,
            labels_by_sample,
        )
        _write_playback_payload(playback_payload, playback_path)
        playback_payloads[sequence] = playback_payload
        _draw_sequence_plot(sequence, by_sequence[sequence], scores, trajectory_path, data_root)
        _draw_heatmap(sequence, by_sequence[sequence], scores, heatmap_path, data_root)
        _draw_timeline(sequence, by_sequence[sequence], scores, timeline_path)
        _draw_modal_consistency(sequence, by_sequence[sequence], scores, modal_path)
        sequence_assets.append(
            {
                "sequence": sequence,
                "trajectory": trajectory_path,
                "heatmap": heatmap_path,
                "timeline": timeline_path,
                "modal": modal_path,
                "playback": playback_path,
                "background": background_asset,
            }
        )

    avg_confidence = (
        sum(_trajectory_confidence(trajectory) for trajectory in trajectories) / len(trajectories)
        if trajectories
        else 0.0
    )
    max_score = max([float(row.get("score", 0.0)) for row in scored_rows] or [0.0])
    trajectory_by_sample = {trajectory["sample_id"]: trajectory for trajectory in trajectories}
    playback_data_json = json.dumps(playback_payloads, ensure_ascii=True).replace("</", "<\\/")
    experiment_metrics = experiment_context.get("metrics", {}) or {}
    metric_keys = ["auroc", "auprc", "f1", "precision_at_k", "recall_at_k"]
    metric_items_html = "\n".join(
        f"<div class=\"experiment-metric\"><span>{html.escape(_metric_label(key))}</span>"
        f"<strong>{html.escape(_format_metric_value(experiment_metrics[key]))}</strong></div>"
        for key in metric_keys
        if key in experiment_metrics
    )
    experiment_panel_html = ""
    if experiment_context:
        method_name = str(experiment_context.get("method_name", "unknown"))
        task_name = str(experiment_context.get("task", ""))
        split_name = str(experiment_context.get("split", ""))
        seed = experiment_context.get("seed")
        summary = experiment_context.get("summary", {}) or {}
        experiment_panel_html = (
            "<section class=\"experiment-panel\">"
            "<h2>Experiment</h2>"
            f"<div class=\"experiment-name\">{html.escape(method_name)}</div>"
            f"<div class=\"experiment-meta\">{html.escape(task_name)} | {html.escape(split_name)}"
            f"{' | seed ' + html.escape(str(seed)) if seed is not None else ''}</div>"
            f"<div class=\"experiment-meta\">GT positives: {html.escape(str(summary.get('num_positive_labels', 0)))}</div>"
            f"<div class=\"experiment-metrics\">{metric_items_html}</div>"
            "</section>"
        )

    def first_frame_for(row: dict[str, Any]) -> int:
        trajectory = trajectory_by_sample.get(row.get("sample_id", ""))
        frame_points = _trajectory_frame_points(trajectory) if trajectory else []
        return frame_points[0][0] if frame_points else 0

    def label_text_for(row: dict[str, Any]) -> str:
        labels = _positive_ground_truth(labels_by_sample, row.get("sample_id", ""))
        return " ".join(label["anomaly_type"] for label in labels if label.get("anomaly_type"))

    def target_attrs(row: dict[str, Any]) -> str:
        search_text = " ".join(
            [
                row.get("sample_id", ""),
                row.get("sequence", ""),
                row.get("track_id", ""),
                row.get("category_name", "") or "",
                row.get("used_sources", ""),
                label_text_for(row),
            ]
        ).lower()
        return (
            f'data-sequence="{html.escape(row.get("sequence", ""))}" '
            f'data-sample="{html.escape(row.get("sample_id", ""))}" '
            f'data-frame="{first_frame_for(row)}" '
            f'data-score="{float(row.get("score", 0.0)):.6f}" '
            f'data-source="{html.escape(row.get("used_sources", ""))}" '
            f'data-search="{html.escape(search_text)}"'
        )

    top_cards_html = "\n".join(
        f"<button type=\"button\" class=\"target-card target-item\" {target_attrs(row)}>"
        f"<span class=\"score\">{float(row.get('score', 0.0)):.1f}</span>"
        "<span class=\"target-copy\">"
        f"<strong>{html.escape(row.get('sequence', ''))} / {html.escape(row.get('track_id', ''))}</strong>"
        f"<small>{html.escape(row.get('category_name', '') or '')} | {html.escape(row.get('used_sources', ''))}"
        f"{' | GT: ' + html.escape(label_text_for(row)) if label_text_for(row) else ''}</small>"
        "</span>"
        "</button>"
        for row in scored_rows[:10]
    )
    sequence_tabs_html = "\n".join(
        (
            f"<button type=\"button\" class=\"tab-button{' active' if index == 0 else ''}\" "
            f"data-sequence=\"{html.escape(asset['sequence'])}\">{html.escape(asset['sequence'])}</button>"
        )
        for index, asset in enumerate(sequence_assets)
    )
    plots_html = "\n".join(
        f"<section class=\"sequence-block evidence-block\" data-sequence=\"{html.escape(asset['sequence'])}\""
        f"{'' if index == 0 else ' hidden'}>"
        f"<h2>{html.escape(asset['sequence'])} evidence</h2>"
        "<div class=\"evidence-tabs\" role=\"tablist\">"
        "<button type=\"button\" class=\"evidence-tab active\" data-tab=\"trajectory\">Trajectory</button>"
        "<button type=\"button\" class=\"evidence-tab\" data-tab=\"heatmap\">Heatmap</button>"
        "<button type=\"button\" class=\"evidence-tab\" data-tab=\"modal\">Modal consistency</button>"
        "</div>"
        "<div class=\"figure-grid evidence-panels\">"
        f"<figure data-evidence-panel=\"trajectory\"><img class=\"plot-image\" data-title=\"{html.escape(asset['sequence'])} - Fused trajectories\" src=\"assets/{html.escape(asset['trajectory'].name)}\" alt=\"trajectory\"><figcaption>Fused trajectories</figcaption></figure>"
        f"<figure data-evidence-panel=\"heatmap\" hidden><img class=\"plot-image\" data-title=\"{html.escape(asset['sequence'])} - Heatmap\" src=\"assets/{html.escape(asset['heatmap'].name)}\" alt=\"heatmap\"><figcaption>Heatmap</figcaption></figure>"
        f"<figure data-evidence-panel=\"modal\" hidden><img class=\"plot-image\" data-title=\"{html.escape(asset['sequence'])} - Modal consistency\" src=\"assets/{html.escape(asset['modal'].name)}\" alt=\"modal consistency\"><figcaption>Modal consistency</figcaption></figure>"
        "</div></section>"
        for index, asset in enumerate(sequence_assets)
    )
    script_text = """
  <script>
    (() => {
      const playbackData = JSON.parse(document.getElementById("playbackData").textContent);
      const search = document.getElementById("targetSearch");
      const minScore = document.getElementById("minScore");
      const minScoreValue = document.getElementById("minScoreValue");
      const targetItems = Array.from(document.querySelectorAll(".target-item"));
      const sequenceBlocks = Array.from(document.querySelectorAll(".sequence-block"));
      const tabs = Array.from(document.querySelectorAll(".tab-button"));
      const canvas = document.getElementById("playbackCanvas");
      const ctx = canvas.getContext("2d");
      const playPause = document.getElementById("playPause");
      const frameScrubber = document.getElementById("frameScrubber");
      const frameBadge = document.getElementById("frameBadge");
      const speedSelect = document.getElementById("speedSelect");
      const trackReadout = document.getElementById("trackReadout");
      const targetDetail = document.getElementById("targetDetail");
      const demoMode = document.getElementById("demoMode");
      const autoTour = document.getElementById("autoTour");
      const evidenceTabs = Array.from(document.querySelectorAll(".evidence-tab"));
      const lightbox = document.getElementById("lightbox");
      const lightboxImage = document.getElementById("lightboxImage");
      const lightboxTitle = document.getElementById("lightboxTitle");
      const state = {
        sequence: Object.keys(playbackData)[0],
        frame: 0,
        playing: false,
        speed: 1,
        focusSample: null,
        image: null,
        imageKey: null,
        imageSequence: null,
        lastTick: 0,
        touring: false,
        tourTimer: null
      };
      const backgroundCache = new Map();

      function currentData() {
        return playbackData[state.sequence];
      }

      function escapeHtml(value) {
        return String(value ?? "").replace(/[&<>"']/g, (character) => ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;"
        })[character]);
      }

      function findTrack(data, sampleId) {
        return data.tracks.find((track) => track.sample_id === sampleId) || null;
      }

      function renderReasonBreakdown(track) {
        const rows = (track.reason_breakdown || []).map((item) => {
          const value = Math.max(0, Math.min(1, Number(item.value || 0)));
          const percent = Math.round(value * 100);
          return `
            <div class="reason-row">
              <div class="reason-copy"><span>${escapeHtml(item.label)}</span><strong>${percent}%</strong></div>
              <div class="reason-bar"><span style="width: ${percent}%"></span></div>
            </div>
          `;
        }).join("");
        return `
          <div class="reason-panel">
            <div class="reason-title">Anomaly explanation</div>
            ${rows}
          </div>
        `;
      }

      function renderGroundTruth(track) {
        const labels = track.ground_truth || [];
        if (!labels.length) {
          return "";
        }
        const rows = labels.map((label) => `
          <div class="gt-item">
            <span>${escapeHtml(label.anomaly_type || "anomaly")}</span>
            <strong>${escapeHtml(label.frame_start)}-${escapeHtml(label.frame_end)}</strong>
          </div>
        `).join("");
        return `
          <div class="gt-panel">
            <div class="reason-title">GT anomaly</div>
            <div class="gt-list">${rows}</div>
          </div>
        `;
      }

      function renderTargetDetail(track, data) {
        if (!track) {
          targetDetail.innerHTML = `
            <div class="detail-title">${escapeHtml(data.sequence)}</div>
            <div class="detail-subtitle">No target selected</div>
          `;
          return;
        }
        targetDetail.innerHTML = `
          <div class="detail-title">${escapeHtml(track.sequence)} / track ${escapeHtml(track.track_id)}</div>
          <div class="detail-subtitle">${escapeHtml(track.category || "unknown")} | ${escapeHtml(track.source || "score")}</div>
          <div class="detail-grid">
            <div class="metric"><span>Score</span><strong>${Number(track.score).toFixed(2)}</strong></div>
            <div class="metric"><span>Frames</span><strong>${track.first_frame}-${track.last_frame}</strong></div>
            <div class="metric"><span>Duration</span><strong>${track.duration}</strong></div>
            <div class="metric"><span>Avg confidence</span><strong>${Number(track.avg_confidence).toFixed(3)}</strong></div>
            <div class="metric"><span>Max modal offset</span><strong>${Number(track.max_modal_offset).toFixed(2)}</strong></div>
            <div class="metric"><span>Max speed</span><strong>${Number(track.max_speed).toFixed(2)}</strong></div>
          </div>
          ${renderReasonBreakdown(track)}
          ${renderGroundTruth(track)}
        `;
      }

      function colorFor(track) {
        const ratio = Math.max(0, Math.min(1, Number(track.score_ratio || 0)));
        const hue = 42 - ratio * 38;
        return `hsl(${hue}, 96%, 54%)`;
      }

      function visibleTracks(data) {
        const threshold = Number(minScore.value || 0);
        return data.tracks.filter((track) => Number(track.score || 0) >= threshold);
      }

      function pointsUntil(track, frame) {
        const points = [];
        for (const point of track.points) {
          if (point.frame <= frame) {
            points.push(point);
          } else {
            break;
          }
        }
        return points;
      }

      function backgroundForFrame(data, frame) {
        const frames = data.background_frames || [];
        if (!frames.length) {
          return data.background ? { frame: data.frame_range[0], src: data.background } : null;
        }
        let selected = frames[0];
        for (const item of frames) {
          if (Number(item.frame) <= frame) {
            selected = item;
          } else {
            break;
          }
        }
        return selected;
      }

      function ensureBackgroundForFrame(data) {
        const background = backgroundForFrame(data, state.frame);
        if (!background || !background.src) {
          state.image = null;
          state.imageKey = null;
          return;
        }
        const key = `${data.sequence}:${background.src}`;
        if (state.imageKey === key) {
          return;
        }
        state.imageKey = key;
        state.imageSequence = data.sequence;
        if (backgroundCache.has(key)) {
          state.image = backgroundCache.get(key);
          return;
        }
        const image = new Image();
        image.onload = () => {
          backgroundCache.set(key, image);
          if (state.imageSequence === data.sequence && state.imageKey === key) {
            state.image = image;
            drawPlayback();
          }
        };
        image.src = background.src;
      }

      function setCanvasFor(data) {
        canvas.width = data.size.width;
        canvas.height = data.size.height + 72;
        frameScrubber.min = data.frame_range[0];
        frameScrubber.max = data.frame_range[1];
        frameScrubber.value = state.frame;
      }

      function timelineBounds(data) {
        const [start, end] = data.frame_range;
        return {
          x: 12,
          y: data.size.height + 20,
          width: data.size.width - 24,
          height: 34,
          start,
          end,
          span: Math.max(end - start, 1)
        };
      }

      function drawTimeline(data) {
        const bounds = timelineBounds(data);
        const width = data.size.width;
        const maxMass = Math.max(...data.timeline.map((item) => item.mass), 1);
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(0, data.size.height, width, 72);
        ctx.strokeStyle = "#cbd5e1";
        ctx.strokeRect(bounds.x, bounds.y, bounds.width, bounds.height);
        drawGroundTruthSegments(data, bounds);
        ctx.fillStyle = "rgba(37, 99, 235, 0.42)";
        for (const item of data.timeline) {
          const x = bounds.x + ((item.frame - bounds.start) / bounds.span) * bounds.width;
          const h = Math.max(1, (item.mass / maxMass) * bounds.height);
          ctx.fillRect(x, bounds.y + bounds.height - h, 2, h);
        }
        const playX = bounds.x + ((state.frame - bounds.start) / bounds.span) * bounds.width;
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playX, bounds.y - 4);
        ctx.lineTo(playX, bounds.y + bounds.height + 4);
        ctx.stroke();
      }

      function drawGroundTruthSegments(data, bounds) {
        const segments = data.ground_truth_segments || [];
        if (!segments.length) {
          return;
        }
        ctx.fillStyle = "rgba(239, 68, 68, 0.18)";
        ctx.strokeStyle = "rgba(185, 28, 28, 0.52)";
        ctx.lineWidth = 1;
        for (const segment of segments) {
          const start = Math.max(bounds.start, Number(segment.frame_start || bounds.start));
          const end = Math.min(bounds.end, Number(segment.frame_end || bounds.end));
          if (end < bounds.start || start > bounds.end) {
            continue;
          }
          const x0 = bounds.x + ((start - bounds.start) / bounds.span) * bounds.width;
          const x1 = bounds.x + ((end - bounds.start) / bounds.span) * bounds.width;
          const width = Math.max(2, x1 - x0);
          ctx.fillRect(x0, bounds.y, width, bounds.height);
          ctx.strokeRect(x0, bounds.y, width, bounds.height);
        }
      }

      function drawPlayback() {
        const data = currentData();
        if (!data) {
          return;
        }
        ensureBackgroundForFrame(data);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#e2e8f0";
        ctx.fillRect(0, 0, data.size.width, data.size.height);
        if (state.image && state.imageSequence === data.sequence) {
          ctx.drawImage(state.image, 0, 0, data.size.width, data.size.height);
        }
        const tracks = visibleTracks(data);
        for (const track of tracks) {
          const points = pointsUntil(track, state.frame);
          if (!points.length) {
            continue;
          }
          const focused = state.focusSample === track.sample_id;
          const color = focused ? "#ef4444" : colorFor(track);
          ctx.strokeStyle = color;
          ctx.globalAlpha = focused ? 0.95 : 0.42 + 0.38 * Number(track.score_ratio || 0);
          ctx.lineWidth = focused ? 4 : 1.2 + 3 * Number(track.score_ratio || 0);
          ctx.beginPath();
          points.forEach((point, index) => {
            if (index === 0) {
              ctx.moveTo(point.x, point.y);
            } else {
              ctx.lineTo(point.x, point.y);
            }
          });
          ctx.stroke();
          const last = points[points.length - 1];
          ctx.globalAlpha = 1;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(last.x, last.y, focused ? 7 : 4 + 5 * Number(track.score_ratio || 0), 0, Math.PI * 2);
          ctx.fill();
          if (focused) {
            ctx.font = "12px Arial";
            ctx.fillStyle = "#111827";
            ctx.fillText(track.track_id, last.x + 8, last.y - 8);
          }
        }
        ctx.globalAlpha = 1;
        drawTimeline(data);
        frameScrubber.value = state.frame;
        frameBadge.textContent = `Frame ${state.frame}`;
        const focused = findTrack(data, state.focusSample);
        trackReadout.textContent = focused
          ? `${focused.sequence} / track ${focused.track_id} / score ${Number(focused.score).toFixed(2)}`
          : `${data.sequence} / ${tracks.length} visible tracks`;
        renderTargetDetail(focused, data);
      }

      function stopPlayback() {
        state.playing = false;
        playPause.textContent = "Play";
      }

      function handleTimelineClick(event) {
        const data = currentData();
        if (!data) {
          return false;
        }
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * (canvas.width / rect.width);
        const y = (event.clientY - rect.top) * (canvas.height / rect.height);
        const bounds = timelineBounds(data);
        const insideTimeline =
          x >= bounds.x &&
          x <= bounds.x + bounds.width &&
          y >= bounds.y - 8 &&
          y <= bounds.y + bounds.height + 8;
        if (!insideTimeline) {
          return false;
        }
        stopPlayback();
        const ratio = Math.max(0, Math.min(1, (x - bounds.x) / bounds.width));
        state.frame = Math.round(bounds.start + ratio * bounds.span);
        frameScrubber.value = state.frame;
        drawPlayback();
        return true;
      }

      function showSequence(sequence, shouldScroll, focusSample, focusFrame) {
        if (!playbackData[sequence]) {
          return;
        }
        state.sequence = sequence;
        state.focusSample = focusSample || null;
        const data = currentData();
        const [start, end] = data.frame_range;
        state.frame = Math.min(Math.max(Number(focusFrame ?? start), start), end);
        state.image = null;
        state.imageKey = null;
        state.imageSequence = data.sequence;
        setCanvasFor(data);
        ensureBackgroundForFrame(data);
        sequenceBlocks.forEach((block) => {
          const active = block.dataset.sequence === sequence;
          block.hidden = !active;
          if (active && shouldScroll) {
            block.scrollIntoView({ behavior: "smooth", block: "start" });
          }
        });
        tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.sequence === sequence));
        targetItems.forEach((item) => item.classList.toggle("active", item.dataset.sample === state.focusSample));
        drawPlayback();
      }

      function applyFilters() {
        const term = (search.value || "").trim().toLowerCase();
        const threshold = Number(minScore.value || 0);
        minScoreValue.textContent = threshold.toFixed(2);
        targetItems.forEach((item) => {
          const matchesTerm = !term || (item.dataset.search || "").includes(term);
          const matchesScore = Number(item.dataset.score || 0) >= threshold;
          item.hidden = !(matchesTerm && matchesScore);
        });
        drawPlayback();
      }

      function visibleTargetItems() {
        return targetItems.filter((item) => !item.hidden);
      }

      function showTargetItem(item, shouldScroll) {
        showSequence(item.dataset.sequence, shouldScroll, item.dataset.sample, item.dataset.frame);
      }

      function stopAutoTour() {
        if (state.tourTimer) {
          clearInterval(state.tourTimer);
          state.tourTimer = null;
        }
        state.touring = false;
        autoTour.textContent = "Auto tour";
        document.body.classList.remove("tour-mode");
      }

      function startAutoTour() {
        const items = visibleTargetItems();
        if (!items.length) {
          return;
        }
        state.touring = true;
        autoTour.textContent = "Stop tour";
        document.body.classList.add("tour-mode");
        let index = Math.max(0, items.findIndex((item) => item.classList.contains("active")));
        const advance = () => {
          const currentItems = visibleTargetItems();
          if (!currentItems.length) {
            stopAutoTour();
            return;
          }
          const item = currentItems[index % currentItems.length];
          showTargetItem(item, false);
          const wasPlaying = state.playing;
          state.playing = true;
          playPause.textContent = "Pause";
          state.lastTick = performance.now();
          if (!wasPlaying) {
            requestAnimationFrame(tick);
          }
          index += 1;
        };
        advance();
        state.tourTimer = setInterval(advance, 5200);
      }

      function setEvidenceTab(button) {
        const block = button.closest(".evidence-block");
        if (!block) {
          return;
        }
        const selected = button.dataset.tab;
        block.querySelectorAll(".evidence-tab").forEach((tab) => {
          tab.classList.toggle("active", tab.dataset.tab === selected);
        });
        block.querySelectorAll("[data-evidence-panel]").forEach((panel) => {
          panel.hidden = panel.dataset.evidencePanel !== selected;
        });
      }

      tabs.forEach((tab) => tab.addEventListener("click", () => {
        stopAutoTour();
        showSequence(tab.dataset.sequence, true);
      }));
      targetItems.forEach((item) => item.addEventListener("click", () => {
        stopAutoTour();
        showTargetItem(item, true);
      }));
      [search, minScore].forEach((control) => control.addEventListener("input", applyFilters));
      frameScrubber.addEventListener("input", () => {
        stopAutoTour();
        stopPlayback();
        state.frame = Number(frameScrubber.value);
        drawPlayback();
      });
      speedSelect.addEventListener("change", () => {
        state.speed = Number(speedSelect.value || 1);
      });
      playPause.addEventListener("click", () => {
        state.playing = !state.playing;
        playPause.textContent = state.playing ? "Pause" : "Play";
        state.lastTick = performance.now();
        if (state.playing) {
          requestAnimationFrame(tick);
        }
      });
      autoTour.addEventListener("click", () => {
        if (state.touring) {
          stopAutoTour();
          stopPlayback();
        } else {
          startAutoTour();
        }
      });
      demoMode.addEventListener("click", () => {
        document.body.classList.toggle("demo-mode");
        demoMode.textContent = document.body.classList.contains("demo-mode") ? "Exit demo" : "Demo mode";
        drawPlayback();
      });
      canvas.addEventListener("click", handleTimelineClick);
      canvas.addEventListener("mousemove", (event) => {
        const data = currentData();
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * (canvas.width / rect.width);
        const y = (event.clientY - rect.top) * (canvas.height / rect.height);
        const bounds = timelineBounds(data);
        canvas.style.cursor =
          x >= bounds.x && x <= bounds.x + bounds.width && y >= bounds.y - 8 && y <= bounds.y + bounds.height + 8
            ? "pointer"
            : "default";
      });
      evidenceTabs.forEach((button) => button.addEventListener("click", () => setEvidenceTab(button)));
      document.querySelectorAll(".plot-image").forEach((image) => {
        image.addEventListener("click", () => {
          lightboxImage.src = image.src;
          lightboxTitle.textContent = image.dataset.title || image.alt || "";
          lightbox.hidden = false;
        });
      });
      document.getElementById("closeLightbox").addEventListener("click", () => {
        lightbox.hidden = true;
      });
      lightbox.addEventListener("click", (event) => {
        if (event.target === lightbox) {
          lightbox.hidden = true;
        }
      });
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          lightbox.hidden = true;
        }
      });
      function tick(now) {
        if (!state.playing) {
          return;
        }
        const data = currentData();
        const elapsed = now - state.lastTick;
        if (elapsed > 42) {
          const step = Math.max(1, Math.floor((elapsed / 42) * state.speed));
          state.frame += step;
          if (state.frame > data.frame_range[1]) {
            state.frame = data.frame_range[0];
          }
          state.lastTick = now;
          drawPlayback();
        }
        requestAnimationFrame(tick);
      }
      applyFilters();
      const initialTarget = document.querySelector(".target-card");
      if (initialTarget) {
        showSequence(initialTarget.dataset.sequence, false, initialTarget.dataset.sample, initialTarget.dataset.frame);
      } else if (tabs[0]) {
        showSequence(tabs[0].dataset.sequence, false);
      }
    })();
  </script>
"""

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FusionTrack v1 Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: Arial, sans-serif; margin: 0; color: #172033; background: #f6f7f9; }}
    main {{ padding: 20px 24px 36px; }}
    header {{ display: flex; justify-content: space-between; gap: 18px; align-items: end; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 26px; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; min-width: 420px; }}
    .card {{ background: white; border: 1px solid #e1e5eb; border-radius: 8px; padding: 10px 12px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .workspace {{ display: grid; grid-template-columns: minmax(270px, 340px) 1fr; gap: 16px; align-items: start; }}
    .side-panel, .viewer-panel, .evidence-block {{ background: white; border: 1px solid #e1e5eb; border-radius: 8px; padding: 14px; }}
    .side-panel {{ position: sticky; top: 12px; max-height: calc(100vh - 24px); overflow: auto; }}
    .toolbar {{ display: grid; gap: 10px; margin-bottom: 14px; }}
    label {{ display: grid; gap: 5px; font-size: 12px; color: #475569; }}
    input, select {{ min-height: 34px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 8px; background: white; color: #0f172a; }}
    input[type="range"] {{ padding: 0; }}
    .target-grid {{ display: grid; gap: 8px; }}
    .target-card {{ display: grid; grid-template-columns: 58px 1fr; gap: 10px; align-items: center; background: white; border: 1px solid #e1e5eb; border-radius: 8px; padding: 9px; color: inherit; cursor: pointer; font: inherit; text-align: left; }}
    .target-card:hover, .target-card.active {{ border-color: #2563eb; background: #eff6ff; }}
    .target-card .score {{ color: #dc2626; font-size: 20px; font-weight: 700; }}
    .target-copy {{ display: grid; gap: 3px; min-width: 0; }}
    .target-copy small {{ color: #64748b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .experiment-panel {{ margin-top: 0; border-bottom: 1px solid #e1e5eb; padding-bottom: 14px; }}
    .experiment-name {{ font-weight: 700; color: #172033; overflow-wrap: anywhere; }}
    .experiment-meta {{ margin-top: 4px; color: #64748b; font-size: 12px; }}
    .experiment-metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
    .experiment-metric {{ border: 1px solid #e1e5eb; border-radius: 6px; padding: 7px; background: #fbfdff; }}
    .experiment-metric span {{ display: block; color: #64748b; font-size: 11px; }}
    .experiment-metric strong {{ display: block; margin-top: 3px; font-size: 15px; }}
    .tabs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .tab-button {{ border: 1px solid #cbd5e1; background: white; border-radius: 999px; color: #334155; cursor: pointer; padding: 8px 12px; }}
    .tab-button.active {{ background: #111827; border-color: #111827; color: white; }}
    .player-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }}
    .player-controls {{ display: grid; grid-template-columns: auto minmax(180px, 1fr) auto auto auto; gap: 10px; align-items: center; }}
    .control-button {{ border: 1px solid #111827; background: #111827; color: white; border-radius: 6px; padding: 8px 14px; cursor: pointer; }}
    .secondary-button {{ border: 1px solid #cbd5e1; background: white; color: #172033; border-radius: 6px; padding: 8px 12px; cursor: pointer; }}
    .tour-mode #autoTour {{ border-color: #2563eb; background: #eff6ff; color: #1d4ed8; }}
    .detail-panel {{ margin-top: 12px; border: 1px solid #e1e5eb; border-radius: 8px; padding: 12px; background: #fbfdff; }}
    .detail-title {{ font-weight: 700; }}
    .detail-subtitle {{ margin-top: 2px; color: #64748b; font-size: 13px; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(6, minmax(90px, 1fr)); gap: 8px; margin-top: 10px; }}
    .metric {{ border: 1px solid #e1e5eb; border-radius: 6px; padding: 8px; background: white; }}
    .metric span {{ display: block; color: #64748b; font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 3px; font-size: 15px; }}
    .reason-panel {{ margin-top: 12px; display: grid; gap: 8px; }}
    .reason-title {{ font-size: 13px; font-weight: 700; color: #334155; }}
    .reason-row {{ display: grid; gap: 4px; }}
    .reason-copy {{ display: flex; justify-content: space-between; gap: 8px; color: #475569; font-size: 12px; }}
    .reason-copy strong {{ color: #172033; }}
    .reason-bar {{ height: 7px; overflow: hidden; border-radius: 999px; background: #e2e8f0; }}
    .reason-bar span {{ display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #2563eb, #ef4444); }}
    .gt-panel {{ margin-top: 12px; border-top: 1px solid #e1e5eb; padding-top: 10px; }}
    .gt-list {{ display: grid; gap: 6px; margin-top: 7px; }}
    .gt-item {{ display: flex; justify-content: space-between; gap: 8px; border: 1px solid #fecaca; border-radius: 6px; padding: 7px 8px; background: #fff7f7; color: #7f1d1d; font-size: 12px; }}
    #frameBadge, #trackReadout {{ color: #475569; font-size: 13px; }}
    .canvas-shell {{ margin-top: 12px; background: #111827; border-radius: 8px; padding: 10px; }}
    canvas {{ display: block; width: 100%; height: auto; background: #e2e8f0; border-radius: 6px; }}
    .evidence-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .evidence-tab {{ border: 1px solid #cbd5e1; background: white; color: #334155; border-radius: 999px; padding: 7px 12px; cursor: pointer; }}
    .evidence-tab.active {{ border-color: #111827; background: #111827; color: white; }}
    .figure-grid {{ display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 12px; }}
    .evidence-panels {{ grid-template-columns: minmax(260px, 1fr); }}
    figure {{ margin: 0; background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; }}
    figcaption {{ font-size: 13px; color: #4b5563; padding: 6px 2px 0; }}
    img {{ max-width: 100%; background: white; border-radius: 6px; }}
    .plot-image {{ cursor: zoom-in; }}
    section {{ margin-top: 16px; }}
    .lightbox {{ position: fixed; inset: 0; z-index: 20; display: grid; grid-template-rows: auto 1fr; gap: 10px; padding: 18px; background: rgba(15, 23, 42, 0.86); }}
    .lightbox[hidden] {{ display: none; }}
    .lightbox-bar {{ display: flex; justify-content: space-between; align-items: center; color: white; }}
    .lightbox img {{ align-self: center; justify-self: center; max-height: calc(100vh - 90px); max-width: 96vw; }}
    .close-button {{ border: 1px solid rgba(255, 255, 255, 0.35); background: rgba(255, 255, 255, 0.12); color: white; border-radius: 6px; padding: 8px 12px; cursor: pointer; }}
    .demo-mode main {{ padding: 14px; }}
    .demo-mode .side-panel, .demo-mode .evidence-block, .demo-mode .cards {{ display: none; }}
    .demo-mode .workspace {{ grid-template-columns: 1fr; }}
    .demo-mode .viewer-panel {{ min-height: calc(100vh - 84px); }}
    .demo-mode .canvas-shell {{ max-height: calc(100vh - 230px); overflow: hidden; }}
    @media (max-width: 900px) {{
      main {{ padding: 16px; }}
      header {{ display: grid; }}
      .cards {{ min-width: 0; grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .workspace {{ grid-template-columns: 1fr; }}
      .side-panel {{ position: static; max-height: none; }}
      .player-controls {{ grid-template-columns: 1fr; }}
      .detail-grid {{ grid-template-columns: repeat(3, minmax(90px, 1fr)); }}
      .figure-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 520px) {{
      .detail-grid {{ grid-template-columns: repeat(2, minmax(90px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>FusionTrack v1</h1>
        <div id="trackReadout">Interactive anomaly playback</div>
      </div>
      <div class="cards">
        <div class="card"><div>Sequences</div><div class="value">{len(by_sequence)}</div></div>
        <div class="card"><div>Fused tracks</div><div class="value">{len(trajectories)}</div></div>
        <div class="card"><div>Scores</div><div class="value">{len(scores)}</div></div>
        <div class="card"><div>Mean confidence</div><div class="value">{avg_confidence:.3f}</div></div>
      </div>
    </header>
    <div class="workspace">
      <aside class="side-panel">
        <div class="toolbar">
          <label>Search
            <input id="targetSearch" type="search" placeholder="sequence, track, category">
          </label>
          <label>Minimum score <span id="minScoreValue">0.00</span>
            <input id="minScore" type="range" min="0" max="{max_score:.2f}" value="0" step="0.01">
          </label>
        </div>
        {experiment_panel_html}
        <section>
          <h2>Sequences</h2>
          <div id="sequenceTabs" class="tabs">{sequence_tabs_html}</div>
        </section>
        <section>
          <h2>Top targets</h2>
          <div class="target-grid">{top_cards_html}</div>
        </section>
      </aside>
      <div>
        <section class="viewer-panel">
          <div class="player-head">
            <h2>Dynamic playback</h2>
            <span id="frameBadge">Frame 0</span>
          </div>
          <div class="player-controls">
            <button type="button" id="playPause" class="control-button">Play</button>
            <input id="frameScrubber" type="range" min="0" max="0" value="0" step="1">
            <select id="speedSelect" aria-label="Playback speed">
              <option value="1">1x</option>
              <option value="2">2x</option>
              <option value="4">4x</option>
            </select>
            <button type="button" id="demoMode" class="secondary-button">Demo mode</button>
            <button type="button" id="autoTour" class="secondary-button">Auto tour</button>
          </div>
          <div id="targetDetail" class="detail-panel"></div>
          <div class="canvas-shell"><canvas id="playbackCanvas" width="960" height="612"></canvas></div>
        </section>
        {plots_html}
      </div>
    </div>
  </main>
  <script id="playbackData" type="application/json">{playback_data_json}</script>
  <div id="lightbox" class="lightbox" hidden>
    <div class="lightbox-bar">
      <strong id="lightboxTitle"></strong>
      <button type="button" id="closeLightbox" class="close-button">Close</button>
    </div>
    <img id="lightboxImage" alt="">
  </div>
{script_text}
</body>
</html>
"""
    report_html = output_dir / "index.html"
    report_html.write_text(html_text, encoding="utf-8")

    return {
        "report_html": str(report_html),
        "assets_dir": str(assets_dir),
        "num_sequences": len(by_sequence),
        "num_trajectories": len(trajectories),
        "num_scores": len(scores),
        "experiment": _compact_experiment_context(experiment_context),
        "sequence_assets": [
            {key: str(value) if isinstance(value, Path) else value for key, value in asset.items()}
            for asset in sequence_assets
        ],
    }
