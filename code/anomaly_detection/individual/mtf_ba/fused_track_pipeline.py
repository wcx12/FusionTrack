from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from mtf_ba.group_interface import GroupWindowConfig, build_group_windows
from mtf_ba.individual_trajectories import load_object_trajectories


@dataclass(frozen=True)
class TrackQualityConfig:
    min_points: int = 1
    min_visible_any_frames: int = 1
    max_frame_gap: int | None = None
    min_fused_ratio: float = 0.0
    keep_filtered: bool = False

    def __post_init__(self) -> None:
        if self.min_points < 0:
            raise ValueError("min_points must be non-negative.")
        if self.min_visible_any_frames < 0:
            raise ValueError("min_visible_any_frames must be non-negative.")
        if self.max_frame_gap is not None and self.max_frame_gap < 0:
            raise ValueError("max_frame_gap must be non-negative when set.")
        if not 0.0 <= self.min_fused_ratio <= 1.0:
            raise ValueError("min_fused_ratio must be in [0, 1].")


@dataclass(frozen=True)
class FusedTrackPipelineConfig:
    split: str = "train"
    offset_scale: float = 25.0
    group: GroupWindowConfig = field(default_factory=GroupWindowConfig)
    quality: TrackQualityConfig = field(default_factory=TrackQualityConfig)


def run_fused_track_pipeline(
    csv_path: str | Path,
    output_dir: str | Path,
    config: FusedTrackPipelineConfig | None = None,
) -> dict[str, Any]:
    config = config or FusedTrackPipelineConfig()
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_object_trajectories(csv_path, show_progress=False)
    fused_trajectories = build_fused_trajectories(
        trajectories,
        offset_scale=config.offset_scale,
        quality=config.quality,
    )
    kept_trajectories = [
        trajectory
        for trajectory in fused_trajectories
        if trajectory.get("quality", {}).get("keep", True)
    ]
    filtered_trajectories = [
        trajectory
        for trajectory in fused_trajectories
        if not trajectory.get("quality", {}).get("keep", True)
    ]
    written_fused_trajectories = (
        fused_trajectories if config.quality.keep_filtered else kept_trajectories
    )
    written_sample_ids = {
        str(trajectory.get("sample_id"))
        for trajectory in written_fused_trajectories
    }
    written_individual_trajectories = [
        trajectory
        for trajectory in trajectories
        if str(trajectory.get("sample_id")) in written_sample_ids
    ]
    group_windows = build_group_windows(csv_path, config=config.group)
    group_payloads = _filter_group_window_payloads(
        [window.to_dict() for window in group_windows],
        sample_ids=written_sample_ids,
    )

    individual_path = output_dir / f"individual_trajectories_{config.split}.jsonl"
    fused_path = output_dir / f"fused_trajectories_{config.split}.jsonl"
    group_path = output_dir / f"group_windows_{config.split}.jsonl"
    summary_path = output_dir / f"fused_track_pipeline_summary_{config.split}.json"
    manifest_path = output_dir / f"fused_track_pipeline_manifest_{config.split}.json"

    _write_jsonl(individual_path, written_individual_trajectories)
    _write_jsonl(fused_path, written_fused_trajectories)
    _write_jsonl(group_path, group_payloads)

    coverage = _modality_coverage(written_fused_trajectories)
    quality_summary = _trajectory_quality_summary(fused_trajectories)
    summary = {
        "schema_version": 1,
        "pipeline": "fused_track_pipeline",
        "split": config.split,
        "counts": {
            "observations": _count_csv_rows(csv_path),
            "raw_trajectories": len(trajectories),
            "trajectories": len(written_individual_trajectories),
            "fused_trajectories": len(written_fused_trajectories),
            "filtered_trajectories": len(filtered_trajectories),
            "points": sum(int(item.get("num_points", 0)) for item in written_individual_trajectories),
            "fused_points": coverage["fused_points"],
            "group_windows": len(group_payloads),
            "group_window_objects": sum(int(item.get("num_objects", 0)) for item in group_payloads),
        },
        "modality_coverage": coverage,
        "trajectory_quality": quality_summary,
        "outputs": {
            "individual_trajectories": individual_path.name,
            "fused_trajectories": fused_path.name,
            "group_windows": group_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
        },
        "config": _config_dict(config),
    }
    _write_json(summary_path, summary)

    manifest = _build_manifest(
        csv_path=csv_path,
        output_dir=output_dir,
        config=config,
        summary=summary,
        artifacts={
            "individual_trajectories": individual_path,
            "fused_trajectories": fused_path,
            "group_windows": group_path,
            "summary": summary_path,
        },
    )
    _write_json(manifest_path, manifest)
    return summary


def build_fused_trajectories(
    trajectories: Iterable[dict[str, Any]],
    *,
    offset_scale: float = 25.0,
    quality: TrackQualityConfig | None = None,
) -> list[dict[str, Any]]:
    return [
        build_fused_trajectory(
            trajectory,
            offset_scale=offset_scale,
            quality=quality,
        )
        for trajectory in trajectories
    ]


def build_fused_trajectory(
    trajectory: dict[str, Any],
    *,
    offset_scale: float = 25.0,
    quality: TrackQualityConfig | None = None,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for point in trajectory.get("points", []):
        fused_point = dict(point)
        fused_point["fused"] = fuse_point_state(point, offset_scale=offset_scale)
        points.append(fused_point)

    fused_trajectory = {
        key: trajectory[key]
        for key in (
            "sample_id",
            "sequence",
            "track_id",
            "category_id",
            "category_name",
            "fps",
            "num_points",
            "visible_rgb_frames",
            "visible_thermal_frames",
        )
        if key in trajectory
    }
    fused_trajectory["points"] = points
    fused_trajectory["temporal_linkage"] = temporal_linkage(points)
    fused_trajectory["quality"] = evaluate_track_quality(
        fused_trajectory,
        quality or TrackQualityConfig(),
    )
    return fused_trajectory


def fuse_point_state(
    point: dict[str, Any],
    *,
    offset_scale: float = 25.0,
) -> dict[str, Any] | None:
    states = [
        (modality, state)
        for modality in ("rgb", "thermal")
        if (state := point.get(modality)) is not None
        and isinstance(state, dict)
        and _center_xy(state) is not None
    ]
    if not states:
        return None

    source_modalities = [modality for modality, _ in states]
    centers = [_center_xy(state) for _, state in states]
    valid_centers = [center for center in centers if center is not None]
    center_xy = [
        sum(center[0] for center in valid_centers) / len(valid_centers),
        sum(center[1] for center in valid_centers) / len(valid_centers),
    ]
    bbox_xywh = _mean_bbox([state.get("bbox_xywh") for _, state in states])

    if len(valid_centers) == 1:
        modal_offset = 0.0
        confidence = 0.55
    else:
        modal_offset = _modal_offset_distance(point, valid_centers)
        safe_scale = offset_scale if offset_scale > 0.0 else 25.0
        confidence = 1.0 / (1.0 + modal_offset / safe_scale)

    weight = 1.0 / len(source_modalities)
    return {
        "center_xy": center_xy,
        "bbox_xywh": bbox_xywh,
        "confidence": confidence,
        "source_modalities": source_modalities,
        "component_scores": {
            "modal_offset_distance": modal_offset,
            **{f"{modality}_weight": weight for modality in source_modalities},
        },
    }


def temporal_linkage(points: Iterable[dict[str, Any]]) -> dict[str, Any]:
    frame_ids = sorted(
        int(frame_id)
        for point in points
        if (frame_id := point.get("frame_id")) is not None
    )
    if not frame_ids:
        return {
            "frame_start": None,
            "frame_end": None,
            "frame_ids": [],
            "frame_gaps": [],
            "max_frame_gap": None,
            "mean_frame_gap": None,
            "missing_frame_count": 0,
        }
    frame_gaps = [
        frame_ids[index] - frame_ids[index - 1]
        for index in range(1, len(frame_ids))
    ]
    frame_span = frame_ids[-1] - frame_ids[0] + 1
    return {
        "frame_start": frame_ids[0],
        "frame_end": frame_ids[-1],
        "frame_ids": frame_ids,
        "frame_gaps": frame_gaps,
        "max_frame_gap": max(frame_gaps) if frame_gaps else 0,
        "mean_frame_gap": sum(frame_gaps) / len(frame_gaps) if frame_gaps else 0.0,
        "missing_frame_count": max(frame_span - len(set(frame_ids)), 0),
    }


def evaluate_track_quality(
    trajectory: dict[str, Any],
    config: TrackQualityConfig | None = None,
) -> dict[str, Any]:
    config = config or TrackQualityConfig()
    points = list(trajectory.get("points", []))
    linkage = trajectory.get("temporal_linkage") or temporal_linkage(points)
    num_points = len(points)
    visible_any_frames = sum(
        1 for point in points if point.get("rgb") is not None or point.get("thermal") is not None
    )
    fused_points = sum(1 for point in points if point.get("fused") is not None)
    fused_ratio = fused_points / num_points if num_points else 0.0
    max_frame_gap = linkage.get("max_frame_gap")
    drop_reasons: list[str] = []

    if num_points < config.min_points:
        drop_reasons.append("short_track")
    if visible_any_frames < config.min_visible_any_frames:
        drop_reasons.append("low_visible_frames")
    if (
        config.max_frame_gap is not None
        and max_frame_gap is not None
        and int(max_frame_gap) > config.max_frame_gap
    ):
        drop_reasons.append("large_frame_gap")
    if fused_ratio < config.min_fused_ratio:
        drop_reasons.append("low_fused_ratio")

    return {
        "keep": not drop_reasons,
        "drop_reasons": drop_reasons,
        "num_points": num_points,
        "visible_any_frames": visible_any_frames,
        "fused_points": fused_points,
        "fused_ratio": fused_ratio,
        "max_frame_gap": max_frame_gap,
        "missing_frame_count": linkage.get("missing_frame_count"),
    }


def _center_xy(state: dict[str, Any]) -> list[float] | None:
    center = state.get("center_xy")
    if not isinstance(center, list) or len(center) != 2:
        return None
    return [float(center[0]), float(center[1])]


def _modal_offset_distance(point: dict[str, Any], centers: list[list[float]]) -> float:
    modal = point.get("modal")
    if isinstance(modal, dict) and modal.get("offset_distance") is not None:
        return float(modal["offset_distance"])
    if len(centers) < 2:
        return 0.0
    return math.dist(centers[0], centers[1])


def _mean_bbox(boxes: Iterable[Any]) -> list[float] | None:
    valid_boxes: list[list[float]] = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(value is None for value in box):
            continue
        valid_boxes.append([float(value) for value in box])
    if not valid_boxes:
        return None
    return [
        sum(box[index] for box in valid_boxes) / len(valid_boxes)
        for index in range(4)
    ]


def _modality_coverage(fused_trajectories: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rgb_points = 0
    thermal_points = 0
    paired_points = 0
    rgb_only_points = 0
    thermal_only_points = 0
    missing_both_points = 0
    fused_points = 0

    for trajectory in fused_trajectories:
        for point in trajectory.get("points", []):
            has_rgb = point.get("rgb") is not None
            has_thermal = point.get("thermal") is not None
            if point.get("fused") is not None:
                fused_points += 1
            if has_rgb:
                rgb_points += 1
            if has_thermal:
                thermal_points += 1
            if has_rgb and has_thermal:
                paired_points += 1
            elif has_rgb:
                rgb_only_points += 1
            elif has_thermal:
                thermal_only_points += 1
            else:
                missing_both_points += 1

    total_points = paired_points + rgb_only_points + thermal_only_points + missing_both_points
    return {
        "total_points": total_points,
        "fused_points": fused_points,
        "rgb_points": rgb_points,
        "thermal_points": thermal_points,
        "paired_points": paired_points,
        "rgb_only_points": rgb_only_points,
        "thermal_only_points": thermal_only_points,
        "missing_both_points": missing_both_points,
        "paired_ratio": paired_points / total_points if total_points else 0.0,
        "fused_ratio": fused_points / total_points if total_points else 0.0,
    }


def _trajectory_quality_summary(fused_trajectories: Iterable[dict[str, Any]]) -> dict[str, Any]:
    trajectories = list(fused_trajectories)
    kept = [
        trajectory
        for trajectory in trajectories
        if trajectory.get("quality", {}).get("keep", True)
    ]
    filtered = [
        trajectory
        for trajectory in trajectories
        if not trajectory.get("quality", {}).get("keep", True)
    ]
    reason_counts: dict[str, int] = {}
    for trajectory in filtered:
        for reason in trajectory.get("quality", {}).get("drop_reasons", []):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "kept_trajectories": len(kept),
        "filtered_trajectories": len(filtered),
        "drop_reason_counts": dict(sorted(reason_counts.items())),
    }


def _filter_group_window_payloads(
    windows: Iterable[dict[str, Any]],
    *,
    sample_ids: set[str],
) -> list[dict[str, Any]]:
    filtered_windows: list[dict[str, Any]] = []
    for window in windows:
        objects = [
            obj
            for obj in window.get("objects", [])
            if str(obj.get("sample_id")) in sample_ids
        ]
        if not objects:
            continue
        payload = dict(window)
        payload["objects"] = objects
        payload["num_objects"] = len(objects)
        filtered_windows.append(payload)
    return filtered_windows


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_manifest(
    *,
    csv_path: Path,
    output_dir: Path,
    config: FusedTrackPipelineConfig,
    summary: dict[str, Any],
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    return {
        "manifest_schema_version": 1,
        "pipeline": "fused_track_pipeline",
        "split": config.split,
        "inputs": {
            "observations_csv": _portable_path(csv_path),
            "observations_csv_sha256": _sha256(csv_path),
        },
        "config": _config_dict(config),
        "summary": summary,
        "artifacts": {
            name: {
                "path": _portable_path(path, base=output_dir),
                "sha256": _sha256(path),
            }
            for name, path in artifacts.items()
        },
    }


def _config_dict(config: FusedTrackPipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["group"] = asdict(config.group)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path, base: Path | None = None) -> str:
    path = Path(path)
    if base is not None:
        try:
            return str(path.relative_to(base))
        except ValueError:
            pass
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return path.name
