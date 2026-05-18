#!/usr/bin/env python3
"""
Extract RGB/thermal multi-object trajectories from VT-Tiny-MOT annotations.

The script reads the two modality-specific COCO-style annotation files:
  annotations/instances_00_train2017.json
  annotations/instances_01_train2017.json

It writes:
  observations_<split>.csv   one row per sequence/track/frame
  trajectories_<split>.jsonl one full trajectory per line
  summary_<split>.json       counts and paths
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODALITIES = {
    "00": "rgb",
    "01": "thermal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract paired RGB/thermal object trajectories from VT-Tiny-MOT."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("..") / "datasets" / "VT-Tiny-MOT",
        help="Path to the VT-Tiny-MOT root directory.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "test", "val"),
        help="Dataset split to read. 'val' uses the test annotation files if no val file exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_trajectories",
        help="Directory for extracted trajectory files.",
    )
    parser.add_argument(
        "--track-field",
        default="track_id",
        choices=("track_id", "tracking_id"),
        help="Annotation field used as the object identity. track_id is globally safer.",
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Keep annotations with ignore=1 or iscrowd=1.",
    )
    parser.add_argument(
        "--indent-json",
        action="store_true",
        help="Pretty-print JSON objects in the JSONL file. Larger, but easier to inspect.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_annotation_dir(data_root: Path) -> Path:
    for name in ("annotations", "annotations_tc"):
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find an annotation directory under {data_root}. "
        "Expected 'annotations' or 'annotations_tc'."
    )


def annotation_path(annotation_dir: Path, modality: str, split: str) -> Path:
    split_name = "test" if split == "val" else split
    path = annotation_dir / f"instances_{modality}_{split_name}2017.json"
    if path.exists():
        return path
    if split == "val":
        fallback = annotation_dir / f"instances_{modality}_test2017.json"
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Missing annotation file: {path}")


def normalize_sequence_name(value: str | None) -> str:
    if not value:
        return ""
    value = value.replace("\\", "/").strip("/")
    parts = value.split("/")
    if parts and parts[-1] in MODALITIES:
        parts = parts[:-1]
    return "/".join(parts)


def bbox_to_center(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return None
    return inter / union


def category_maps(*datasets: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    for data in datasets:
        for category in data.get("categories", []):
            category_id = int(category["id"])
            name = str(category.get("name", category_id))
            id_to_name[category_id] = name
            name_to_id[name] = category_id
    return id_to_name, name_to_id


def collect_modality_points(
    data: dict[str, Any],
    modality: str,
    track_field: str,
    include_ignored: bool,
    id_to_category_name: dict[int, str],
) -> tuple[dict[tuple[str, str], dict[int, dict[str, Any]]], Counter]:
    images = {image["id"]: image for image in data.get("images", [])}
    videos = {video["id"]: video for video in data.get("videos", [])}

    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    stats: Counter = Counter()

    for ann in data.get("annotations", []):
        if not include_ignored and (ann.get("ignore", 0) or ann.get("iscrowd", 0)):
            stats["ignored_annotations_skipped"] += 1
            continue

        image = images.get(ann.get("image_id"))
        if image is None:
            stats["missing_image"] += 1
            continue

        video = videos.get(image.get("video_id"), {})
        sequence_name = normalize_sequence_name(video.get("name"))
        if not sequence_name:
            sequence_name = normalize_sequence_name(image.get("file_name", "")).split("/")[0]

        track_value = ann.get(track_field)
        if track_value is None:
            stats[f"missing_{track_field}"] += 1
            continue
        track_id = str(track_value)

        frame_id = int(image.get("mot_frame_id", image.get("frame_id", 0)))
        dataset_frame_id = int(image.get("frame_id", frame_id))
        bbox = [float(v) for v in ann["bbox"]]
        cx, cy = bbox_to_center(bbox)
        category_id = int(ann.get("category_id", -1))

        point = {
            "modality": modality,
            "sequence": sequence_name,
            "video_id": image.get("video_id"),
            "fps": video.get("fps"),
            "frame_id": frame_id,
            "dataset_frame_id": dataset_frame_id,
            "image_id": image.get("id"),
            "file_name": image.get("file_name"),
            "annotation_id": ann.get("id"),
            "track_id": track_id,
            "category_id": category_id,
            "category_name": id_to_category_name.get(category_id, str(category_id)),
            "bbox": bbox,
            "center": [cx, cy],
            "area": float(ann.get("area", bbox[2] * bbox[3])),
        }

        key = (sequence_name, track_id)
        if frame_id in grouped[key]:
            stats["duplicate_frame_track"] += 1
        grouped[key][frame_id] = point
        stats["annotations_used"] += 1

    stats["tracks"] = len(grouped)
    return grouped, stats


def add_temporal_features(points: list[dict[str, Any]], modality: str) -> None:
    previous: dict[str, Any] | None = None
    for point in points:
        current = point.get(modality)
        if current is None:
            continue

        if previous is None:
            current["delta_frame"] = None
            current["delta_time"] = None
            current["vx_px_per_frame"] = None
            current["vy_px_per_frame"] = None
            current["speed_px_per_frame"] = None
            current["vx_px_per_second"] = None
            current["vy_px_per_second"] = None
            current["speed_px_per_second"] = None
            previous = current
            continue

        delta_frame = current["frame_id"] - previous["frame_id"]
        fps = current.get("fps") or previous.get("fps")
        dt = (delta_frame / float(fps)) if fps and delta_frame else None
        dx = current["center"][0] - previous["center"][0]
        dy = current["center"][1] - previous["center"][1]

        current["delta_frame"] = delta_frame
        current["delta_time"] = dt
        current["vx_px_per_frame"] = dx / delta_frame if delta_frame else None
        current["vy_px_per_frame"] = dy / delta_frame if delta_frame else None
        current["speed_px_per_frame"] = (
            math.hypot(dx, dy) / delta_frame if delta_frame else None
        )
        current["vx_px_per_second"] = dx / dt if dt else None
        current["vy_px_per_second"] = dy / dt if dt else None
        current["speed_px_per_second"] = math.hypot(dx, dy) / dt if dt else None
        previous = current


def merge_trajectories(
    rgb_tracks: dict[tuple[str, str], dict[int, dict[str, Any]]],
    thermal_tracks: dict[tuple[str, str], dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Merge RGB-track and thermal-track observations into one object trajectory.

    For each `(sequence, track_id)`, the output trajectory contains a time-sorted
    `points` list. Each point keeps:

    - `rgb`: the RGB-modality observation at that frame, if present
    - `thermal`: the thermal-modality observation at that frame, if present
    - modal relation fields: frame-level cross-modal differences computed only
      from the paired RGB/thermal observations of the same object

    These modal relation fields are intentionally kept because later anomaly
    modeling may use them as cross-modal consistency signals:

    - `modal_offset_dx_thermal_minus_rgb`
    - `modal_offset_dy_thermal_minus_rgb`
    - `modal_offset_distance`
    - `modal_bbox_iou`
    """
    trajectories: list[dict[str, Any]] = []
    all_keys = sorted(set(rgb_tracks) | set(thermal_tracks))

    for sequence, track_id in all_keys:
        rgb_by_frame = rgb_tracks.get((sequence, track_id), {})
        thermal_by_frame = thermal_tracks.get((sequence, track_id), {})
        frames = sorted(set(rgb_by_frame) | set(thermal_by_frame))

        points: list[dict[str, Any]] = []
        category_names: Counter = Counter()
        category_ids: Counter = Counter()
        fps_values: Counter = Counter()

        for frame_id in frames:
            rgb = rgb_by_frame.get(frame_id)
            thermal = thermal_by_frame.get(frame_id)

            for item in (rgb, thermal):
                if item is not None:
                    category_names[item["category_name"]] += 1
                    category_ids[item["category_id"]] += 1
                    if item.get("fps"):
                        fps_values[item["fps"]] += 1

            rgb_center = rgb["center"] if rgb else None
            thermal_center = thermal["center"] if thermal else None
            if rgb_center and thermal_center:
                # Cross-modal center disagreement for the same object at the
                # same frame. Positive dx means the thermal center is to the
                # right of the RGB center; positive dy means it is lower.
                offset_dx = thermal_center[0] - rgb_center[0]
                offset_dy = thermal_center[1] - rgb_center[1]
                offset_dist = math.hypot(offset_dx, offset_dy)
            else:
                offset_dx = offset_dy = offset_dist = None

            points.append(
                {
                    "frame_id": frame_id,
                    "rgb": rgb,
                    "thermal": thermal,
                    # `modal_*` fields do not describe a third modality.
                    # They describe how different RGB and thermal are from
                    # each other for this object at this frame.
                    "modal_offset_dx_thermal_minus_rgb": offset_dx,
                    "modal_offset_dy_thermal_minus_rgb": offset_dy,
                    "modal_offset_distance": offset_dist,
                    # IoU between the RGB and thermal boxes for the same
                    # object and frame; higher values imply stronger
                    # cross-modal spatial agreement.
                    "modal_bbox_iou": bbox_iou(
                        rgb["bbox"] if rgb else None,
                        thermal["bbox"] if thermal else None,
                    ),
                }
            )

        add_temporal_features(points, "rgb")
        add_temporal_features(points, "thermal")

        category_name = category_names.most_common(1)[0][0] if category_names else None
        category_id = category_ids.most_common(1)[0][0] if category_ids else None
        fps = fps_values.most_common(1)[0][0] if fps_values else None

        trajectories.append(
            {
                "sequence": sequence,
                "track_id": track_id,
                "category_id": category_id,
                "category_name": category_name,
                "fps": fps,
                "start_frame": frames[0] if frames else None,
                "end_frame": frames[-1] if frames else None,
                "num_frames": len(frames),
                "num_rgb_points": len(rgb_by_frame),
                "num_thermal_points": len(thermal_by_frame),
                "points": points,
            }
        )

    return trajectories


def flat_value(point: dict[str, Any] | None, key: str, index: int | None = None) -> Any:
    if point is None:
        return ""
    value = point.get(key)
    if index is not None:
        if value is None:
            return ""
        return value[index]
    return "" if value is None else value


def write_observations_csv(path: Path, trajectories: list[dict[str, Any]]) -> int:
    fieldnames = [
        "sequence",
        "track_id",
        "category_id",
        "category_name",
        "fps",
        "frame_id",
        "rgb_file",
        "thermal_file",
        "rgb_x",
        "rgb_y",
        "rgb_w",
        "rgb_h",
        "rgb_cx",
        "rgb_cy",
        "rgb_vx_px_per_frame",
        "rgb_vy_px_per_frame",
        "rgb_speed_px_per_frame",
        "rgb_vx_px_per_second",
        "rgb_vy_px_per_second",
        "rgb_speed_px_per_second",
        "thermal_x",
        "thermal_y",
        "thermal_w",
        "thermal_h",
        "thermal_cx",
        "thermal_cy",
        "thermal_vx_px_per_frame",
        "thermal_vy_px_per_frame",
        "thermal_speed_px_per_frame",
        "thermal_vx_px_per_second",
        "thermal_vy_px_per_second",
        "thermal_speed_px_per_second",
        "modal_offset_dx_thermal_minus_rgb",
        "modal_offset_dy_thermal_minus_rgb",
        "modal_offset_distance",
        "modal_bbox_iou",
    ]

    rows_written = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trajectory in trajectories:
            for point in trajectory["points"]:
                rgb = point["rgb"]
                thermal = point["thermal"]
                writer.writerow(
                    {
                        "sequence": trajectory["sequence"],
                        "track_id": trajectory["track_id"],
                        "category_id": trajectory["category_id"],
                        "category_name": trajectory["category_name"],
                        "fps": trajectory["fps"],
                        "frame_id": point["frame_id"],
                        "rgb_file": flat_value(rgb, "file_name"),
                        "thermal_file": flat_value(thermal, "file_name"),
                        "rgb_x": flat_value(rgb, "bbox", 0),
                        "rgb_y": flat_value(rgb, "bbox", 1),
                        "rgb_w": flat_value(rgb, "bbox", 2),
                        "rgb_h": flat_value(rgb, "bbox", 3),
                        "rgb_cx": flat_value(rgb, "center", 0),
                        "rgb_cy": flat_value(rgb, "center", 1),
                        "rgb_vx_px_per_frame": flat_value(rgb, "vx_px_per_frame"),
                        "rgb_vy_px_per_frame": flat_value(rgb, "vy_px_per_frame"),
                        "rgb_speed_px_per_frame": flat_value(rgb, "speed_px_per_frame"),
                        "rgb_vx_px_per_second": flat_value(rgb, "vx_px_per_second"),
                        "rgb_vy_px_per_second": flat_value(rgb, "vy_px_per_second"),
                        "rgb_speed_px_per_second": flat_value(rgb, "speed_px_per_second"),
                        "thermal_x": flat_value(thermal, "bbox", 0),
                        "thermal_y": flat_value(thermal, "bbox", 1),
                        "thermal_w": flat_value(thermal, "bbox", 2),
                        "thermal_h": flat_value(thermal, "bbox", 3),
                        "thermal_cx": flat_value(thermal, "center", 0),
                        "thermal_cy": flat_value(thermal, "center", 1),
                        "thermal_vx_px_per_frame": flat_value(thermal, "vx_px_per_frame"),
                        "thermal_vy_px_per_frame": flat_value(thermal, "vy_px_per_frame"),
                        "thermal_speed_px_per_frame": flat_value(thermal, "speed_px_per_frame"),
                        "thermal_vx_px_per_second": flat_value(thermal, "vx_px_per_second"),
                        "thermal_vy_px_per_second": flat_value(thermal, "vy_px_per_second"),
                        "thermal_speed_px_per_second": flat_value(thermal, "speed_px_per_second"),
                        "modal_offset_dx_thermal_minus_rgb": point[
                            "modal_offset_dx_thermal_minus_rgb"
                        ],
                        "modal_offset_dy_thermal_minus_rgb": point[
                            "modal_offset_dy_thermal_minus_rgb"
                        ],
                        "modal_offset_distance": point["modal_offset_distance"],
                        "modal_bbox_iou": point["modal_bbox_iou"],
                    }
                )
                rows_written += 1
    return rows_written


def write_trajectories_jsonl(
    path: Path, trajectories: list[dict[str, Any]], indent: bool
) -> None:
    json_kwargs = {"ensure_ascii": False}
    if indent:
        json_kwargs["indent"] = 2

    with path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            f.write(json.dumps(trajectory, **json_kwargs))
            f.write("\n")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    annotation_dir = find_annotation_dir(data_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_data = load_json(annotation_path(annotation_dir, "00", args.split))
    thermal_data = load_json(annotation_path(annotation_dir, "01", args.split))
    id_to_category_name, _ = category_maps(rgb_data, thermal_data)

    rgb_tracks, rgb_stats = collect_modality_points(
        rgb_data,
        modality="rgb",
        track_field=args.track_field,
        include_ignored=args.include_ignored,
        id_to_category_name=id_to_category_name,
    )
    thermal_tracks, thermal_stats = collect_modality_points(
        thermal_data,
        modality="thermal",
        track_field=args.track_field,
        include_ignored=args.include_ignored,
        id_to_category_name=id_to_category_name,
    )

    trajectories = merge_trajectories(rgb_tracks, thermal_tracks)

    csv_path = output_dir / f"observations_{args.split}.csv"
    jsonl_path = output_dir / f"trajectories_{args.split}.jsonl"
    summary_path = output_dir / f"summary_{args.split}.json"

    rows_written = write_observations_csv(csv_path, trajectories)
    write_trajectories_jsonl(jsonl_path, trajectories, args.indent_json)

    summary = {
        "data_root": str(data_root),
        "annotation_dir": str(annotation_dir),
        "split": args.split,
        "track_field": args.track_field,
        "num_trajectories": len(trajectories),
        "num_observation_rows": rows_written,
        "num_rgb_tracks": len(rgb_tracks),
        "num_thermal_tracks": len(thermal_tracks),
        "rgb_stats": dict(rgb_stats),
        "thermal_stats": dict(thermal_stats),
        "outputs": {
            "observations_csv": str(csv_path),
            "trajectories_jsonl": str(jsonl_path),
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
