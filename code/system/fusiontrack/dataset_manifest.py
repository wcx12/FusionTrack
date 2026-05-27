from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 1
DATASET_NAME = "VT-Tiny-MOT"
MODALITIES = {
    "rgb": "00",
    "thermal": "01",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_SPLITS = ("train", "test", "val")


def build_dataset_manifest(data_root: str | Path, splits: Iterable[str] | None = None) -> dict[str, Any]:
    root = Path(data_root)
    requested_splits = tuple(splits or DEFAULT_SPLITS)
    errors: list[str] = []
    annotation_dir = _find_annotation_dir(root)
    if not root.exists():
        errors.append(f"missing data root: {root}")
    elif annotation_dir is None:
        errors.append(f"missing annotation directory under data root: {root}")

    split_payloads = {
        split: _split_payload(root, annotation_dir, split, errors)
        for split in requested_splits
    }
    image_dirs = {
        image_dir: _image_dir_payload(root, image_dir)
        for image_dir in ("train2017", "test2017", "val2017")
    }
    status = _status(root, annotation_dir, errors)
    fingerprint_payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": DATASET_NAME,
        "splits": split_payloads,
        "image_dirs": image_dirs,
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": DATASET_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(root),
        "data_root_exists": root.exists(),
        "annotation_dir": None if annotation_dir is None else _relative_path(annotation_dir, root),
        "status": status,
        "errors": errors,
        "splits": split_payloads,
        "image_dirs": image_dirs,
        "dataset_fingerprint": _stable_sha256(fingerprint_payload),
    }


def write_dataset_manifest(
    data_root: str | Path,
    output_path: str | Path,
    splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    manifest = build_dataset_manifest(data_root=data_root, splits=splits)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _find_annotation_dir(data_root: Path) -> Path | None:
    for name in ("annotations", "annotations_tc"):
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    return None


def _split_payload(
    data_root: Path,
    annotation_dir: Path | None,
    split: str,
    errors: list[str],
) -> dict[str, Any]:
    modalities: dict[str, Any] = {}
    annotation_paths: dict[str, Path | None] = {}
    for modality_name, modality_code in MODALITIES.items():
        path, resolved_split = _annotation_path(annotation_dir, modality_code, split)
        annotation_paths[modality_name] = path
        if path is None:
            errors.append(f"missing annotation for split={split} modality={modality_name}")
            modalities[modality_name] = {
                "exists": False,
                "path": None,
                "requested_split": split,
                "resolved_split": None,
                "size_bytes": 0,
                "sha256": None,
                "num_images": 0,
                "num_annotations": 0,
                "num_videos": 0,
                "num_categories": 0,
            }
            continue
        metadata = _annotation_metadata(path)
        modalities[modality_name] = {
            "exists": True,
            "path": _relative_path(path, data_root),
            "requested_split": split,
            "resolved_split": resolved_split,
            **metadata,
        }
    return {
        "requested_split": split,
        "modalities": modalities,
        "quality": _cross_modal_quality(annotation_paths.get("rgb"), annotation_paths.get("thermal")),
    }


def _annotation_path(annotation_dir: Path | None, modality_code: str, split: str) -> tuple[Path | None, str | None]:
    if annotation_dir is None:
        return None, None
    candidates = [split]
    if split == "val":
        candidates.append("test")
    for candidate_split in candidates:
        path = annotation_dir / f"instances_{modality_code}_{candidate_split}2017.json"
        if path.exists():
            return path, candidate_split
    return None, None


def _annotation_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        payload = {}
    return {
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "num_images": _list_len(payload.get("images")),
        "num_annotations": _list_len(payload.get("annotations")),
        "num_videos": _list_len(payload.get("videos")),
        "num_categories": _list_len(payload.get("categories")),
    }


def _cross_modal_quality(rgb_path: Path | None, thermal_path: Path | None) -> dict[str, Any]:
    rgb_records = _annotation_records(rgb_path)
    thermal_records = _annotation_records(thermal_path)
    rgb_keys = set(rgb_records)
    thermal_keys = set(thermal_records)
    union_keys = rgb_keys | thermal_keys
    paired_keys = rgb_keys & thermal_keys
    missing_rgb_keys = thermal_keys - rgb_keys
    missing_thermal_keys = rgb_keys - thermal_keys
    offsets = [
        _center_distance(rgb_records[key].get("center_xy"), thermal_records[key].get("center_xy"))
        for key in paired_keys
    ]
    offsets = [value for value in offsets if value is not None]
    ious = [
        _bbox_iou(rgb_records[key].get("bbox_xywh"), thermal_records[key].get("bbox_xywh"))
        for key in paired_keys
    ]
    ious = [value for value in ious if value is not None]
    return {
        "schema_version": 1,
        "status": _quality_status(union_keys, missing_rgb_keys, missing_thermal_keys),
        "pair_key_fields": ["sequence", "track_id", "frame_id"],
        "num_observation_keys": len(union_keys),
        "num_rgb_annotations": len(rgb_records),
        "num_thermal_annotations": len(thermal_records),
        "num_paired_annotations": len(paired_keys),
        "num_missing_rgb_annotations": len(missing_rgb_keys),
        "num_missing_thermal_annotations": len(missing_thermal_keys),
        "rgb_annotation_coverage": _coverage(len(rgb_keys), len(union_keys)),
        "thermal_annotation_coverage": _coverage(len(thermal_keys), len(union_keys)),
        "paired_annotation_coverage": _coverage(len(paired_keys), len(union_keys)),
        "modal_offset_mean": _mean(offsets),
        "modal_offset_max": _max_or_zero(offsets),
        "modal_iou_mean": _mean(ious),
        "by_sequence": _quality_by_sequence(union_keys, rgb_records, thermal_records),
    }


def _annotation_records(path: Path | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    images = {str(image.get("id")): image for image in payload.get("images", []) if isinstance(image, dict)}
    videos = {
        str(video.get("id")): str(video.get("name") or video.get("file_name") or video.get("video_name") or video.get("id"))
        for video in payload.get("videos", [])
        if isinstance(video, dict) and video.get("id") is not None
    }
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        image = images.get(str(annotation.get("image_id")), {})
        key = _annotation_pair_key(annotation, image, videos)
        if key is None or key in records:
            continue
        bbox = _bbox_xywh(annotation)
        records[key] = {
            "sequence": key[0],
            "track_id": key[1],
            "frame_id": key[2],
            "bbox_xywh": bbox,
            "center_xy": _bbox_center(bbox),
        }
    return records


def _annotation_pair_key(
    annotation: dict[str, Any],
    image: dict[str, Any],
    videos: dict[str, str],
) -> tuple[str, str, int] | None:
    sequence = _sequence_name(annotation, image, videos)
    track_id = _track_id(annotation)
    frame_id = _frame_id(annotation, image)
    if sequence is None or track_id is None or frame_id is None:
        return None
    return sequence, track_id, frame_id


def _sequence_name(annotation: dict[str, Any], image: dict[str, Any], videos: dict[str, str]) -> str | None:
    for field in ("sequence", "seq", "video", "video_name"):
        value = annotation.get(field) or image.get(field)
        if value not in (None, ""):
            return _normalize_sequence_name(value)
    video_id = image.get("video_id") or annotation.get("video_id")
    if video_id is not None and str(video_id) in videos:
        return _normalize_sequence_name(videos[str(video_id)])
    file_name = str(image.get("file_name") or annotation.get("file_name") or "")
    if file_name:
        parts = Path(file_name).parts
        if len(parts) >= 3 and parts[-2] in {"00", "01"}:
            return _normalize_sequence_name(parts[-3])
        if len(parts) >= 2:
            return _normalize_sequence_name(parts[0])
    return None


def _normalize_sequence_name(value: Any) -> str:
    text = str(value).replace("\\", "/").strip("/")
    for suffix in ("/00", "/01"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _track_id(annotation: dict[str, Any]) -> str | None:
    for field in ("track_id", "instance_id", "object_id", "target_id"):
        value = annotation.get(field)
        if value not in (None, ""):
            return str(value)
    value = annotation.get("id")
    return None if value in (None, "") else str(value)


def _frame_id(annotation: dict[str, Any], image: dict[str, Any]) -> int | None:
    for field in ("frame_id", "frame", "frame_index", "frame_number"):
        value = image.get(field, annotation.get(field))
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    file_name = str(image.get("file_name") or annotation.get("file_name") or "")
    if file_name:
        stem = Path(file_name).stem
        parsed = _int_or_none(stem)
        if parsed is not None:
            return parsed
    return None


def _bbox_xywh(annotation: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    values = [_float_or_none(item) for item in bbox[:4]]
    if any(value is None for value in values):
        return None
    x, y, w, h = values
    return float(x), float(y), float(w), float(h)


def _bbox_center(bbox: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    if bbox is None:
        return None
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _center_distance(
    first: tuple[float, float] | None,
    second: tuple[float, float] | None,
) -> float | None:
    if first is None or second is None:
        return None
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _bbox_iou(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> float | None:
    if first is None or second is None:
        return None
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    if union <= 0.0:
        return None
    return _round(intersection / union)


def _quality_by_sequence(
    union_keys: set[tuple[str, str, int]],
    rgb_records: dict[tuple[str, str, int], dict[str, Any]],
    thermal_records: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in sorted({key[0] for key in union_keys}):
        sequence_keys = {key for key in union_keys if key[0] == sequence}
        rgb_keys = {key for key in sequence_keys if key in rgb_records}
        thermal_keys = {key for key in sequence_keys if key in thermal_records}
        paired_keys = rgb_keys & thermal_keys
        offsets = [
            _center_distance(rgb_records[key].get("center_xy"), thermal_records[key].get("center_xy"))
            for key in paired_keys
        ]
        offsets = [value for value in offsets if value is not None]
        rows.append(
            {
                "sequence": sequence,
                "num_observation_keys": len(sequence_keys),
                "num_paired_annotations": len(paired_keys),
                "num_missing_rgb_annotations": len(sequence_keys - rgb_keys),
                "num_missing_thermal_annotations": len(sequence_keys - thermal_keys),
                "rgb_annotation_coverage": _coverage(len(rgb_keys), len(sequence_keys)),
                "thermal_annotation_coverage": _coverage(len(thermal_keys), len(sequence_keys)),
                "modal_offset_mean": _mean(offsets),
            }
        )
    return rows


def _quality_status(
    union_keys: set[tuple[str, str, int]],
    missing_rgb_keys: set[tuple[str, str, int]],
    missing_thermal_keys: set[tuple[str, str, int]],
) -> str:
    if not union_keys:
        return "missing"
    if missing_rgb_keys or missing_thermal_keys:
        return "partial"
    return "ok"


def _coverage(numerator: int, denominator: int) -> float:
    return _round(numerator / denominator) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return _round(sum(values) / len(values)) if values else 0.0


def _max_or_zero(values: list[float]) -> float:
    return _round(max(values)) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _image_dir_payload(data_root: Path, image_dir: str) -> dict[str, Any]:
    path = data_root / image_dir
    suffix_counts: dict[str, int] = {}
    count = 0
    if path.is_dir():
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            count += 1
    return {
        "exists": path.is_dir(),
        "path": _relative_path(path, data_root),
        "num_files": count,
        "suffix_counts": dict(sorted(suffix_counts.items())),
    }


def _status(data_root: Path, annotation_dir: Path | None, errors: list[str]) -> str:
    if not data_root.exists():
        return "missing_data_root"
    if annotation_dir is None:
        return "missing_annotations"
    return "partial" if errors else "ok"


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
