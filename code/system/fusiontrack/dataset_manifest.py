from __future__ import annotations

import hashlib
import json
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
    for modality_name, modality_code in MODALITIES.items():
        path, resolved_split = _annotation_path(annotation_dir, modality_code, split)
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
