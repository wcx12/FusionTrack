from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fusiontrack.dataset_manifest import build_dataset_manifest


def test_build_dataset_manifest_records_annotation_hashes_and_image_counts(tmp_path: Path) -> None:
    data_root = tmp_path / "VT-Tiny-MOT"
    annotations = data_root / "annotations"
    train_images = data_root / "train2017" / "DJI_0001" / "00"
    test_images = data_root / "test2017" / "DJI_0002" / "01"
    annotations.mkdir(parents=True)
    train_images.mkdir(parents=True)
    test_images.mkdir(parents=True)
    rgb_payload = {"images": [{"id": 1}], "annotations": [{"id": 9}], "videos": [{"id": 3}]}
    thermal_payload = {"images": [{"id": 2}], "annotations": [], "videos": []}
    rgb_path = annotations / "instances_00_train2017.json"
    thermal_path = annotations / "instances_01_train2017.json"
    rgb_path.write_text(json.dumps(rgb_payload), encoding="utf-8")
    thermal_path.write_text(json.dumps(thermal_payload), encoding="utf-8")
    (train_images / "000001.jpg").write_bytes(b"rgb")
    (test_images / "000001.png").write_bytes(b"thermal")

    manifest = build_dataset_manifest(data_root, splits=["train"])

    split = manifest["splits"]["train"]
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "ok"
    assert split["modalities"]["rgb"]["sha256"] == hashlib.sha256(
        rgb_path.read_bytes()
    ).hexdigest()
    assert split["modalities"]["rgb"]["num_images"] == 1
    assert split["modalities"]["rgb"]["num_annotations"] == 1
    assert split["modalities"]["thermal"]["num_images"] == 1
    assert manifest["image_dirs"]["train2017"]["num_files"] == 1
    assert manifest["image_dirs"]["test2017"]["num_files"] == 1
    assert len(manifest["dataset_fingerprint"]) == 64


def test_build_dataset_manifest_records_cross_modal_quality(tmp_path: Path) -> None:
    data_root = tmp_path / "VT-Tiny-MOT"
    annotations = data_root / "annotations"
    annotations.mkdir(parents=True)
    rgb_payload = {
        "images": [
            {"id": 1, "video_id": 10, "frame_id": 1, "file_name": "S1/00/000001.jpg"},
            {"id": 2, "video_id": 10, "frame_id": 2, "file_name": "S1/00/000002.jpg"},
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "track_id": 7, "bbox": [0, 0, 10, 10]},
            {"id": 102, "image_id": 2, "track_id": 8, "bbox": [50, 50, 10, 10]},
        ],
        "videos": [{"id": 10, "name": "S1/00"}],
    }
    thermal_payload = {
        "images": [
            {"id": 11, "video_id": 10, "frame_id": 1, "file_name": "S1/01/000001.jpg"},
            {"id": 12, "video_id": 10, "frame_id": 3, "file_name": "S1/01/000003.jpg"},
        ],
        "annotations": [
            {"id": 201, "image_id": 11, "track_id": 7, "bbox": [3, 4, 10, 10]},
            {"id": 202, "image_id": 12, "track_id": 9, "bbox": [20, 20, 8, 8]},
        ],
        "videos": [{"id": 10, "name": "S1/01"}],
    }
    (annotations / "instances_00_train2017.json").write_text(json.dumps(rgb_payload), encoding="utf-8")
    (annotations / "instances_01_train2017.json").write_text(json.dumps(thermal_payload), encoding="utf-8")

    manifest = build_dataset_manifest(data_root, splits=["train"])

    quality = manifest["splits"]["train"]["quality"]
    assert quality["status"] == "partial"
    assert quality["num_observation_keys"] == 3
    assert quality["num_rgb_annotations"] == 2
    assert quality["num_thermal_annotations"] == 2
    assert quality["num_paired_annotations"] == 1
    assert quality["num_missing_rgb_annotations"] == 1
    assert quality["num_missing_thermal_annotations"] == 1
    assert quality["rgb_annotation_coverage"] == 0.666667
    assert quality["thermal_annotation_coverage"] == 0.666667
    assert quality["paired_annotation_coverage"] == 0.333333
    assert quality["modal_offset_mean"] == 5.0
    assert quality["modal_offset_max"] == 5.0
    assert quality["by_sequence"] == [
        {
            "sequence": "S1",
            "num_observation_keys": 3,
            "num_paired_annotations": 1,
            "num_missing_rgb_annotations": 1,
            "num_missing_thermal_annotations": 1,
            "rgb_annotation_coverage": 0.666667,
            "thermal_annotation_coverage": 0.666667,
            "modal_offset_mean": 5.0,
        }
    ]
