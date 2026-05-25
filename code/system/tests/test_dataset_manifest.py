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
