from __future__ import annotations

import csv
import json
from pathlib import Path

from mtf_ba.fused_track_pipeline import (
    FusedTrackPipelineConfig,
    run_fused_track_pipeline,
)
from mtf_ba.group_interface import GroupWindowConfig


FIELDNAMES = [
    "dataset",
    "sequence",
    "track_id",
    "category_id",
    "category_name",
    "fps",
    "frame_id",
    "rgb_x",
    "rgb_y",
    "rgb_w",
    "rgb_h",
    "rgb_cx",
    "rgb_cy",
    "thermal_x",
    "thermal_y",
    "thermal_w",
    "thermal_h",
    "thermal_cx",
    "thermal_cy",
    "modal_offset_distance",
]


def _write_observations(path: Path) -> None:
    rows = [
        {
            "dataset": "toy",
            "sequence": "SEQ-A",
            "track_id": "1",
            "category_id": "1",
            "category_name": "vehicle",
            "fps": "25",
            "frame_id": "1",
            "rgb_x": "0",
            "rgb_y": "10",
            "rgb_w": "20",
            "rgb_h": "20",
            "rgb_cx": "10",
            "rgb_cy": "20",
            "thermal_x": "4",
            "thermal_y": "14",
            "thermal_w": "20",
            "thermal_h": "20",
            "thermal_cx": "14",
            "thermal_cy": "24",
            "modal_offset_distance": "5.656854",
        },
        {
            "dataset": "toy",
            "sequence": "SEQ-A",
            "track_id": "1",
            "category_id": "1",
            "category_name": "vehicle",
            "fps": "25",
            "frame_id": "2",
            "rgb_x": "2",
            "rgb_y": "10",
            "rgb_w": "20",
            "rgb_h": "20",
            "rgb_cx": "12",
            "rgb_cy": "20",
            "thermal_x": "6",
            "thermal_y": "14",
            "thermal_w": "20",
            "thermal_h": "20",
            "thermal_cx": "16",
            "thermal_cy": "24",
            "modal_offset_distance": "5.656854",
        },
        {
            "dataset": "toy",
            "sequence": "SEQ-A",
            "track_id": "1",
            "category_id": "1",
            "category_name": "vehicle",
            "fps": "25",
            "frame_id": "3",
            "rgb_x": "4",
            "rgb_y": "10",
            "rgb_w": "20",
            "rgb_h": "20",
            "rgb_cx": "14",
            "rgb_cy": "20",
            "thermal_x": "8",
            "thermal_y": "14",
            "thermal_w": "20",
            "thermal_h": "20",
            "thermal_cx": "18",
            "thermal_cy": "24",
            "modal_offset_distance": "5.656854",
        },
        {
            "dataset": "toy",
            "sequence": "SEQ-A",
            "track_id": "2",
            "category_id": "1",
            "category_name": "vehicle",
            "fps": "25",
            "frame_id": "1",
            "rgb_x": "100",
            "rgb_y": "30",
            "rgb_w": "20",
            "rgb_h": "20",
            "rgb_cx": "110",
            "rgb_cy": "40",
        },
        {
            "dataset": "toy",
            "sequence": "SEQ-A",
            "track_id": "2",
            "category_id": "1",
            "category_name": "vehicle",
            "fps": "25",
            "frame_id": "2",
            "rgb_x": "104",
            "rgb_y": "30",
            "rgb_w": "20",
            "rgb_h": "20",
            "rgb_cx": "114",
            "rgb_cy": "40",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_fused_track_pipeline_writes_linked_outputs_and_manifest(tmp_path: Path) -> None:
    csv_path = tmp_path / "observations_train.csv"
    output_dir = tmp_path / "pipeline"
    _write_observations(csv_path)

    summary = run_fused_track_pipeline(
        csv_path,
        output_dir,
        FusedTrackPipelineConfig(
            split="train",
            group=GroupWindowConfig(
                sample_mode="window",
                window_size=2,
                stride=1,
                min_visible_frames=1,
            ),
        ),
    )

    assert summary["schema_version"] == 1
    assert summary["counts"]["observations"] == 5
    assert summary["counts"]["trajectories"] == 2
    assert summary["counts"]["fused_trajectories"] == 2
    assert summary["counts"]["group_windows"] == 2
    assert summary["modality_coverage"]["paired_points"] == 3
    assert summary["modality_coverage"]["rgb_only_points"] == 2

    fused_path = output_dir / "fused_trajectories_train.jsonl"
    group_path = output_dir / "group_windows_train.jsonl"
    manifest_path = output_dir / "fused_track_pipeline_manifest_train.json"
    assert fused_path.exists()
    assert group_path.exists()
    assert manifest_path.exists()

    fused = _read_jsonl(fused_path)
    track_one = next(item for item in fused if item["track_id"] == "1")
    assert track_one["temporal_linkage"]["frame_ids"] == [1, 2, 3]
    assert track_one["temporal_linkage"]["frame_gaps"] == [1, 1]
    assert track_one["temporal_linkage"]["max_frame_gap"] == 1
    assert track_one["points"][0]["fused"]["center_xy"] == [12.0, 22.0]
    assert track_one["points"][0]["fused"]["source_modalities"] == ["rgb", "thermal"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 1
    assert manifest["pipeline"] == "fused_track_pipeline"
    assert "sha256" in manifest["artifacts"]["fused_trajectories"]
    assert not Path(manifest["inputs"]["observations_csv"]).is_absolute()
