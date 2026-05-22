from __future__ import annotations

import csv
import json
from pathlib import Path

from fusiontrack.fusion import ModalityState, fuse_centers, fuse_observations_csv


def state(x: float, y: float) -> ModalityState:
    return ModalityState(center_xy=(x, y), bbox_xywh=(None, None, None, None), speed_px_per_frame=None)


def test_fuse_centers_averages_paired_modalities() -> None:
    center, components = fuse_centers(state(10.0, 20.0), state(14.0, 24.0))

    assert center == (12.0, 22.0)
    assert components["num_modalities"] == 2.0
    assert components["confidence"] > 0.55


def test_fuse_centers_single_modality_has_lower_confidence() -> None:
    paired_center, paired_components = fuse_centers(state(10.0, 20.0), state(11.0, 21.0))
    single_center, single_components = fuse_centers(state(10.0, 20.0), None)

    assert paired_center == (10.5, 20.5)
    assert single_center == (10.0, 20.0)
    assert single_components["confidence"] < paired_components["confidence"]


def test_fuse_centers_penalizes_large_modal_offset() -> None:
    _, close_components = fuse_centers(state(10.0, 20.0), state(11.0, 21.0))
    _, far_components = fuse_centers(state(10.0, 20.0), state(80.0, 90.0))

    assert far_components["confidence"] < close_components["confidence"]


def test_fuse_observations_csv_writes_jsonl_and_flat_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "observations_test.csv"
    jsonl_path = tmp_path / "fused_trajectories_test.jsonl"
    states_csv = tmp_path / "fused_states_test.csv"

    fieldnames = [
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
    rows = [
        {
            "sequence": "S1",
            "track_id": "7",
            "category_id": "1",
            "category_name": "ship",
            "fps": "30",
            "frame_id": "0",
            "rgb_x": "0",
            "rgb_y": "0",
            "rgb_w": "10",
            "rgb_h": "10",
            "rgb_cx": "10",
            "rgb_cy": "20",
            "thermal_x": "0",
            "thermal_y": "0",
            "thermal_w": "10",
            "thermal_h": "10",
            "thermal_cx": "14",
            "thermal_cy": "24",
            "modal_offset_distance": "5.656854",
        },
        {
            "sequence": "S1",
            "track_id": "7",
            "category_id": "1",
            "category_name": "ship",
            "fps": "30",
            "frame_id": "1",
            "rgb_x": "1",
            "rgb_y": "1",
            "rgb_w": "10",
            "rgb_h": "10",
            "rgb_cx": "11",
            "rgb_cy": "21",
            "thermal_x": "",
            "thermal_y": "",
            "thermal_w": "",
            "thermal_h": "",
            "thermal_cx": "",
            "thermal_cy": "",
            "modal_offset_distance": "",
        },
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = fuse_observations_csv(csv_path, jsonl_path, states_csv)

    assert summary["num_fused_trajectories"] == 1
    assert summary["num_fused_states"] == 2
    payload = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["sample_id"] == "S1:7"
    assert payload["points"][0]["fused"]["center_xy"] == [12.0, 22.0]
    assert payload["points"][0]["fused"]["source_modalities"] == ["rgb", "thermal"]
    assert payload["points"][1]["fused"]["source_modalities"] == ["rgb"]
    assert states_csv.exists()
