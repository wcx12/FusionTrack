from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_adapters.tracking_observations import convert_mot_roots_to_observations


def _write_sequence(root: Path, sequence: str, rows: list[str], fps: int = 25) -> None:
    sequence_dir = root / sequence
    gt_dir = sequence_dir / "gt"
    gt_dir.mkdir(parents=True)
    (sequence_dir / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={sequence}",
                "imDir=img1",
                "imExt=.jpg",
                f"frameRate={fps}",
                f"fps={fps}",
            ]
        ),
        encoding="utf-8",
    )
    (gt_dir / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_mot_family_root_exports_rgb_observations_with_motion(tmp_path: Path) -> None:
    root = tmp_path / "MOT17" / "train"
    _write_sequence(
        root,
        "MOT17-02-FRCNN",
        [
            "1,1,10,20,30,40,1,1,0.9",
            "2,1,20,20,30,40,1,1,0.8",
            "2,2,100,50,20,20,0,1,0.7",
            "3,3,100,50,20,20,1,2,0.7",
        ],
    )
    output_csv = tmp_path / "observations_train.csv"

    summary = convert_mot_roots_to_observations(
        output_csv=output_csv,
        dataset="MOT17",
        rgb_root=root,
        profile="motchallenge",
        split="train",
    )

    rows = _read_rows(output_csv)
    assert summary["num_rows"] == 2
    assert summary["modalities"]["rgb"]["num_observations"] == 2
    assert rows[0]["sequence"] == "MOT17-02-FRCNN"
    assert rows[0]["track_id"] == "1"
    assert rows[0]["rgb_file"] == "MOT17-02-FRCNN/img1/000001.jpg"
    assert rows[0]["rgb_cx"] == "25.0"
    assert rows[0]["thermal_cx"] == ""
    assert rows[1]["rgb_vx_px_per_frame"] == "10.0"
    assert rows[1]["rgb_speed_px_per_second"] == "250.0"
    assert {row["track_id"] for row in rows} == {"1"}


def test_m3ot_paired_roots_merge_rgb_and_thermal_modal_offsets(tmp_path: Path) -> None:
    rgb_root = tmp_path / "M3OT" / "RGB" / "train"
    thermal_root = tmp_path / "M3OT" / "IR" / "train"
    _write_sequence(rgb_root, "M3OT-0001", ["1,7,10,20,20,20,1,1,1.0"])
    _write_sequence(thermal_root, "M3OT-0001", ["1,7,16,24,20,20,1,1,1.0"])
    output_csv = tmp_path / "observations_train.csv"

    summary = convert_mot_roots_to_observations(
        output_csv=output_csv,
        dataset="M3OT",
        rgb_root=rgb_root,
        thermal_root=thermal_root,
        profile="m3ot",
    )

    rows = _read_rows(output_csv)
    assert summary["modalities"]["rgb"]["num_observations"] == 1
    assert summary["modalities"]["thermal"]["num_observations"] == 1
    assert rows[0]["rgb_cx"] == "20.0"
    assert rows[0]["thermal_cx"] == "26.0"
    assert rows[0]["modal_offset_dx_thermal_minus_rgb"] == "6.0"
    assert rows[0]["modal_offset_dy_thermal_minus_rgb"] == "4.0"
    assert rows[0]["modal_offset_distance"] == "7.211102550927978"
    assert float(rows[0]["modal_bbox_iou"]) > 0.38


def test_tracking_converter_cli_writes_summary_json(tmp_path: Path) -> None:
    root = tmp_path / "DanceTrack" / "train"
    _write_sequence(root, "dancetrack0001", ["1,1,5,6,7,8,1,1,1.0"])
    output_csv = tmp_path / "observations_train.csv"
    summary_json = tmp_path / "summary.json"
    script = Path("code/anomaly_detection/benchmark/runners/convert_tracking_dataset_to_observations.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "DanceTrack",
            "--profile",
            "dancetrack",
            "--mot-root",
            str(root),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
        ],
        cwd=Path(__file__).resolve().parents[4],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "DanceTrack" in result.stdout
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = _read_rows(output_csv)
    assert summary["num_rows"] == 1
    assert rows[0]["category_name"] == "person"
    assert rows[0]["rgb_file"] == "dancetrack0001/img1/00000001.jpg"
