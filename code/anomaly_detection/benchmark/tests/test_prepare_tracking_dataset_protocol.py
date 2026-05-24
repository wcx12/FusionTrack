from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_adapters.tracking_observations import OBSERVATION_COLUMNS


def _write_observations_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence, offset in (("seq_a", 0.0), ("seq_b", 100.0)):
        for track_id, y in (("1", 0.0), ("2", 10.0)):
            for frame_id in range(1, 5):
                x = offset + float(frame_id)
                rows.append(
                    {
                        "dataset": "TinyMOTFamily",
                        "sequence": sequence,
                        "track_id": track_id,
                        "category_id": 1,
                        "category_name": "person",
                        "fps": 25,
                        "frame_id": frame_id,
                        "rgb_file": f"{sequence}/img1/{frame_id:06d}.jpg",
                        "rgb_x": x,
                        "rgb_y": y,
                        "rgb_w": 10,
                        "rgb_h": 20,
                        "rgb_cx": x + 5,
                        "rgb_cy": y + 10,
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(OBSERVATION_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_prepare_tracking_dataset_protocol_validation_from_observations(tmp_path: Path) -> None:
    observations_csv = tmp_path / "observations_train.csv"
    output_root = tmp_path / "protocol"
    _write_observations_csv(observations_csv)
    script = Path("code/anomaly_detection/benchmark/runners/prepare_tracking_dataset_protocol.py")

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "TinyMOTFamily",
            "--mode",
            "validation",
            "--observations-csv",
            str(observations_csv),
            "--output-root",
            str(output_root),
            "--window-size",
            "2",
            "--stride",
            "1",
            "--seed",
            "42",
        ],
        cwd=Path(__file__).resolve().parents[4],
        text=True,
        capture_output=True,
        check=True,
    )

    manifest = json.loads((output_root / "protocol_manifest.json").read_text(encoding="utf-8"))
    individual_matrix = json.loads((output_root / "individual_val_matrix.json").read_text(encoding="utf-8"))
    group_matrix = json.loads((output_root / "group_val_matrix.json").read_text(encoding="utf-8"))

    assert manifest["dataset"] == "TinyMOTFamily"
    assert manifest["mode"] == "validation"
    assert manifest["num_fused_train"] > 0
    assert manifest["num_fused_eval_clean"] > 0
    assert manifest["num_group_train"] > 0
    assert manifest["num_group_eval_clean"] > 0
    assert individual_matrix["require_score_key_match"] is True
    assert group_matrix["key_fields"] == ["sample_id", "window_id"]
    assert (output_root / "individual_labels_val.jsonl").exists()
    assert (output_root / "group_labels_val.jsonl").exists()
