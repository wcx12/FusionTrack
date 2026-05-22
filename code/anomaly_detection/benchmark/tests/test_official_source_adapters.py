from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import load_jsonl, write_jsonl
from external_sources.official_adapters import (
    convert_pidpm_scores_to_jsonl,
    write_pidpm_trajectory_csv,
)


def _trajectory(sample_id: str, centers: list[list[float]]) -> dict:
    sequence, track_id = sample_id.split(":", 1)
    return {
        "sample_id": sample_id,
        "sequence": sequence,
        "track_id": track_id,
        "points": [
            {"frame_id": index + 1, "fused": {"center_xy": center}}
            for index, center in enumerate(centers)
        ],
    }


def test_write_pidpm_trajectory_csv_flattens_and_writes_sidecar(tmp_path: Path) -> None:
    output_csv = tmp_path / "pidpm.csv"
    sidecar_json = tmp_path / "sidecar.json"

    sidecar = write_pidpm_trajectory_csv(
        [
            _trajectory("seq:a", [[0.0, 0.0], [1.0, 0.0]]),
            _trajectory("seq:b", [[2.0, 2.0]]),
        ],
        output_csv=output_csv,
        sidecar_json=sidecar_json,
        max_points=3,
    )

    matrix = np.loadtxt(output_csv, delimiter=",")
    assert matrix.shape == (2, 6)
    assert matrix[0].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert matrix[1].tolist() == [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    assert [row["sample_id"] for row in sidecar] == ["seq:a", "seq:b"]
    assert json.loads(sidecar_json.read_text(encoding="utf-8")) == sidecar


def test_convert_pidpm_scores_to_jsonl_uses_sidecar_mapping(tmp_path: Path) -> None:
    sidecar_json = tmp_path / "sidecar.json"
    scores_csv = tmp_path / "anomaly_scores.csv"
    output_jsonl = tmp_path / "scores.jsonl"
    sidecar_json.write_text(
        json.dumps(
            [
                {"trajectory_id": 0, "sample_id": "seq:a", "sequence": "seq", "track_id": "a", "max_points": 3},
                {"trajectory_id": 1, "sample_id": "seq:b", "sequence": "seq", "track_id": "b", "max_points": 3},
            ]
        ),
        encoding="utf-8",
    )
    with scores_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trajectory_id", "anomaly_score"])
        writer.writeheader()
        writer.writerow({"trajectory_id": 0, "anomaly_score": 0.25})
        writer.writerow({"trajectory_id": 1, "anomaly_score": 1.5})

    rows = convert_pidpm_scores_to_jsonl(scores_csv, sidecar_json, output_jsonl)

    assert [row["sample_id"] for row in rows] == ["seq:a", "seq:b"]
    assert [row["score"] for row in rows] == [0.25, 1.5]
    assert rows[0]["source"] == "official_pidpm:diffusion"
    assert rows[1]["component_scores"] == {"diffusion_reconstruction_mse": 1.5}
    assert load_jsonl(output_jsonl) == rows


def test_pidpm_adapter_clis_work_as_direct_scripts(tmp_path: Path) -> None:
    trajectories_jsonl = tmp_path / "trajectories.jsonl"
    pidpm_csv = tmp_path / "pidpm.csv"
    sidecar_json = tmp_path / "sidecar.json"
    scores_csv = tmp_path / "anomaly_scores.csv"
    output_jsonl = tmp_path / "scores.jsonl"
    repo_root = Path(__file__).resolve().parents[4]
    prepare_script = Path("code/anomaly_detection/benchmark/runners/prepare_pidpm_official_inputs.py")
    convert_script = Path("code/anomaly_detection/benchmark/runners/convert_pidpm_official_scores.py")
    write_jsonl(
        trajectories_jsonl,
        [_trajectory("seq:a", [[0.0, 0.0], [1.0, 0.0]])],
    )

    prepare = subprocess.run(
        [
            sys.executable,
            str(prepare_script),
            "--trajectory-jsonl",
            str(trajectories_jsonl),
            "--output-csv",
            str(pidpm_csv),
            "--sidecar-json",
            str(sidecar_json),
            "--max-points",
            "2",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert prepare.returncode == 0, prepare.stderr
    assert json.loads(prepare.stdout)["num_trajectories"] == 1

    scores_csv.write_text("trajectory_id,anomaly_score\n0,0.75\n", encoding="utf-8")
    convert = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            "--pidpm-scores-csv",
            str(scores_csv),
            "--sidecar-json",
            str(sidecar_json),
            "--output-jsonl",
            str(output_jsonl),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert convert.returncode == 0, convert.stderr
    assert json.loads(convert.stdout)["num_scores"] == 1
    assert load_jsonl(output_jsonl)[0]["score"] == 0.75
