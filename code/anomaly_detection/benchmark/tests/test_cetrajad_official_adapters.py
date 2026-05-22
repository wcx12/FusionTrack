from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import load_jsonl, write_jsonl
from external_sources.cetrajad_adapters import (
    convert_cetrajad_scores_to_jsonl,
    write_cetrajad_official_input_bundle,
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


def test_write_cetrajad_input_bundle_exports_pickle_manifest_and_sidecar(tmp_path: Path) -> None:
    output_dir = tmp_path / "cetrajad_inputs"

    manifest = write_cetrajad_official_input_bundle(
        [
            _trajectory("seq:a", [[0.0, 0.0], [1.0, 2.0]]),
            _trajectory("seq:b", [[3.0, 4.0]]),
        ],
        output_dir=output_dir,
        dict_name="fusiontrack",
    )

    assert manifest["dict_name"] == "fusiontrack"
    assert manifest["num_trajectories"] == 2
    assert manifest["official_repository"] == "https://github.com/ShuruiCao/comp-ensemble-ad"
    assert "external CETrajAD checkout" in manifest["usage_note"]

    sidecar = json.loads((output_dir / "cetrajad_sidecar.json").read_text(encoding="utf-8"))
    assert [row["sample_id"] for row in sidecar] == ["seq:a", "seq:b"]
    assert sidecar[0]["trajectory_id"] == 0

    trip_dict = pd.read_pickle(output_dir / "fusiontrack_evaluation_gps.pkl")
    assert sorted(trip_dict) == [0, 1]
    assert trip_dict[0][["timestamp", "latitude", "longitude"]].to_dict("records") == [
        {"timestamp": 1, "latitude": 0.0, "longitude": 0.0},
        {"timestamp": 2, "latitude": 2.0, "longitude": 1.0},
    ]
    assert trip_dict[1][["latitude", "longitude"]].to_dict("records") == [
        {"latitude": 4.0, "longitude": 3.0},
        {"latitude": 4.0, "longitude": 3.0},
    ]


def test_convert_cetrajad_scores_to_jsonl_accepts_common_csv_columns(tmp_path: Path) -> None:
    sidecar_json = tmp_path / "sidecar.json"
    scores_csv = tmp_path / "scores.csv"
    output_jsonl = tmp_path / "scores.jsonl"
    sidecar_json.write_text(
        json.dumps(
            [
                {"trajectory_id": 0, "sample_id": "seq:a", "sequence": "seq", "track_id": "a"},
                {"trajectory_id": 1, "sample_id": "seq:b", "sequence": "seq", "track_id": "b"},
            ]
        ),
        encoding="utf-8",
    )
    with scores_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trajectory_id", "anomaly_score"])
        writer.writeheader()
        writer.writerow({"trajectory_id": 0, "anomaly_score": 0.5})
        writer.writerow({"trajectory_id": 1, "anomaly_score": 2.0})

    rows = convert_cetrajad_scores_to_jsonl(scores_csv, sidecar_json, output_jsonl)

    assert [row["sample_id"] for row in rows] == ["seq:a", "seq:b"]
    assert [row["score"] for row in rows] == [0.5, 2.0]
    assert rows[0]["source"] == "official_cetrajad:comp_ensemble"
    assert rows[1]["component_scores"] == {"cetrajad_official_score": 2.0}
    assert load_jsonl(output_jsonl) == rows


def test_convert_cetrajad_scores_to_jsonl_accepts_jsonl_sample_ids(tmp_path: Path) -> None:
    sidecar_json = tmp_path / "sidecar.json"
    scores_jsonl = tmp_path / "scores.jsonl"
    output_jsonl = tmp_path / "converted.jsonl"
    sidecar_json.write_text(
        json.dumps(
            [
                {"trajectory_id": 0, "sample_id": "seq:a", "sequence": "seq", "track_id": "a"},
                {"trajectory_id": 1, "sample_id": "seq:b", "sequence": "seq", "track_id": "b"},
            ]
        ),
        encoding="utf-8",
    )
    write_jsonl(
        scores_jsonl,
        [
            {"sample_id": "seq:b", "score": 4.0},
            {"sample_id": "seq:a", "score": 1.25},
        ],
    )

    rows = convert_cetrajad_scores_to_jsonl(scores_jsonl, sidecar_json, output_jsonl)

    assert [row["sample_id"] for row in rows] == ["seq:b", "seq:a"]
    assert [row["track_id"] for row in rows] == ["b", "a"]
    assert [row["score"] for row in rows] == [4.0, 1.25]


def test_convert_cetrajad_scores_requires_sidecar_match_for_trajectory_ids(tmp_path: Path) -> None:
    sidecar_json = tmp_path / "sidecar.json"
    scores_csv = tmp_path / "scores.csv"
    output_jsonl = tmp_path / "scores.jsonl"
    sidecar_json.write_text(
        json.dumps([{"trajectory_id": 0, "sample_id": "seq:a"}]),
        encoding="utf-8",
    )
    scores_csv.write_text("trajectory_id,score\n7,0.5\n", encoding="utf-8")

    try:
        convert_cetrajad_scores_to_jsonl(scores_csv, sidecar_json, output_jsonl)
    except ValueError as exc:
        assert "unknown trajectory_id 7" in str(exc)
    else:
        raise AssertionError("expected unknown trajectory_id to fail")


def test_convert_cetrajad_scores_rejects_duplicate_track_id_sidecar_match(tmp_path: Path) -> None:
    sidecar_json = tmp_path / "sidecar.json"
    scores_csv = tmp_path / "scores.csv"
    output_jsonl = tmp_path / "scores.jsonl"
    sidecar_json.write_text(
        json.dumps(
            [
                {"trajectory_id": 0, "sample_id": "seq1:a", "sequence": "seq1", "track_id": "a"},
                {"trajectory_id": 1, "sample_id": "seq2:a", "sequence": "seq2", "track_id": "a"},
            ]
        ),
        encoding="utf-8",
    )
    scores_csv.write_text("track_id,score\na,0.5\n", encoding="utf-8")

    try:
        convert_cetrajad_scores_to_jsonl(scores_csv, sidecar_json, output_jsonl)
    except ValueError as exc:
        assert "unknown track_id a" in str(exc)
    else:
        raise AssertionError("expected duplicate track_id to fail")


def test_convert_cetrajad_scores_rejects_ambiguous_score_columns(tmp_path: Path) -> None:
    scores_csv = tmp_path / "scores.csv"
    output_jsonl = tmp_path / "scores.jsonl"
    scores_csv.write_text("sample_id,score,anomaly_score\nseq:a,0.5,1.5\n", encoding="utf-8")

    try:
        convert_cetrajad_scores_to_jsonl(scores_csv, None, output_jsonl)
    except ValueError as exc:
        assert "multiple candidate score columns" in str(exc)
    else:
        raise AssertionError("expected ambiguous score columns to fail")

    rows = convert_cetrajad_scores_to_jsonl(
        scores_csv,
        None,
        output_jsonl,
        score_column="anomaly_score",
    )

    assert rows[0]["score"] == 1.5


def test_convert_cetrajad_scores_supports_single_json_object(tmp_path: Path) -> None:
    scores_json = tmp_path / "scores.json"
    output_jsonl = tmp_path / "scores.jsonl"
    scores_json.write_text(
        json.dumps({"sample_id": "seq:a", "score": 2.0}),
        encoding="utf-8",
    )

    rows = convert_cetrajad_scores_to_jsonl(scores_json, None, output_jsonl)

    assert rows[0]["sample_id"] == "seq:a"
    assert rows[0]["score"] == 2.0


def test_cetrajad_adapter_clis_work_as_direct_scripts(tmp_path: Path) -> None:
    trajectories_jsonl = tmp_path / "trajectories.jsonl"
    output_dir = tmp_path / "cetrajad_inputs"
    scores_csv = tmp_path / "scores.csv"
    output_jsonl = tmp_path / "scores.jsonl"
    repo_root = Path(__file__).resolve().parents[4]
    prepare_script = Path("code/anomaly_detection/benchmark/runners/prepare_cetrajad_official_inputs.py")
    convert_script = Path("code/anomaly_detection/benchmark/runners/convert_cetrajad_official_scores.py")
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
            "--output-dir",
            str(output_dir),
            "--dict-name",
            "fusiontrack",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert prepare.returncode == 0, prepare.stderr
    assert json.loads(prepare.stdout)["num_trajectories"] == 1
    assert (output_dir / "fusiontrack_evaluation_gps.pkl").exists()

    scores_csv.write_text("sample_id,score\nseq:a,0.75\n", encoding="utf-8")
    convert = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            "--cetrajad-scores",
            str(scores_csv),
            "--sidecar-json",
            str(output_dir / "cetrajad_sidecar.json"),
            "--output-jsonl",
            str(output_jsonl),
            "--score-column",
            "score",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert convert.returncode == 0, convert.stderr
    assert json.loads(convert.stdout)["num_scores"] == 1
    assert load_jsonl(output_jsonl)[0]["score"] == 0.75
