from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import load_jsonl, write_jsonl
from external_sources.lmtad_adapters import (
    convert_lmtad_scores_to_jsonl,
    write_lmtad_official_inputs,
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


def test_write_lmtad_official_inputs_writes_manifest_sequences_and_vocab(tmp_path: Path) -> None:
    output_dir = tmp_path / "lmtad"

    manifest = write_lmtad_official_inputs(
        [
            _trajectory("seq:a", [[0.0, 0.0], [25.0, 0.0], [50.0, 25.0]]),
            _trajectory("seq:b", [[0.0, 0.0], [0.0, 25.0]]),
        ],
        output_dir=output_dir,
        grid_size=25.0,
    )

    sequence_rows = load_jsonl(output_dir / "lmtad_sequences.jsonl")
    vocab = json.loads((output_dir / "vocab.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "fusiontrack.lmtad_official_inputs.v1"
    assert manifest["official_repository"] == "https://github.com/jonathankabala/LMTAD"
    assert "custom dataset loader" in manifest["external_checkout_integration"].lower()
    assert manifest["files"]["sequence_jsonl"] == "lmtad_sequences.jsonl"
    assert manifest["files"]["vocab_json"] == "vocab.json"
    assert [row["sample_id"] for row in sequence_rows] == ["seq:a", "seq:b"]
    assert sequence_rows[0]["tokens"] == ["cell_0_0", "cell_1_0", "cell_2_1"]
    assert sequence_rows[0]["token_ids"] == [3, 4, 5]
    assert sequence_rows[0]["frames"] == [1, 2, 3]
    assert sequence_rows[0]["centers"] == [[0.0, 0.0], [25.0, 0.0], [50.0, 25.0]]
    assert vocab["PAD"] == 0
    assert vocab["EOT"] == 1
    assert vocab["SOT"] == 2
    assert vocab["cell_0_0"] == 3


def test_convert_lmtad_scores_to_jsonl_maps_trajectory_ids_with_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "lmtad"
    manifest_path = output_dir / "manifest.json"
    scores_csv = tmp_path / "official_scores.csv"
    output_jsonl = tmp_path / "scores.jsonl"
    write_lmtad_official_inputs(
        [
            _trajectory("seq:a", [[0.0, 0.0], [25.0, 0.0]]),
            _trajectory("seq:b", [[0.0, 0.0], [0.0, 25.0]]),
        ],
        output_dir=output_dir,
    )
    with scores_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trajectory_id", "anomaly_score"])
        writer.writeheader()
        writer.writerow({"trajectory_id": 0, "anomaly_score": 0.25})
        writer.writerow({"trajectory_id": 1, "anomaly_score": 1.5})

    rows = convert_lmtad_scores_to_jsonl(
        scores_path=scores_csv,
        output_jsonl=output_jsonl,
        manifest_json=manifest_path,
    )

    assert [row["sample_id"] for row in rows] == ["seq:a", "seq:b"]
    assert [row["score"] for row in rows] == [0.25, 1.5]
    assert rows[0]["source"] == "official_lmtad:lm"
    assert rows[1]["component_scores"] == {"official_lmtad_anomaly_score": 1.5}
    assert rows[1]["metadata"]["official_source"] == "LM-TAD"
    assert load_jsonl(output_jsonl) == rows


def test_convert_lmtad_scores_to_jsonl_accepts_jsonl_sample_ids_and_nll(tmp_path: Path) -> None:
    scores_jsonl = tmp_path / "official_scores.jsonl"
    output_jsonl = tmp_path / "scores.jsonl"
    write_jsonl(
        scores_jsonl,
        [
            {"sample_id": "seq:a", "track_id": "a", "nll": 2.25},
            {"sample_id": "seq:b", "track_id": "b", "nll": "3.5"},
        ],
    )

    rows = convert_lmtad_scores_to_jsonl(
        scores_path=scores_jsonl,
        output_jsonl=output_jsonl,
    )

    assert [row["sample_id"] for row in rows] == ["seq:a", "seq:b"]
    assert [row["track_id"] for row in rows] == ["a", "b"]
    assert [row["score"] for row in rows] == [2.25, 3.5]
    assert rows[0]["component_scores"] == {"official_lmtad_nll": 2.25}


def test_convert_lmtad_scores_rejects_missing_explicit_id_column(tmp_path: Path) -> None:
    scores_jsonl = tmp_path / "official_scores.jsonl"
    output_jsonl = tmp_path / "scores.jsonl"
    write_jsonl(scores_jsonl, [{"sample_id": "seq:a", "score": 1.0}])

    try:
        convert_lmtad_scores_to_jsonl(
            scores_path=scores_jsonl,
            output_jsonl=output_jsonl,
            id_column="trajectory_id",
        )
    except ValueError as exc:
        assert "missing explicit ID column trajectory_id" in str(exc)
    else:
        raise AssertionError("expected missing explicit id column to fail")


def test_convert_lmtad_scores_rejects_ambiguous_score_columns(tmp_path: Path) -> None:
    scores_jsonl = tmp_path / "official_scores.jsonl"
    output_jsonl = tmp_path / "scores.jsonl"
    write_jsonl(scores_jsonl, [{"sample_id": "seq:a", "score": 1.0, "nll": 2.0}])

    try:
        convert_lmtad_scores_to_jsonl(scores_path=scores_jsonl, output_jsonl=output_jsonl)
    except ValueError as exc:
        assert "multiple candidate score columns" in str(exc)
    else:
        raise AssertionError("expected ambiguous score columns to fail")

    rows = convert_lmtad_scores_to_jsonl(
        scores_path=scores_jsonl,
        output_jsonl=output_jsonl,
        score_column="nll",
    )

    assert rows[0]["score"] == 2.0


def test_lmtad_adapter_clis_work_as_direct_scripts(tmp_path: Path) -> None:
    trajectories_jsonl = tmp_path / "trajectories.jsonl"
    output_dir = tmp_path / "prepared"
    scores_json = tmp_path / "official_scores.json"
    output_jsonl = tmp_path / "scores.jsonl"
    repo_root = Path(__file__).resolve().parents[4]
    prepare_script = Path("code/anomaly_detection/benchmark/runners/prepare_lmtad_official_inputs.py")
    convert_script = Path("code/anomaly_detection/benchmark/runners/convert_lmtad_official_scores.py")
    write_jsonl(
        trajectories_jsonl,
        [_trajectory("seq:a", [[0.0, 0.0], [25.0, 0.0]])],
    )

    help_result = subprocess.run(
        [sys.executable, str(prepare_script), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "external LMTAD checkout" in help_result.stdout

    prepare = subprocess.run(
        [
            sys.executable,
            str(prepare_script),
            "--trajectory-jsonl",
            str(trajectories_jsonl),
            "--output-dir",
            str(output_dir),
            "--grid-size",
            "25",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert prepare.returncode == 0, prepare.stderr
    assert json.loads(prepare.stdout)["num_trajectories"] == 1

    scores_json.write_text(
        json.dumps([{"trajectory_id": 0, "score": 0.75}]),
        encoding="utf-8",
    )
    convert = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            "--lmtad-scores",
            str(scores_json),
            "--manifest-json",
            str(output_dir / "manifest.json"),
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
