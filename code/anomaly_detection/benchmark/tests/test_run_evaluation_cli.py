from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import write_jsonl


def test_run_evaluation_cli_writes_metrics_with_metadata(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "scores.jsonl"
    output_json = tmp_path / "metrics.json"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_evaluation.py")

    write_jsonl(
        label_path,
        [
            {"sample_id": "a", "label": 1},
            {"sample_id": "b", "label": 0},
            {"sample_id": "c", "label": 1},
        ],
    )
    write_jsonl(
        score_path,
        [
            {"sample_id": "a", "score": 0.9},
            {"sample_id": "b", "score": 0.1},
            {"sample_id": "c", "score": 0.8},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--label-file",
            str(label_path),
            "--score-file",
            str(score_path),
            "--output-json",
            str(output_json),
            "--method",
            "toy_method",
            "--split",
            "val",
            "--seed",
            "7",
            "--k",
            "2",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(output_json.read_text(encoding="utf-8"))
    assert metrics["method"] == "toy_method"
    assert metrics["split"] == "val"
    assert metrics["seed"] == 7
    assert metrics["source"] == str(score_path)
    assert metrics["num_positive"] == 2
    assert metrics["precision_at_k"] == 1.0


def test_run_evaluation_cli_can_enforce_strict_alignment(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "scores.jsonl"
    output_json = tmp_path / "metrics.json"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_evaluation.py")

    write_jsonl(
        label_path,
        [
            {"sample_id": "a", "label": 1},
            {"sample_id": "b", "label": 0},
        ],
    )
    write_jsonl(
        score_path,
        [
            {"sample_id": "a", "score": 0.9},
            {"sample_id": "extra", "score": 0.1},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--label-file",
            str(label_path),
            "--score-file",
            str(score_path),
            "--output-json",
            str(output_json),
            "--method",
            "toy_method",
            "--split",
            "val",
            "--seed",
            "7",
            "--require-unique-keys",
            "--require-score-key-match",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Score keys do not exactly match label keys" in result.stderr
    assert not output_json.exists()


def test_run_evaluation_cli_fails_fast_on_invalid_score_schema(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "scores.jsonl"
    output_json = tmp_path / "metrics.json"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_evaluation.py")

    write_jsonl(label_path, [{"sample_id": "a", "label": 1}])
    write_jsonl(score_path, [{"sample_id": "a", "score": "nan"}])

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--label-file",
            str(label_path),
            "--score-file",
            str(score_path),
            "--output-json",
            str(output_json),
            "--method",
            "toy_method",
            "--split",
            "val",
            "--seed",
            "7",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "finite numeric score" in result.stderr
    assert not output_json.exists()


def test_run_evaluation_cli_help_works_as_direct_script() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_evaluation.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--score-file" in result.stdout
    assert "--require-score-key-match" in result.stdout
