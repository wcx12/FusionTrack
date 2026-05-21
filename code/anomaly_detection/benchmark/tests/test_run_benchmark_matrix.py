from __future__ import annotations

from pathlib import Path
import csv
import json
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import write_jsonl


def _object(track_id: str, centers: list[list[float]]) -> dict:
    return {
        "track_id": track_id,
        "states": [
            {"frame_id": index + 1, "fused": {"center_xy": center}}
            for index, center in enumerate(centers)
        ],
    }


def test_run_benchmark_matrix_runs_methods_evaluates_and_summarizes(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    train_group_windows_path = tmp_path / "group_windows_train.jsonl"
    group_windows_path = tmp_path / "group_windows.jsonl"
    train_trajectories_path = tmp_path / "trajectories_train.jsonl"
    trajectories_path = tmp_path / "trajectories.jsonl"
    existing_scores_path = tmp_path / "existing_scores.jsonl"
    config_path = tmp_path / "matrix.json"
    output_dir = tmp_path / "matrix_out"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py")

    write_jsonl(
        label_path,
        [
            {"sample_id": "seq_a:steady", "label": 0},
            {"sample_id": "seq_a:jump", "label": 1},
        ],
    )
    write_jsonl(
        group_windows_path,
        [
            {
                "window_id": "w1",
                "sequence": "seq_a",
                "objects": [
                    _object("steady", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
                    _object("jump", [[0.0, 1.0], [1.0, 1.0], [8.0, 1.0]]),
                ],
            }
        ],
    )
    write_jsonl(
        train_group_windows_path,
        [
            {
                "window_id": "train_w1",
                "sequence": "seq_train",
                "objects": [
                    _object("steady", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
                    _object("jump", [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
                ],
            }
        ],
    )
    write_jsonl(
        trajectories_path,
        [
            {
                "sample_id": "seq_a:steady",
                "sequence": "seq_a",
                "track_id": "steady",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
                    {"frame_id": 2, "fused": {"center_xy": [1.0, 0.0]}},
                    {"frame_id": 3, "fused": {"center_xy": [2.0, 0.0]}},
                ],
            },
            {
                "sample_id": "seq_a:jump",
                "sequence": "seq_a",
                "track_id": "jump",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [0.0, 1.0]}},
                    {"frame_id": 2, "fused": {"center_xy": [1.0, 1.0]}},
                    {"frame_id": 3, "fused": {"center_xy": [8.0, 1.0]}},
                ],
            },
        ],
    )
    write_jsonl(
        train_trajectories_path,
        [
            {
                "sample_id": "seq_train:steady",
                "sequence": "seq_train",
                "track_id": "steady",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
                    {"frame_id": 2, "fused": {"center_xy": [1.0, 0.0]}},
                    {"frame_id": 3, "fused": {"center_xy": [2.0, 0.0]}},
                ],
            },
            {
                "sample_id": "seq_train:jump",
                "sequence": "seq_train",
                "track_id": "jump",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [0.0, 1.0]}},
                    {"frame_id": 2, "fused": {"center_xy": [1.0, 1.0]}},
                    {"frame_id": 3, "fused": {"center_xy": [2.0, 1.0]}},
                ],
            },
        ],
    )
    write_jsonl(
        existing_scores_path,
        [
            {"sample_id": "seq_a:steady", "score": 0.1},
            {"sample_id": "seq_a:jump", "score": 0.9},
        ],
    )
    config_path.write_text(
        json.dumps(
            {
                "split": "val",
                "seed": 13,
                "label_file": str(label_path),
                "k": 1,
                "experiments": [
                    {
                        "name": "group_prediction",
                        "task": "group_prediction",
                        "score_windows": str(group_windows_path),
                    },
                    {
                        "name": "existing",
                        "task": "existing_scores",
                        "score_file": str(existing_scores_path),
                    },
                    {
                        "name": "group_temporal_autoencoder",
                        "task": "group_temporal_autoencoder",
                        "train_windows": str(train_group_windows_path),
                        "score_windows": str(group_windows_path),
                        "n_components": 2,
                        "seed": 13,
                    },
                    {
                        "name": "fusiontrack_individual",
                        "task": "fusiontrack_individual",
                        "train_jsonl": str(train_trajectories_path),
                        "score_jsonl": str(trajectories_path),
                        "n_neighbors": 1,
                    },
                    {
                        "name": "fusiontrack_individual_context",
                        "task": "fusiontrack_individual_context",
                        "train_jsonl": str(train_trajectories_path),
                        "score_jsonl": str(trajectories_path),
                        "train_windows": str(train_group_windows_path),
                        "score_windows": str(group_windows_path),
                        "n_neighbors": 1,
                    },
                    {
                        "name": "individual_complementary",
                        "task": "individual_complementary",
                        "train_jsonl": str(train_trajectories_path),
                        "score_jsonl": str(trajectories_path),
                        "n_neighbors": 1,
                        "contamination": 0.1,
                        "seed": 13,
                    },
                    {
                        "name": "individual_trajectory_lm_ngram",
                        "task": "individual_trajectory_lm",
                        "train_jsonl": str(train_trajectories_path),
                        "score_jsonl": str(trajectories_path),
                        "ngram_order": 2,
                        "alpha": 1.0,
                        "grid_size": 16,
                        "seed": 13,
                    },
                    {
                        "name": "individual_physics",
                        "task": "individual_physics",
                        "train_jsonl": str(train_trajectories_path),
                        "score_jsonl": str(trajectories_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config-json",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [run["name"] for run in manifest["runs"]] == [
        "group_prediction",
        "existing",
        "group_temporal_autoencoder",
        "fusiontrack_individual",
        "fusiontrack_individual_context",
        "individual_complementary",
        "individual_trajectory_lm_ngram",
        "individual_physics",
    ]
    assert (output_dir / "scores" / "group_prediction.jsonl").exists()
    assert (output_dir / "scores" / "fusiontrack_individual_context.jsonl").exists()
    assert (output_dir / "scores" / "individual_trajectory_lm_ngram.jsonl").exists()
    assert (output_dir / "scores" / "individual_physics.jsonl").exists()
    assert (output_dir / "metrics" / "group_prediction.json").exists()
    assert manifest["summary_csv"] == str(output_dir / "summary.csv")

    metrics = json.loads(
        (output_dir / "metrics" / "group_prediction.json").read_text(encoding="utf-8")
    )
    assert metrics["method"] == "group_prediction"
    assert metrics["split"] == "val"
    assert metrics["seed"] == 13
    assert metrics["precision_at_k"] == 1.0

    with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["method"] for row in rows] == [
        "group_prediction",
        "existing",
        "group_temporal_autoencoder",
        "fusiontrack_individual",
        "fusiontrack_individual_context",
        "individual_complementary",
        "individual_trajectory_lm_ngram",
        "individual_physics",
    ]


def test_run_benchmark_matrix_can_evaluate_group_rows_by_object_window(
    tmp_path: Path,
) -> None:
    label_path = tmp_path / "labels.jsonl"
    group_windows_path = tmp_path / "group_windows.jsonl"
    config_path = tmp_path / "matrix.json"
    output_dir = tmp_path / "matrix_out"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py")

    write_jsonl(
        label_path,
        [
            {"sample_id": "seq_a:a", "window_id": "w1", "label": 0},
            {"sample_id": "seq_a:a", "window_id": "w2", "label": 1},
        ],
    )
    write_jsonl(
        group_windows_path,
        [
            {
                "window_id": "w1",
                "sequence": "seq_a",
                "objects": [_object("a", [[0.0, 0.0], [1.0, 0.0], [8.0, 0.0]])],
            },
            {
                "window_id": "w2",
                "sequence": "seq_a",
                "objects": [_object("a", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])],
            },
        ],
    )
    config_path.write_text(
        json.dumps(
            {
                "split": "val",
                "seed": 13,
                "label_file": str(label_path),
                "key_fields": ["sample_id", "window_id"],
                "k": 1,
                "experiments": [
                    {
                        "name": "group_prediction",
                        "task": "group_prediction",
                        "score_windows": str(group_windows_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config-json",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(
        (output_dir / "metrics" / "group_prediction.json").read_text(encoding="utf-8")
    )
    assert metrics["precision_at_k"] == 0.0


def test_run_benchmark_matrix_can_require_unique_alignment_keys(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "scores.jsonl"
    config_path = tmp_path / "matrix.json"
    output_dir = tmp_path / "matrix_out"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py")

    write_jsonl(label_path, [{"sample_id": "a", "label": 1}])
    write_jsonl(
        score_path,
        [
            {"sample_id": "a", "score": 0.2},
            {"sample_id": "a", "score": 0.9},
        ],
    )
    config_path.write_text(
        json.dumps(
            {
                "split": "val",
                "seed": 13,
                "label_file": str(label_path),
                "require_unique_keys": True,
                "experiments": [
                    {
                        "name": "existing",
                        "task": "existing_scores",
                        "score_file": str(score_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config-json",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Duplicate score keys" in result.stderr


def test_run_benchmark_matrix_can_require_exact_score_key_match(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "scores.jsonl"
    config_path = tmp_path / "matrix.json"
    output_dir = tmp_path / "matrix_out"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py")

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
    config_path.write_text(
        json.dumps(
            {
                "split": "val",
                "seed": 13,
                "label_file": str(label_path),
                "require_score_key_match": True,
                "experiments": [
                    {
                        "name": "existing",
                        "task": "existing_scores",
                        "score_file": str(score_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config-json",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Score keys do not exactly match label keys" in result.stderr


def test_run_benchmark_matrix_help_works_as_direct_script() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--config-json" in result.stdout


@pytest.mark.parametrize(
    "experiment",
    [
        {
            "name": "group_iforest",
            "task": "group_classical",
            "method": "isolation_forest",
        },
        {
            "name": "group_temporal_autoencoder",
            "task": "group_temporal_autoencoder",
            "n_components": 2,
        },
    ],
)
def test_run_benchmark_matrix_requires_group_train_windows(
    tmp_path: Path,
    experiment: dict,
) -> None:
    label_path = tmp_path / "labels.jsonl"
    group_windows_path = tmp_path / "group_windows.jsonl"
    config_path = tmp_path / "matrix.json"
    output_dir = tmp_path / "matrix_out"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_benchmark_matrix.py")

    write_jsonl(label_path, [{"sample_id": "seq_a:steady", "label": 0}])
    write_jsonl(
        group_windows_path,
        [
            {
                "window_id": "w1",
                "sequence": "seq_a",
                "objects": [
                    _object("steady", [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
                ],
            }
        ],
    )
    experiment = dict(experiment)
    experiment["score_windows"] = str(group_windows_path)
    config_path.write_text(
        json.dumps(
            {
                "label_file": str(label_path),
                "experiments": [experiment],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config-json",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing required config field 'train_windows'" in result.stderr
