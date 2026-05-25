from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import write_jsonl


def test_run_suite_executes_multiple_matrices_and_aggregates_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/run_suite.py")
    label_individual = tmp_path / "individual_labels.jsonl"
    score_individual = tmp_path / "individual_scores.jsonl"
    label_group = tmp_path / "group_labels.jsonl"
    score_group = tmp_path / "group_scores.jsonl"
    individual_matrix = tmp_path / "individual_matrix.json"
    group_matrix = tmp_path / "group_matrix.json"
    suite_json = tmp_path / "suite.json"
    output_dir = tmp_path / "suite_out"

    write_jsonl(
        label_individual,
        [
            {"sample_id": "seq:a", "label": 0},
            {"sample_id": "seq:b", "label": 1},
        ],
    )
    write_jsonl(
        score_individual,
        [
            {"sample_id": "seq:a", "score": 0.1},
            {"sample_id": "seq:b", "score": 0.9},
        ],
    )
    write_jsonl(
        label_group,
        [
            {"sample_id": "seq:a", "window_id": "0-15", "label": 0},
            {"sample_id": "seq:b", "window_id": "0-15", "label": 1},
        ],
    )
    write_jsonl(
        score_group,
        [
            {"sample_id": "seq:a", "window_id": "0-15", "score": 0.2},
            {"sample_id": "seq:b", "window_id": "0-15", "score": 0.8},
        ],
    )
    individual_matrix.write_text(
        json.dumps(
            {
                "split": "val",
                "seed": 42,
                "label_file": str(label_individual),
                "require_unique_keys": True,
                "require_score_key_match": True,
                "experiments": [
                    {
                        "name": "individual_existing",
                        "task": "existing_scores",
                        "score_file": str(score_individual),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    group_matrix.write_text(
        json.dumps(
            {
                "split": "val",
                "seed": 42,
                "label_file": str(label_group),
                "key_fields": ["sample_id", "window_id"],
                "require_unique_keys": True,
                "require_score_key_match": True,
                "experiments": [
                    {
                        "name": "group_existing",
                        "task": "existing_scores",
                        "score_file": str(score_group),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite_json.write_text(
        json.dumps(
            {
                "suite_name": "unit_suite",
                "matrices": [
                    {"name": "individual", "config_json": str(individual_matrix)},
                    {"name": "group", "config_json": str(group_matrix)},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--suite-json",
            str(suite_json),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output_dir / "suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 1
    assert manifest["suite_name"] == "unit_suite"
    assert [matrix["name"] for matrix in manifest["matrices"]] == ["individual", "group"]
    assert manifest["matrices"][0]["num_runs"] == 1
    assert (output_dir / "individual" / "manifest.json").exists()
    assert (output_dir / "group" / "manifest.json").exists()
    with (output_dir / "aggregate_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["matrix"] for row in rows] == ["individual", "group"]
    assert [row["method"] for row in rows] == ["individual_existing", "group_existing"]
