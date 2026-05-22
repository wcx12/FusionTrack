from __future__ import annotations

import csv
import json
from pathlib import Path

from fusiontrack.experiment_adapter import load_experiment_result, write_scores_csv


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_experiment_result_reads_manifest_scores_labels_and_metrics(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    _write_jsonl(
        result_dir / "scores" / "fusiontrack_individual_nn.jsonl",
        [
            {
                "sample_id": "S1:7",
                "sequence": "S1",
                "track_id": "7",
                "score": 0.82,
                "source": "fusiontrack_individual:nearest_feature",
                "component_scores": {"nearest_feature_distance": 4.2},
            },
            {
                "sample_id": "S2:9",
                "sequence": "S2",
                "track_id": "9",
                "score": 0.12,
                "source": "fusiontrack_individual:nearest_feature",
            },
        ],
    )
    _write_jsonl(
        result_dir / "labels.jsonl",
        [
            {
                "sample_id": "S1:7",
                "sequence": "S1",
                "track_id": "7",
                "frame_start": 12,
                "frame_end": 24,
                "label": 1,
                "anomaly_type": "speed_spike",
                "injection_seed": 42,
                "metadata": {"source": "individual_injection"},
            }
        ],
    )
    (result_dir / "metrics").mkdir(parents=True)
    (result_dir / "metrics" / "fusiontrack_individual_nn.json").write_text(
        json.dumps(
            {
                "method": "fusiontrack_individual_nn",
                "task": "fusiontrack_individual",
                "split": "local_smoke_individual",
                "seed": 42,
                "auroc": 0.91,
                "auprc": 0.83,
                "f1": 0.76,
                "precision_at_k": 1.0,
                "recall_at_k": 0.9,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "split": "local_smoke_individual",
        "seed": 42,
        "label_file": "labels.jsonl",
        "runs": [
            {
                "name": "fusiontrack_individual_nn",
                "task": "fusiontrack_individual",
                "score_file": "scores/fusiontrack_individual_nn.jsonl",
                "metrics_file": "metrics/fusiontrack_individual_nn.json",
            }
        ],
    }
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = load_experiment_result(manifest_path, method_name="fusiontrack_individual_nn")

    assert result.method_name == "fusiontrack_individual_nn"
    assert result.task == "fusiontrack_individual"
    assert result.split == "local_smoke_individual"
    assert result.seed == 42
    assert result.scores_by_sample["S1:7"]["score"] == 0.82
    assert result.labels_by_sample["S1:7"][0]["anomaly_type"] == "speed_spike"
    assert result.metrics["auroc"] == 0.91
    assert result.to_report_context()["summary"]["num_positive_labels"] == 1


def test_write_scores_csv_exports_visualization_compatible_rows(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    _write_jsonl(
        result_dir / "scores" / "method_a.jsonl",
        [
            {
                "sample_id": "S1:7",
                "sequence": "S1",
                "track_id": "7",
                "category_id": 1,
                "category_name": "ship",
                "score": 0.82,
            }
        ],
    )
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "split": "test",
                "seed": 7,
                "runs": [
                    {
                        "name": "method_a",
                        "task": "individual",
                        "score_file": "scores/method_a.jsonl",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = load_experiment_result(manifest_path)
    output_csv = tmp_path / "scores.csv"

    summary = write_scores_csv(result, output_csv)

    assert summary == {"output_csv": str(output_csv), "num_scores": 1, "method": "method_a"}
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "sample_id": "S1:7",
            "sequence": "S1",
            "track_id": "7",
            "category_id": "1",
            "category_name": "ship",
            "score": "0.82",
            "used_sources": "method_a",
        }
    ]
