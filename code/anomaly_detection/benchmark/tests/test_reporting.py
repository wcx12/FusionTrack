from pathlib import Path
import csv
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import load_jsonl, load_score_rows, write_jsonl
from evaluation.reporting import evaluate_score_file, summarize_metric_files


def test_evaluate_score_file_writes_json_metrics(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "scores.jsonl"
    output_json = tmp_path / "metrics.json"

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
            {"sample_id": "b", "score": 0.2},
            {"sample_id": "c", "score": 0.8},
        ],
    )

    metrics = evaluate_score_file(score_path, label_path, output_json=output_json, k=2)

    assert output_json.exists()
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved == metrics
    assert metrics["num_positive"] == 2
    assert metrics["num_total"] == 3
    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["num_label_rows"] == 3
    assert metrics["num_score_rows"] == 3
    assert metrics["num_duplicate_label_keys"] == 0
    assert metrics["num_duplicate_score_keys"] == 0
    assert metrics["num_missing_score_keys"] == 0
    assert metrics["num_extra_score_keys"] == 0
    assert metrics["schema_diagnostics"]["status"] == "ok"
    assert metrics["schema_diagnostics"]["key_fields"] == ["sample_id"]
    assert metrics["schema_diagnostics"]["label"]["num_rows"] == 3
    assert metrics["schema_diagnostics"]["score"]["field_coverage"]["score"] == {
        "present": 3,
        "missing": 0,
    }


def test_summarize_metric_files_preserves_metadata_and_writes_csv(tmp_path: Path) -> None:
    first = tmp_path / "method_a_metrics.json"
    second = tmp_path / "method_b_metrics.json"
    output_csv = tmp_path / "summary.csv"

    first.write_text(
        json.dumps(
            {
                "method": "baseline_a",
                "source": "scores_a.jsonl",
                "split": "val",
                "seed": 7,
                "auroc": 0.75,
                "f1": 0.8,
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "method": "baseline_b",
                "source": "scores_b.jsonl",
                "split": "test",
                "seed": 11,
                "auroc": None,
                "f1": 0.5,
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_metric_files([first, second], output_csv=output_csv)

    assert rows == [
        {
            "method": "baseline_a",
            "source": "scores_a.jsonl",
            "split": "val",
            "seed": 7,
            "auroc": 0.75,
            "f1": 0.8,
        },
        {
            "method": "baseline_b",
            "source": "scores_b.jsonl",
            "split": "test",
            "seed": 11,
            "auroc": None,
            "f1": 0.5,
        },
    ]
    with output_csv.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["method"] == "baseline_a"
    assert csv_rows[0]["split"] == "val"
    assert csv_rows[1]["method"] == "baseline_b"
    assert csv_rows[1]["auroc"] == ""


def test_load_score_rows_converts_csv_numeric_fields(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.csv"
    score_path.write_text(
        "sample_id,label,frame_start,frame_end,injection_seed,score\n"
        "seq_1:track_1,1,10,20,7,0.875\n",
        encoding="utf-8",
    )

    rows = load_score_rows(score_path)

    assert rows == [
        {
            "sample_id": "seq_1:track_1",
            "label": 1,
            "frame_start": 10,
            "frame_end": 20,
            "injection_seed": 7,
            "score": 0.875,
        }
    ]
    assert isinstance(rows[0]["label"], int)
    assert isinstance(rows[0]["frame_start"], int)
    assert isinstance(rows[0]["frame_end"], int)
    assert isinstance(rows[0]["injection_seed"], int)
    assert isinstance(rows[0]["score"], float)


def test_load_jsonl_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.jsonl"
    path.write_bytes("\ufeff{\"sample_id\":\"a\",\"score\":1.0}\n".encode("utf-8"))

    assert load_jsonl(path) == [{"sample_id": "a", "score": 1.0}]


def test_evaluate_score_file_reports_missing_key_fields(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "scores.jsonl"

    write_jsonl(label_path, [{"sample_id": "a", "label": 1}])
    write_jsonl(score_path, [{"score": 0.9}])

    with pytest.raises(ValueError, match="score row 1 is missing required key field 'sample_id'"):
        evaluate_score_file(score_path, label_path)


def test_evaluate_score_file_can_require_unique_alignment_keys(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "scores.jsonl"

    write_jsonl(label_path, [{"sample_id": "a", "label": 1}])
    write_jsonl(
        score_path,
        [
            {"sample_id": "a", "score": 0.2},
            {"sample_id": "a", "score": 0.9},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate score keys"):
        evaluate_score_file(score_path, label_path, require_unique_keys=True)


def test_evaluate_score_file_can_require_exact_score_key_match(tmp_path: Path) -> None:
    label_path = tmp_path / "labels.jsonl"
    score_path = tmp_path / "scores.jsonl"

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
            {"sample_id": "extra", "score": 0.2},
        ],
    )

    with pytest.raises(ValueError, match="Score keys do not exactly match label keys"):
        evaluate_score_file(
            score_path,
            label_path,
            require_score_key_match=True,
        )
