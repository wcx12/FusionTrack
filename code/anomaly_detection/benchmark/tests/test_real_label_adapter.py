from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_adapters.real_labels import normalize_real_label_rows
from evaluation.io import load_jsonl
from evaluation.schema import validate_label_rows


def test_normalize_real_individual_labels_from_common_aliases() -> None:
    rows = normalize_real_label_rows(
        [
            {
                "sequence_name": "DJI_0001",
                "object_id": "7",
                "is_anomaly": "yes",
                "type": "route_shift",
                "start_frame": "12",
                "end_frame": "18",
            }
        ],
        level="individual",
    )

    assert rows == [
        {
            "sample_id": "DJI_0001:7",
            "sequence": "DJI_0001",
            "track_id": "7",
            "frame_start": 12,
            "frame_end": 18,
            "label": 1,
            "anomaly_type": "route_shift",
            "metadata": {"source": "real_label"},
        }
    ]
    validate_label_rows(rows, key_fields=("sample_id",), require_unique_keys=True)


def test_normalize_real_group_labels_requires_window_key() -> None:
    rows = normalize_real_label_rows(
        [
            {
                "sequence": "DJI_0001",
                "track_id": "7",
                "window": "12-28",
                "label": "0",
            }
        ],
        level="group",
    )

    assert rows[0]["sample_id"] == "DJI_0001:7"
    assert rows[0]["window_id"] == "12-28"
    validate_label_rows(rows, key_fields=("sample_id", "window_id"), require_unique_keys=True)


def test_normalize_real_group_labels_rejects_missing_window_id() -> None:
    with pytest.raises(ValueError, match="window_id"):
        normalize_real_label_rows(
            [{"sequence": "DJI_0001", "track_id": "7", "label": 1}],
            level="group",
        )


def test_prepare_real_labels_cli_writes_schema_compatible_jsonl(tmp_path: Path) -> None:
    input_csv = tmp_path / "real_labels.csv"
    output_jsonl = tmp_path / "labels.jsonl"
    with input_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence", "track_id", "window_id", "is_anomaly", "anomaly_type"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sequence": "DJI_0001",
                "track_id": "7",
                "window_id": "12-28",
                "is_anomaly": "true",
                "anomaly_type": "leave_group",
            }
        )

    result = subprocess.run(
        [
            sys.executable,
            "code/anomaly_detection/benchmark/runners/prepare_real_labels.py",
            "--level",
            "group",
            "--input-labels",
            str(input_csv),
            "--output-labels",
            str(output_jsonl),
        ],
        cwd=Path(__file__).resolve().parents[4],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    labels = load_jsonl(output_jsonl)
    assert labels[0]["label"] == 1
    assert labels[0]["window_id"] == "12-28"
    validate_label_rows(labels, key_fields=("sample_id", "window_id"), require_unique_keys=True)
