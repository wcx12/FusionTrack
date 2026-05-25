from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.io import load_jsonl, write_jsonl


def test_prepare_anomaly_data_injects_individual_rows_and_labels(tmp_path: Path) -> None:
    input_path = tmp_path / "trajectories.jsonl"
    output_path = tmp_path / "trajectories_injected.jsonl"
    label_path = tmp_path / "labels.jsonl"
    manifest_path = tmp_path / "individual_manifest.json"
    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/prepare_anomaly_data.py")

    write_jsonl(
        input_path,
        [
            {
                "sample_id": "seq_a:t1",
                "sequence": "seq_a",
                "track_id": "t1",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
                    {"frame_id": 2, "fused": {"center_xy": [1.0, 1.0]}},
                ],
            }
        ],
    )
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "VT-Tiny-MOT",
                "status": "ok",
                "dataset_fingerprint": "dataset-fp-001",
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--level",
            "individual",
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--labels-jsonl",
            str(label_path),
            "--anomaly-fraction",
            "1.0",
            "--seed",
            "7",
            "--anomaly-types",
            "route_shift",
            "--manifest-json",
            str(manifest_path),
            "--dataset-manifest-json",
            str(dataset_manifest_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    labels = load_jsonl(label_path)
    assert len(labels) == 1
    assert labels[0]["sample_id"] == "seq_a:t1"
    assert labels[0]["anomaly_type"] == "route_shift"
    assert load_jsonl(output_path)[0]["points"] != load_jsonl(input_path)[0]["points"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_schema_version"] == 2
    assert manifest["kind"] == "synthetic_anomaly_protocol"
    assert manifest["protocol"]["level"] == "individual"
    assert manifest["protocol"]["key_fields"] == ["sample_id"]
    assert manifest["protocol"]["anomaly_fraction"] == 1.0
    assert manifest["protocol"]["anomaly_types"] == ["route_shift"]
    assert manifest["artifacts"]["input_jsonl"]["sha256"] == hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()
    assert manifest["artifacts"]["output_jsonl"]["sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    assert manifest["artifacts"]["labels_jsonl"]["sha256"] == hashlib.sha256(
        label_path.read_bytes()
    ).hexdigest()
    assert manifest["label_distribution"]["num_positive"] == 1
    assert manifest["label_distribution"]["by_anomaly_type"]["route_shift"] == 1
    assert manifest["dataset_manifest"]["dataset_fingerprint"] == "dataset-fp-001"
    assert manifest["dataset_manifest"]["sha256"] == hashlib.sha256(
        dataset_manifest_path.read_bytes()
    ).hexdigest()
    assert "--anomaly-fraction" in manifest["replay"]["argv"]


def test_prepare_anomaly_data_injects_group_rows_and_labels(tmp_path: Path) -> None:
    input_path = tmp_path / "windows.jsonl"
    output_path = tmp_path / "windows_injected.jsonl"
    label_path = tmp_path / "labels.jsonl"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/prepare_anomaly_data.py")

    write_jsonl(
        input_path,
        [
            {
                "window_id": "seq_a:1-3",
                "sequence": "seq_a",
                "objects": [
                    {
                        "sample_id": "seq_a:a",
                        "track_id": "a",
                        "states": [
                            {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
                            {"frame_id": 2, "fused": {"center_xy": [1.0, 0.0]}},
                        ],
                    },
                    {
                        "sample_id": "seq_a:b",
                        "track_id": "b",
                        "states": [
                            {"frame_id": 1, "fused": {"center_xy": [0.0, 1.0]}},
                            {"frame_id": 2, "fused": {"center_xy": [1.0, 1.0]}},
                        ],
                    },
                ],
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--level",
            "group",
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--labels-jsonl",
            str(label_path),
            "--anomaly-fraction",
            "1.0",
            "--seed",
            "11",
            "--anomaly-types",
            "leave_group",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    labels = load_jsonl(label_path)
    assert len(labels) == 2
    positive_labels = [label for label in labels if label["label"] == 1]
    negative_labels = [label for label in labels if label["label"] == 0]
    assert len(positive_labels) == 1
    assert len(negative_labels) == 1
    assert positive_labels[0]["metadata"]["source"] == "group_injection"
    assert negative_labels[0]["anomaly_type"] == "normal"
    assert load_jsonl(output_path)[0]["objects"] != load_jsonl(input_path)[0]["objects"]


def test_prepare_anomaly_data_keeps_group_labels_at_object_window_level(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "windows.jsonl"
    output_path = tmp_path / "windows_injected.jsonl"
    label_path = tmp_path / "labels.jsonl"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/prepare_anomaly_data.py")

    base_object = {
        "sample_id": "seq_a:a",
        "track_id": "a",
        "states": [
            {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
            {"frame_id": 2, "fused": {"center_xy": [1.0, 0.0]}},
        ],
    }
    write_jsonl(
        input_path,
        [
            {"window_id": "w1", "sequence": "seq_a", "objects": [base_object]},
            {"window_id": "w2", "sequence": "seq_a", "objects": [base_object]},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--level",
            "group",
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--labels-jsonl",
            str(label_path),
            "--anomaly-fraction",
            "0.0",
            "--seed",
            "17",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    labels = load_jsonl(label_path)
    assert [(label["sample_id"], label["window_id"]) for label in labels] == [
        ("seq_a:a", "w1"),
        ("seq_a:a", "w2"),
    ]
    assert {label["label"] for label in labels} == {0}


def test_prepare_anomaly_data_help_works_as_direct_script() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/prepare_anomaly_data.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--anomaly-fraction" in result.stdout
