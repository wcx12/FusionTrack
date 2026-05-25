from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.prepare_vt_tiny_mot_holdout_protocol import _experiments_for_split
from runners.prepare_vt_tiny_mot_protocol import (
    DEFAULT_EXPERIMENTS,
    _prepare_anomaly_data_args,
    _write_protocol_dataset_manifest,
)


def _write_annotation(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "000001.jpg"}],
                "annotations": [{"id": 1, "image_id": 1}],
                "videos": [{"id": 1, "name": "seq_a"}],
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )


def test_protocol_dataset_manifest_records_requested_splits(tmp_path: Path) -> None:
    data_root = tmp_path / "VT-Tiny-MOT"
    for split in ("train", "test"):
        _write_annotation(data_root / "annotations" / f"instances_00_{split}2017.json")
        _write_annotation(data_root / "annotations" / f"instances_01_{split}2017.json")
    (data_root / "train2017" / "seq_a").mkdir(parents=True)
    (data_root / "train2017" / "seq_a" / "000001.jpg").write_bytes(b"jpg")
    output_root = tmp_path / "protocol"

    manifest, manifest_path = _write_protocol_dataset_manifest(
        data_root=data_root,
        output_root=output_root,
        splits=("train", "test"),
    )

    assert manifest_path == output_root / "dataset_manifest.json"
    assert manifest_path.exists()
    assert manifest["status"] == "ok"
    assert set(manifest["splits"]) == {"train", "test"}
    assert len(manifest["dataset_fingerprint"]) == 64


def test_prepare_anomaly_data_args_bind_dataset_manifest() -> None:
    args = _prepare_anomaly_data_args(
        level="individual",
        input_jsonl=Path("input.jsonl"),
        output_jsonl=Path("output.jsonl"),
        labels_jsonl=Path("labels.jsonl"),
        anomaly_fraction=0.1,
        seed=42,
        manifest_json=Path("injection_manifest.json"),
        dataset_manifest_json=Path("dataset_manifest.json"),
    )

    assert "--dataset-manifest-json" in args
    assert args[args.index("--dataset-manifest-json") + 1] == "dataset_manifest.json"


def test_default_group_matrix_includes_learning_fusiontrack_knn() -> None:
    methods = {
        experiment["name"]: experiment
        for experiment in DEFAULT_EXPERIMENTS["group"]
    }

    experiment = methods["fusiontrack_group_temporal_knn"]
    assert experiment["task"] == "fusiontrack_group_temporal_knn"
    assert experiment["train_windows"] == "group_windows_train.jsonl"
    assert experiment["score_windows"] == "group_windows_val.jsonl"
    assert experiment["n_neighbors"] == 3


def test_default_matrix_includes_enhanced_fusiontrack_candidates() -> None:
    individual_methods = {
        experiment["name"]: experiment
        for experiment in DEFAULT_EXPERIMENTS["individual"]
    }
    group_methods = {
        experiment["name"]: experiment
        for experiment in DEFAULT_EXPERIMENTS["group"]
    }

    individual = individual_methods["fusiontrack_individual_ensemble_calibrated"]
    assert individual["task"] == "fusiontrack_individual_ensemble"
    assert individual["calibration_columns"] == ["mean_speed", "duration_frames", "num_points"]
    assert individual["calibration_bins"] == 4
    assert individual["calibration_global_weight"] == 0.5

    group = group_methods["fusiontrack_group_hybrid_gated"]
    assert group["task"] == "fusiontrack_group_hybrid"
    assert group["use_residual_gate"] is True
    assert group["residual_gate_power"] == 2.0
    assert group["residual_gate_floor"] == 0.05


def test_default_matrix_includes_validation_tuned_fusiontrack_candidates() -> None:
    individual_methods = {
        experiment["name"]: experiment
        for experiment in DEFAULT_EXPERIMENTS["individual"]
    }
    group_methods = {
        experiment["name"]: experiment
        for experiment in DEFAULT_EXPERIMENTS["group"]
    }

    individual_auprc = individual_methods["fusiontrack_individual_ensemble_tuned_auprc"]
    assert individual_auprc["task"] == "fusiontrack_individual_ensemble"
    assert individual_auprc["nearest_weight"] == 0.45
    assert individual_auprc["lof_weight"] == 0.45
    assert individual_auprc["iforest_weight"] == 0.10
    assert individual_auprc["calibration_columns"] == [
        "mean_speed",
        "duration_frames",
        "num_points",
    ]
    assert individual_auprc["calibration_global_weight"] == 0.3
    assert individual_auprc["selection_scope"] == "validation_score_grid"

    individual_topk = individual_methods["fusiontrack_individual_ensemble_tuned_topk"]
    assert individual_topk["nearest_weight"] == 0.60
    assert individual_topk["lof_weight"] == 0.30
    assert individual_topk["iforest_weight"] == 0.10
    assert individual_topk["selection_scope"] == "validation_score_grid"

    group_auroc = group_methods["fusiontrack_group_hybrid_tuned_auroc_topk"]
    assert group_auroc["task"] == "fusiontrack_group_hybrid"
    assert group_auroc["prediction_weight"] == 0.50
    assert group_auroc["graph_weight"] == 0.25
    assert group_auroc["temporal_weight"] == 0.25
    assert group_auroc["use_residual_gate"] is False
    assert group_auroc["selection_scope"] == "validation_score_grid"

    group_auprc = group_methods["fusiontrack_group_hybrid_tuned_auprc_f1"]
    assert group_auprc["prediction_weight"] == 0.60
    assert group_auprc["graph_weight"] == 0.30
    assert group_auprc["temporal_weight"] == 0.10
    assert group_auprc["use_residual_gate"] is False
    assert group_auprc["selection_scope"] == "validation_score_grid"

    group_fine_auprc = group_methods["fusiontrack_group_hybrid_tuned_fine_auprc"]
    assert group_fine_auprc["prediction_weight"] == 0.47
    assert group_fine_auprc["graph_weight"] == 0.41
    assert group_fine_auprc["temporal_weight"] == 0.12
    assert group_fine_auprc["use_residual_gate"] is False
    assert group_fine_auprc["selection_scope"] == "validation_fine_weight_search"

    group_fine_topk = group_methods["fusiontrack_group_hybrid_tuned_fine_topk"]
    assert group_fine_topk["prediction_weight"] == 0.45
    assert group_fine_topk["graph_weight"] == 0.43
    assert group_fine_topk["temporal_weight"] == 0.12
    assert group_fine_topk["use_residual_gate"] is False
    assert group_fine_topk["selection_scope"] == "validation_fine_weight_search"

    group_fine_f1 = group_methods["fusiontrack_group_hybrid_tuned_fine_f1"]
    assert group_fine_f1["prediction_weight"] == 0.46
    assert group_fine_f1["graph_weight"] == 0.42
    assert group_fine_f1["temporal_weight"] == 0.12
    assert group_fine_f1["use_residual_gate"] is False
    assert group_fine_f1["selection_scope"] == "validation_fine_weight_search"


def test_holdout_experiments_rewrite_score_split_and_seed_without_mutating_defaults() -> None:
    rewritten = _experiments_for_split(
        DEFAULT_EXPERIMENTS["individual"],
        split_name="test",
        seed=44,
    )
    methods = {experiment["name"]: experiment for experiment in rewritten}

    tuned = methods["fusiontrack_individual_ensemble_tuned_auprc"]
    assert tuned["score_jsonl"] == "fused_trajectories_test.jsonl"
    assert tuned["train_jsonl"] == "fused_trajectories_train.jsonl"
    assert tuned["seed"] == 44

    context = methods["fusiontrack_individual_context"]
    assert context["score_windows"] == "group_windows_test_clean.jsonl"

    original_methods = {
        experiment["name"]: experiment
        for experiment in DEFAULT_EXPERIMENTS["individual"]
    }
    assert original_methods["fusiontrack_individual_ensemble_tuned_auprc"]["seed"] == 42
    assert (
        original_methods["fusiontrack_individual_ensemble_tuned_auprc"]["score_jsonl"]
        == "fused_trajectories_val.jsonl"
    )
