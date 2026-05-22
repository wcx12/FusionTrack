from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.prepare_vt_tiny_mot_holdout_protocol import _experiments_for_split
from runners.prepare_vt_tiny_mot_protocol import DEFAULT_EXPERIMENTS


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
