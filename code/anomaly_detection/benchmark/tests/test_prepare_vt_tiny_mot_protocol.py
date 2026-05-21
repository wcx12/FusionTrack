from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
