from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.method_registry import load_method_registry


def test_default_method_registry_describes_main_methods() -> None:
    registry = load_method_registry()

    individual = registry.profile_for("fusiontrack_individual_ensemble_tuned_auprc")
    assert individual["task"] == "individual"
    assert individual["owner"] == "our_method"
    assert individual["role"] == "proposed_method"
    assert individual["method_family"] == "fusiontrack_rank_ensemble"
    assert individual["learning_type"] == "non_learning_rank_fusion"

    group = registry.profile_for("fusiontrack_group_hybrid_tuned_auroc_topk")
    assert group["task"] == "group"
    assert group["owner"] == "our_method"
    assert group["role"] == "proposed_method"
    assert group["method_family"] == "fusiontrack_group_hybrid"

    registration = registry.profile_for("mps_gaf_learned_svd", task="registration")
    assert registration["task"] == "registration"
    assert registration["owner"] == "our_method"
    assert registration["role"] == "proposed_registration"
    assert registration["learning_type"] == "learning_deep_registration"


def test_method_registry_fails_on_duplicate_task_method_keys(tmp_path: Path) -> None:
    path = tmp_path / "method_registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "methods": [
                    {"task": "individual", "name": "duplicate", "owner": "a"},
                    {"task": "individual", "name": "duplicate", "owner": "b"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate method registry entry"):
        load_method_registry(path)
