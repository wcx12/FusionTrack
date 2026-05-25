from __future__ import annotations

import json
from pathlib import Path

from fusiontrack.method_registry import method_profile, validate_method_registry


def test_validate_method_registry_reports_duplicate_aliases_and_missing_fields(tmp_path: Path) -> None:
    registry_path = tmp_path / "method_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "methods": [
                    {
                        "task": "group",
                        "name": "method_a",
                        "owner": "our_method",
                        "role": "component",
                        "method_family": "family_a",
                        "learning_type": "non_learning",
                        "source_type": "fusiontrack",
                        "status": "integrated",
                        "aliases": ["shared_alias"],
                    },
                    {
                        "task": "group",
                        "name": "method_b",
                        "owner": "",
                        "role": "component",
                        "method_family": "family_b",
                        "learning_type": "non_learning",
                        "source_type": "fusiontrack",
                        "status": "integrated",
                        "aliases": ["shared_alias"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_method_registry(registry_path)

    assert report["status"] == "invalid"
    assert report["num_methods"] == 2
    assert any("missing required field owner" in error for error in report["errors"])
    assert any("duplicate alias" in error for error in report["errors"])


def test_method_profile_marks_registered_and_unregistered_methods() -> None:
    registered = method_profile("fusiontrack_group_graph", "group")
    missing = method_profile("not_registered", "group")

    assert registered["registry_status"] == "registered"
    assert registered["owner"] == "our_method"
    assert missing["registry_status"] == "unregistered"
    assert missing["owner"] == "unregistered"
