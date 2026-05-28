from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "system_completion_status.json"
CHECKLIST_PATH = ROOT / "system_completion_checklist.md"


def test_system_completion_status_schema() -> None:
    payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert set(payload["status_values"]) == {"done", "partial", "pending"}
    modules = payload["modules"]
    assert len(modules) >= 8

    seen_ids: set[str] = set()
    for module in modules:
        assert module["id"] not in seen_ids
        seen_ids.add(module["id"])
        assert module["status"] in payload["status_values"]
        assert module["name"].strip()
        assert module["current_evidence"]
        assert module["remaining_work"]


def test_system_completion_checklist_covers_all_modules() -> None:
    payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    checklist = CHECKLIST_PATH.read_text(encoding="utf-8")

    for module in payload["modules"]:
        assert module["name"] in checklist
