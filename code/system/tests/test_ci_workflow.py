from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_system_ci_workflow_covers_dashboard_and_registry_checks() -> None:
    workflow = REPO_ROOT / ".github" / "workflows" / "system-ci.yml"

    assert workflow.exists()

    text = workflow.read_text(encoding="utf-8")
    assert "code/system/tests" in text
    assert "validate_method_registry" in text
    assert "py_compile" in text
    assert "final_dashboard.py" in text
    assert "code/system/tools/build_sample_dashboard.py" in text
    assert "actions/upload-artifact" in text
    assert "sample-dashboard" in text
    assert "workflow_dispatch" in text
