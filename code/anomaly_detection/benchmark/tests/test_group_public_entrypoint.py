from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
ANOMALY_ROOT = REPO_ROOT / "code" / "anomaly_detection"
if str(ANOMALY_ROOT) not in sys.path:
    sys.path.insert(0, str(ANOMALY_ROOT))


def test_group_package_exposes_scoring_entrypoint() -> None:
    from group import COMPONENT_NAMES, score_group_windows

    assert "object_group" in COMPONENT_NAMES
    assert callable(score_group_windows)


def test_group_submodules_reexport_benchmark_implementation() -> None:
    from group.graph import extract_object_states
    from group.scoring import score_group_windows
    from group.tracking import track_groups

    assert callable(extract_object_states)
    assert callable(score_group_windows)
    assert callable(track_groups)


def test_group_cli_help_works_as_public_entrypoint() -> None:
    script = REPO_ROOT / "code" / "anomaly_detection" / "group" / "run_group_method.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--k-neighbors" in result.stdout
