from __future__ import annotations

from pathlib import Path
import re

from fusiontrack.config import FusionTrackPaths
from fusiontrack.pipeline import build_extraction_command


def test_extraction_command_uses_relative_server_paths() -> None:
    paths = FusionTrackPaths.defaults()
    command = build_extraction_command(paths, "test")
    command_text = " ".join(command)

    assert "data/VT-Tiny-MOT" in command_text.replace("\\", "/")
    assert "runs/fusiontrack_v1/trajectories" in command_text.replace("\\", "/")
    assert re.search(r"[A-Za-z]:[\\/]", command_text) is None
