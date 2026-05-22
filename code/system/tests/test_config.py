from __future__ import annotations

from pathlib import Path

from fusiontrack.config import FusionTrackPaths


def test_default_paths_are_relative_for_server_runs() -> None:
    paths = FusionTrackPaths.defaults()

    assert paths.data_root == Path("data") / "VT-Tiny-MOT"
    assert paths.work_root == Path("runs") / "fusiontrack_v1"
    assert not paths.data_root.is_absolute()
    assert not paths.work_root.is_absolute()


def test_split_paths_resolve_under_work_root() -> None:
    paths = FusionTrackPaths.defaults()

    assert paths.observations_csv("train") == (
        Path("runs") / "fusiontrack_v1" / "trajectories" / "observations_train.csv"
    )
    assert paths.fused_jsonl("test") == (
        Path("runs") / "fusiontrack_v1" / "fusion" / "fused_trajectories_test.jsonl"
    )
    assert paths.fused_states_csv("test") == (
        Path("runs") / "fusiontrack_v1" / "fusion" / "fused_states_test.csv"
    )
