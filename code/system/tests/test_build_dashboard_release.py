from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.build_dashboard_release import build_dashboard_release


def _make_dashboard(root: Path) -> Path:
    dashboard_dir = root / "runs" / "final_results_dashboard" / "final_dashboard"
    assets_dir = dashboard_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dashboard_dir / "index.html").write_text("<!doctype html><title>FusionTrack</title>", encoding="utf-8")
    (assets_dir / "final_dashboard_data.json").write_text("{}", encoding="utf-8")
    (assets_dir / "final_playback_data.json").write_text("{}", encoding="utf-8")
    return dashboard_dir


def test_build_dashboard_release_runs_pipeline_publishes_pages_and_writes_sanitized_manifest(
    tmp_path: Path,
) -> None:
    dashboard_dir = _make_dashboard(tmp_path)
    config_path = tmp_path / "code" / "system" / "configs" / "final_dashboard.local.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('{"work_root": "runs/final_results_dashboard"}', encoding="utf-8")
    export_package = tmp_path / "exports" / "fusiontrack_release.zip"
    pages_dir = tmp_path / "gh-pages"
    calls: list[list[str]] = []

    def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        export_package.parent.mkdir(parents=True)
        export_package.write_bytes(b"fusiontrack zip")
        summary = {
            "mode": "final_results_dashboard",
            "work_root": str(tmp_path / "runs" / "final_results_dashboard"),
            "summary_path": str(tmp_path / "runs" / "final_results_dashboard" / "pipeline_summary_final_dashboard.json"),
            "manifest_path": str(
                tmp_path / "runs" / "final_results_dashboard" / "pipeline_manifest_final_dashboard_all.json"
            ),
            "dashboard": {
                "report_html": str(dashboard_dir / "index.html"),
                "assets_dir": str(dashboard_dir / "assets"),
            },
            "export_package": {"package_path": str(export_package)},
        }
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(summary), stderr="")

    release = build_dashboard_release(
        run_config=config_path,
        pages_dir=pages_dir,
        run_id="20260528_demo",
        export_package=export_package,
        repo_root=tmp_path,
        python_executable="python",
        command_runner=fake_runner,
    )

    assert calls == [
        [
            "python",
            str(tmp_path / "code" / "system" / "run_fusiontrack.py"),
            "--run-config",
            str(config_path),
            "--export-package",
            str(export_package),
        ]
    ]
    assert (pages_dir / "index.html").exists()
    assert (pages_dir / "assets" / "fusiontrack_release.zip").read_bytes() == b"fusiontrack zip"
    assert (pages_dir / "history" / "20260528_demo" / "index.html").exists()
    assert (pages_dir / "history" / "20260528_demo" / "assets" / "fusiontrack_release.zip").exists()
    assert release["run_id"] == "20260528_demo"
    assert release["pipeline"]["mode"] == "final_results_dashboard"
    assert release["pipeline"]["dashboard_dir"] == "runs/final_results_dashboard/final_dashboard"
    assert release["pages_publish"]["archive"]["index"] == "history/20260528_demo/index.html"
    assert "fusiontrack_release.zip" in release["pages_publish"]["copied_assets"]

    manifest_path = tmp_path / "runs" / "final_results_dashboard" / "dashboard_release_20260528_demo.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted == release
    assert str(tmp_path) not in json.dumps(persisted, ensure_ascii=False)


def test_build_dashboard_release_requires_dashboard_output_when_publishing(tmp_path: Path) -> None:
    config_path = tmp_path / "dashboard.json"
    config_path.write_text("{}", encoding="utf-8")

    def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"mode": "final_results_dashboard"}), stderr="")

    with pytest.raises(ValueError, match="dashboard"):
        build_dashboard_release(
            run_config=config_path,
            pages_dir=tmp_path / "gh-pages",
            run_id="demo",
            repo_root=tmp_path,
            command_runner=fake_runner,
        )
