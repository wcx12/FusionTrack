from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.publish_dashboard_pages import publish_dashboard_to_pages


def _make_dashboard(root: Path) -> Path:
    dashboard_dir = root / "final_dashboard"
    assets_dir = dashboard_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dashboard_dir / "index.html").write_text("<!doctype html><title>FusionTrack</title>", encoding="utf-8")
    (assets_dir / "final_dashboard_data.json").write_text('{"ok": true}', encoding="utf-8")
    (assets_dir / "final_playback_data.json").write_text('{"sequences": []}', encoding="utf-8")
    (assets_dir / "background_demo.jpg").write_bytes(b"fake-jpeg")
    return dashboard_dir


def test_publish_dashboard_updates_pages_root_and_run_history_without_absolute_paths(tmp_path: Path) -> None:
    source_dir = _make_dashboard(tmp_path / "source")
    pages_dir = tmp_path / "pages"
    stale_asset = pages_dir / "assets" / "old_asset.json"
    stale_asset.parent.mkdir(parents=True)
    stale_asset.write_text("stale", encoding="utf-8")
    (pages_dir / "CNAME").write_text("example.com\n", encoding="utf-8")

    manifest = publish_dashboard_to_pages(source_dir, pages_dir, run_id="20260528_demo")

    assert (pages_dir / "index.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (pages_dir / "assets" / "final_dashboard_data.json").exists()
    assert (pages_dir / "assets" / "final_playback_data.json").exists()
    assert not stale_asset.exists()
    assert (pages_dir / "CNAME").read_text(encoding="utf-8") == "example.com\n"

    archive_dir = pages_dir / "history" / "20260528_demo"
    assert (archive_dir / "index.html").exists()
    assert (archive_dir / "assets" / "background_demo.jpg").read_bytes() == b"fake-jpeg"

    manifest_path = pages_dir / "publish_manifest.json"
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest == manifest
    assert persisted_manifest["run_id"] == "20260528_demo"
    assert persisted_manifest["root"]["index"] == "index.html"
    assert persisted_manifest["archive"]["index"] == "history/20260528_demo/index.html"
    assert persisted_manifest["copied_asset_count"] == 3

    serialized = json.dumps(persisted_manifest, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert not Path(persisted_manifest["source_dashboard"]).is_absolute()


def test_publish_dashboard_rejects_path_traversal_run_id(tmp_path: Path) -> None:
    source_dir = _make_dashboard(tmp_path / "source")

    with pytest.raises(ValueError, match="run_id"):
        publish_dashboard_to_pages(source_dir, tmp_path / "pages", run_id="../bad")
