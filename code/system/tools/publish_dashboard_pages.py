#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter or digit and contain only letters, digits, '.', '_' or '-'."
        )
    return run_id


def _resolve_existing_dashboard(source_dir: Path) -> Path:
    source = source_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dashboard source directory does not exist: {source_dir}")
    if not (source / "index.html").is_file():
        raise FileNotFoundError(f"Dashboard source is missing index.html: {source_dir}")
    if not (source / "assets").is_dir():
        raise FileNotFoundError(f"Dashboard source is missing assets directory: {source_dir}")
    return source


def _resolve_pages_dir(pages_dir: Path) -> Path:
    pages = pages_dir.expanduser().resolve()
    pages.mkdir(parents=True, exist_ok=True)
    if not pages.is_dir():
        raise NotADirectoryError(f"Pages target is not a directory: {pages_dir}")
    return pages


def _ensure_inside(base: Path, target: Path) -> Path:
    base_resolved = base.resolve()
    target_resolved = target.resolve()
    if target_resolved == base_resolved or base_resolved in target_resolved.parents:
        return target_resolved
    raise ValueError(f"Refusing to write outside pages directory: {target}")


def _relative_posix(base: Path, target: Path) -> str:
    return target.resolve().relative_to(base.resolve()).as_posix()


def _replace_assets(source_assets: Path, target_assets: Path, pages_dir: Path) -> list[str]:
    safe_target = _ensure_inside(pages_dir, target_assets)
    if safe_target.exists():
        shutil.rmtree(safe_target)
    shutil.copytree(source_assets, safe_target)
    return sorted(path.relative_to(safe_target).as_posix() for path in safe_target.rglob("*") if path.is_file())


def _publish_to_target(source_dir: Path, target_dir: Path, pages_dir: Path) -> list[str]:
    safe_target = _ensure_inside(pages_dir, target_dir)
    safe_target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "index.html", safe_target / "index.html")
    return _replace_assets(source_dir / "assets", safe_target / "assets", pages_dir)


def publish_dashboard_to_pages(
    source_dir: Path | str,
    pages_dir: Path | str,
    *,
    run_id: str | None = None,
    history_dir: str = "history",
    update_root: bool = True,
) -> dict[str, Any]:
    """Publish a generated FusionTrack dashboard into a GitHub Pages worktree."""

    source = _resolve_existing_dashboard(Path(source_dir))
    pages = _resolve_pages_dir(Path(pages_dir))
    validated_run_id = _validate_run_id(run_id) if run_id else None
    root_assets: list[str] = []
    archive_assets: list[str] = []

    if update_root:
        root_assets = _publish_to_target(source, pages, pages)

    archive: dict[str, str] | None = None
    if validated_run_id:
        archive_target = _ensure_inside(pages, pages / history_dir / validated_run_id)
        if archive_target.exists():
            shutil.rmtree(archive_target)
        archive_assets = _publish_to_target(source, archive_target, pages)
        archive = {
            "index": _relative_posix(pages, archive_target / "index.html"),
            "assets": _relative_posix(pages, archive_target / "assets"),
        }

    copied_assets = root_assets if update_root else archive_assets
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "published_at_utc": _utc_now_iso(),
        "source_dashboard": source.name,
        "run_id": validated_run_id,
        "root": {
            "updated": update_root,
            "index": "index.html" if update_root else None,
            "assets": "assets" if update_root else None,
        },
        "archive": archive,
        "copied_asset_count": len(copied_assets),
        "copied_assets": copied_assets,
    }
    (pages / "publish_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a generated FusionTrack dashboard directory to a GitHub Pages worktree."
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Generated dashboard directory.")
    parser.add_argument("--pages-dir", type=Path, required=True, help="GitHub Pages worktree or static root.")
    parser.add_argument("--run-id", help="Optional version id archived under history/<run-id>.")
    parser.add_argument("--history-dir", default="history", help="Relative history directory name.")
    parser.add_argument("--no-root", action="store_true", help="Only archive the run; do not update root index/assets.")
    args = parser.parse_args()

    manifest = publish_dashboard_to_pages(
        args.source_dir,
        args.pages_dir,
        run_id=args.run_id,
        history_dir=args.history_dir,
        update_root=not args.no_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
