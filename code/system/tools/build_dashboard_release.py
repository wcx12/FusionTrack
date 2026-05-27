#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from tools.publish_dashboard_pages import publish_dashboard_to_pages


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_json_from_stdout(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError("run_fusiontrack.py did not print a JSON summary")
    payload = json.loads(stdout[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("run_fusiontrack.py JSON summary must be an object")
    return payload


def _sanitize_path(value: str | Path | None, repo_root: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return f"${{external}}/{path.name}"


def _sanitize_command(command: list[str], repo_root: Path) -> list[str]:
    sanitized: list[str] = []
    for item in command:
        if item in {"--run-config", "--export-package"}:
            sanitized.append(item)
            continue
        sanitized.append(_sanitize_path(item, repo_root) or item)
    return sanitized


def _dashboard_dir_from_summary(summary: dict[str, Any]) -> Path | None:
    dashboard = summary.get("dashboard")
    if isinstance(dashboard, dict):
        report_html = dashboard.get("report_html")
        if report_html:
            return Path(str(report_html)).parent
        assets_dir = dashboard.get("assets_dir")
        if assets_dir:
            return Path(str(assets_dir)).parent
    report = summary.get("report")
    if isinstance(report, dict):
        report_html = report.get("report_html")
        if report_html:
            return Path(str(report_html)).parent
    work_root = summary.get("work_root")
    if work_root:
        candidate = Path(str(work_root)) / "final_dashboard"
        if candidate.exists():
            return candidate
    return None


def _copy_export_package_to_dashboard_assets(
    summary: dict[str, Any],
    dashboard_dir: Path | None,
    export_path: Path | None,
) -> str | None:
    package_value: Any = export_path
    if isinstance(summary.get("export_package"), dict):
        package_value = summary["export_package"].get("package_path") or package_value
    if package_value in (None, "") or dashboard_dir is None:
        return None
    package_path = Path(str(package_value))
    if not package_path.exists() or not package_path.is_file():
        return None
    assets_dir = dashboard_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    target = assets_dir / package_path.name
    if package_path.resolve() != target.resolve():
        shutil.copy2(package_path, target)
    return f"assets/{package_path.name}"


def _release_manifest_path(summary: dict[str, Any], dashboard_dir: Path | None, run_id: str | None) -> Path:
    manifest_name = f"dashboard_release_{run_id or 'latest'}.json"
    work_root = summary.get("work_root")
    if work_root:
        return Path(str(work_root)) / manifest_name
    if dashboard_dir is not None:
        return dashboard_dir.parent / manifest_name
    return Path(manifest_name)


def _run_pipeline(
    command: list[str],
    *,
    repo_root: Path,
    command_runner: CommandRunner | None,
) -> subprocess.CompletedProcess[str]:
    runner = command_runner or subprocess.run
    return runner(command, cwd=repo_root, capture_output=True, text=True, check=False)


def build_dashboard_release(
    *,
    run_config: Path | str,
    pages_dir: Path | str | None = None,
    run_id: str | None = None,
    export_package: Path | str | None = None,
    repo_root: Path | str | None = None,
    python_executable: str | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root is not None else _repo_root_from_script()
    config_path = Path(run_config).resolve()
    command = [
        python_executable or sys.executable,
        str(repo / "code" / "system" / "run_fusiontrack.py"),
        "--run-config",
        str(config_path),
    ]
    export_path = Path(export_package).resolve() if export_package is not None else None
    if export_path is not None:
        command.extend(["--export-package", str(export_path)])

    result = _run_pipeline(command, repo_root=repo, command_runner=command_runner)
    if result.returncode != 0:
        raise RuntimeError(f"run_fusiontrack.py failed with exit code {result.returncode}: {result.stderr}")

    summary = _parse_json_from_stdout(result.stdout)
    dashboard_dir = _dashboard_dir_from_summary(summary)
    if pages_dir is not None and dashboard_dir is None:
        raise ValueError("Cannot publish dashboard release because pipeline summary does not include dashboard output.")
    dashboard_export_href = _copy_export_package_to_dashboard_assets(summary, dashboard_dir, export_path)
    if dashboard_export_href and isinstance(summary.get("dashboard"), dict):
        summary["dashboard"]["export_package_href"] = dashboard_export_href

    pages_publish = None
    if pages_dir is not None:
        pages_publish = publish_dashboard_to_pages(dashboard_dir, pages_dir, run_id=run_id)

    dashboard = summary.get("dashboard") if isinstance(summary.get("dashboard"), dict) else {}
    release: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "command": _sanitize_command(command, repo),
        "pipeline": {
            "mode": summary.get("mode"),
            "work_root": _sanitize_path(summary.get("work_root"), repo),
            "summary_path": _sanitize_path(summary.get("summary_path"), repo),
            "manifest_path": _sanitize_path(summary.get("manifest_path"), repo),
            "dashboard_dir": _sanitize_path(dashboard_dir, repo),
            "dashboard_index": _sanitize_path(dashboard.get("report_html"), repo),
            "assets_dir": _sanitize_path(dashboard.get("assets_dir"), repo),
            "dashboard_export_package_href": dashboard_export_href,
            "export_package": _sanitize_path(
                (summary.get("export_package") or {}).get("package_path")
                if isinstance(summary.get("export_package"), dict)
                else export_path,
                repo,
            ),
        },
        "pages_publish": pages_publish,
    }

    manifest_path = _release_manifest_path(summary, dashboard_dir, run_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, export and optionally publish a FusionTrack dashboard release.")
    parser.add_argument("--run-config", type=Path, required=True, help="run_fusiontrack.py JSON config.")
    parser.add_argument("--pages-dir", type=Path, help="Optional GitHub Pages worktree/static root.")
    parser.add_argument("--run-id", help="Release id used by release manifest and Pages history archive.")
    parser.add_argument("--export-package", type=Path, help="Optional portable zip export package path.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used for run_fusiontrack.py.")
    args = parser.parse_args()

    release = build_dashboard_release(
        run_config=args.run_config,
        pages_dir=args.pages_dir,
        run_id=args.run_id,
        export_package=args.export_package,
        python_executable=args.python_executable,
    )
    print(json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
