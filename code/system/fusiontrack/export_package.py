from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_FORMAT = "fusiontrack_analysis_export_v1"
EXPORTABLE_SUFFIXES = {
    ".csv",
    ".html",
    ".jpg",
    ".jpeg",
    ".json",
    ".jsonl",
    ".md",
    ".png",
}


def build_analysis_export_package(summary: dict[str, Any], output_zip: str | Path) -> dict[str, Any]:
    """Create a portable zip package for dashboard delivery and offline review."""
    output_path = Path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roots = _package_roots(summary)
    files: list[dict[str, Any]] = []
    written: set[str] = set()

    sanitized_summary = _sanitize_value(summary, roots)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_json_bytes(
            archive,
            "summary/pipeline_summary.json",
            sanitized_summary,
            files,
            written,
            source="generated:sanitized_summary",
        )
        summary_path = _path_from_summary(summary, "summary_path")
        manifest_path = _path_from_summary(summary, "manifest_path")
        if manifest_path is not None and manifest_path.exists():
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            _write_json_bytes(
                archive,
                "summary/pipeline_manifest.json",
                _sanitize_value(manifest_payload, roots),
                files,
                written,
                source=_sanitize_path(manifest_path, roots),
            )

        report_dir = _report_dir(summary)
        if report_dir is not None and report_dir.exists():
            for path in sorted(item for item in report_dir.rglob("*") if item.is_file()):
                arcname = "report/" + path.relative_to(report_dir).as_posix()
                _write_file(archive, path, arcname, files, written, roots)

        for path in _iter_referenced_files(summary):
            if report_dir is not None:
                try:
                    path.relative_to(report_dir)
                    continue
                except ValueError:
                    pass
            if manifest_path is not None and path == manifest_path:
                continue
            if summary_path is not None and path == summary_path:
                continue
            arcname = "artifacts/" + _artifact_arcname(path, roots)
            _write_file(archive, path, arcname, files, written, roots)

        export_manifest = {
            "package_format": PACKAGE_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_files": len(files) + 1,
            "summary": sanitized_summary,
            "files": files,
        }
        _write_json_bytes(
            archive,
            "export_manifest.json",
            export_manifest,
            files,
            written,
            source="generated:export_manifest",
            register=False,
        )

    return {
        "package_path": str(output_path),
        "package_format": PACKAGE_FORMAT,
        "num_files": len(written),
        "files": sorted(written),
    }


def _write_file(
    archive: zipfile.ZipFile,
    path: Path,
    arcname: str,
    files: list[dict[str, Any]],
    written: set[str],
    roots: dict[str, Path],
) -> None:
    if arcname in written:
        return
    archive.write(path, arcname)
    written.add(arcname)
    files.append(
        {
            "arcname": arcname,
            "source": _sanitize_path(path, roots),
            "size_bytes": path.stat().st_size,
        }
    )


def _write_json_bytes(
    archive: zipfile.ZipFile,
    arcname: str,
    payload: dict[str, Any],
    files: list[dict[str, Any]],
    written: set[str],
    source: str,
    register: bool = True,
) -> None:
    if arcname in written:
        return
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    info = zipfile.ZipInfo(arcname)
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data)
    written.add(arcname)
    if register:
        files.append({"arcname": arcname, "source": source, "size_bytes": len(data)})


def _package_roots(summary: dict[str, Any]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for key in ("work_root", "data_root", "final_results_root"):
        value = summary.get(key)
        if value:
            roots[key] = Path(str(value)).resolve()
    dashboard = summary.get("dashboard")
    if isinstance(dashboard, dict):
        assets_dir = dashboard.get("assets_dir")
        report_html = dashboard.get("report_html")
        if report_html:
            roots.setdefault("report_root", Path(str(report_html)).resolve().parent)
        elif assets_dir:
            roots.setdefault("report_root", Path(str(assets_dir)).resolve().parent)
    return roots


def _report_dir(summary: dict[str, Any]) -> Path | None:
    dashboard = summary.get("dashboard")
    if not isinstance(dashboard, dict):
        return None
    report_html = dashboard.get("report_html")
    if report_html:
        return Path(str(report_html)).resolve().parent
    assets_dir = dashboard.get("assets_dir")
    if assets_dir:
        return Path(str(assets_dir)).resolve().parent
    return None


def _path_from_summary(summary: dict[str, Any], key: str) -> Path | None:
    value = summary.get(key)
    if not value:
        return None
    return Path(str(value)).resolve()


def _iter_referenced_files(value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, dict):
        for item in value.values():
            paths.extend(_iter_referenced_files(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_iter_referenced_files(item))
    elif isinstance(value, str) and _looks_like_exportable_path(value):
        path = Path(value)
        if path.exists() and path.is_file():
            paths.append(path.resolve())
    return _dedupe_paths(paths)


def _looks_like_exportable_path(value: str) -> bool:
    if not value or value.startswith("${"):
        return False
    path = Path(value)
    return path.suffix.lower() in EXPORTABLE_SUFFIXES and ("/" in value or "\\" in value)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _artifact_arcname(path: Path, roots: dict[str, Path]) -> str:
    sanitized = _sanitize_path(path, roots)
    if sanitized.startswith("${") and "}/" in sanitized:
        sanitized = sanitized[2:].replace("}/", "/", 1)
    else:
        sanitized = Path(sanitized).name
    return sanitized.replace("\\", "/").lstrip("/")


def _sanitize_value(value: Any, roots: dict[str, Path]) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_value(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item, roots) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value, roots)
    return value


def _sanitize_string(value: str, roots: dict[str, Path]) -> str:
    if not _looks_like_path_string(value):
        return value
    path = Path(value)
    if not path.is_absolute():
        return value.replace("\\", "/")
    return _sanitize_path(path, roots)


def _looks_like_path_string(value: str) -> bool:
    if not value:
        return False
    if value.startswith("${"):
        return False
    return Path(value).is_absolute() or "/" in value or "\\" in value


def _sanitize_path(path: Path, roots: dict[str, Path]) -> str:
    resolved = path.resolve()
    for name, root in roots.items():
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        return f"${{{name}}}/{rel.as_posix()}" if rel.as_posix() != "." else f"${{{name}}}"
    if resolved.is_absolute():
        return f"${{external}}/{resolved.name}"
    return resolved.as_posix()
