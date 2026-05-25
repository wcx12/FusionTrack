from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_SCHEMA_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple benchmark matrix configs as one FusionTrack evaluation suite."
    )
    parser.add_argument("--suite-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suite_path = args.suite_json.resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict):
        raise ValueError("Suite config must be a JSON object")
    matrices = suite.get("matrices", [])
    if not isinstance(matrices, list) or not matrices:
        raise ValueError("Suite config field 'matrices' must be a non-empty list")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_results = [
        _run_matrix(matrix, suite_dir=suite_path.parent, suite_output_dir=output_dir)
        for matrix in matrices
    ]
    aggregate_summary = output_dir / "aggregate_summary.csv"
    _write_aggregate_summary(matrix_results, aggregate_summary)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "suite_name": str(suite.get("suite_name") or suite_path.stem),
        "generated_at_utc": _utc_now(),
        "suite_json": str(suite_path),
        "suite_sha256": _file_sha256(suite_path),
        "output_dir": str(output_dir),
        "aggregate_summary_csv": str(aggregate_summary),
        "aggregate_summary_sha256": _file_sha256(aggregate_summary),
        "matrices": matrix_results,
    }
    manifest_path = output_dir / "suite_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _run_matrix(
    matrix: dict[str, Any],
    suite_dir: Path,
    suite_output_dir: Path,
) -> dict[str, Any]:
    if not isinstance(matrix, dict):
        raise ValueError("Each suite matrix entry must be a JSON object")
    name = _safe_name(_required(matrix, "name"))
    config_json = _resolve_path(_required(matrix, "config_json"), suite_dir)
    output_dir = _resolve_path(matrix.get("output_dir", name), suite_output_dir)
    command = [
        sys.executable,
        str(BENCHMARK_ROOT / "runners" / "run_benchmark_matrix.py"),
        "--config-json",
        str(config_json),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(
        command,
        cwd=BENCHMARK_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"Matrix {name!r} failed with exit code {result.returncode}: {' '.join(command)}\n{result.stderr}"
        )
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.csv"
    matrix_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "name": name,
        "config_json": str(config_json),
        "config_sha256": _file_sha256(config_json),
        "output_dir": str(output_dir),
        "manifest_json": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "summary_csv": str(summary_path),
        "summary_sha256": _file_sha256(summary_path),
        "num_runs": len(matrix_manifest.get("runs", [])),
        "split": matrix_manifest.get("split"),
        "key_fields": matrix_manifest.get("key_fields", []),
    }


def _write_aggregate_summary(matrix_results: list[dict[str, Any]], output_csv: Path) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = ["matrix"]
    for matrix in matrix_results:
        summary_path = Path(str(matrix["summary_csv"]))
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                output_row = {"matrix": str(matrix["name"]), **dict(row)}
                rows.append(output_row)
                for field in output_row:
                    if field not in fieldnames:
                        fieldnames.append(field)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _required(mapping: dict[str, Any], field: str) -> Any:
    value = mapping.get(field)
    if value in (None, ""):
        raise ValueError(f"Missing required suite matrix field '{field}'")
    return value


def _safe_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Matrix name cannot be empty")
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
