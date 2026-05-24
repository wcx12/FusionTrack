from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))


NUMERIC_METRICS = (
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "precision_at_k",
    "recall_at_k",
    "threshold",
    "num_label_rows",
    "num_score_rows",
    "num_unique_label_keys",
    "num_unique_score_keys",
    "num_duplicate_label_keys",
    "num_duplicate_score_keys",
    "num_missing_score_keys",
    "num_extra_score_keys",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FusionTrack holdout experiments for multiple injection/model seeds "
            "and aggregate the resulting summary.csv files."
        )
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--train-source-split", default="train", choices=("train", "test", "val"))
    parser.add_argument("--eval-source-split", default="test", choices=("train", "test", "val"))
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--levels", default="individual,group")
    parser.add_argument("--individual-anomaly-fraction", type=float, default=0.1)
    parser.add_argument("--group-anomaly-fraction", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--smoke-max-train", type=int, default=0)
    parser.add_argument("--smoke-max-eval", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = _seed_values(args.seeds)
    levels = _level_values(args.levels)
    split_name = str(args.split_name or args.eval_source_split)
    output_root = args.output_root.resolve()
    work_root = args.work_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        protocol_root = seed_root / "protocol"
        result_root = seed_root / "results"
        _prepare_seed_protocol(
            args=args,
            seed=seed,
            protocol_root=protocol_root,
            work_root=work_root / f"seed_{seed}",
            split_name=split_name,
        )
        for level in levels:
            matrix_path = protocol_root / f"{level}_{split_name}_matrix.json"
            result_dir = result_root / level
            _run_matrix(matrix_path=matrix_path, output_dir=result_dir)
            all_rows.extend(
                _summary_rows(
                    result_dir / "summary.csv",
                    seed=seed,
                    level=level,
                    protocol_root=protocol_root,
                    result_dir=result_dir,
                )
            )

    all_runs_csv = output_root / "all_runs.csv"
    aggregate_csv = output_root / "aggregate.csv"
    best_json = output_root / "best_by_metric.json"
    _write_csv(all_runs_csv, all_rows)
    aggregate_rows = aggregate_summary_rows(all_rows)
    _write_csv(aggregate_csv, aggregate_rows)
    best_json.write_text(
        json.dumps(best_by_metric(aggregate_rows), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest = build_holdout_manifest(
        args=args,
        seeds=seeds,
        levels=levels,
        split_name=split_name,
        output_root=output_root,
        work_root=work_root,
        all_runs_csv=all_runs_csv,
        aggregate_csv=aggregate_csv,
        best_json=best_json,
    )
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def build_holdout_manifest(
    args: argparse.Namespace,
    seeds: Sequence[int],
    levels: Sequence[str],
    split_name: str,
    output_root: Path,
    work_root: Path,
    all_runs_csv: Path,
    aggregate_csv: Path,
    best_json: Path,
) -> dict[str, Any]:
    return {
        "manifest_schema_version": 2,
        "generated_at_utc": _utc_now(),
        "data_root": str(Path(args.data_root).resolve()),
        "output_root": str(output_root),
        "work_root": str(work_root),
        "train_source_split": str(args.train_source_split),
        "eval_source_split": str(args.eval_source_split),
        "split_name": split_name,
        "seeds": [int(seed) for seed in seeds],
        "levels": [str(level) for level in levels],
        "all_runs_csv": str(all_runs_csv),
        "aggregate_csv": str(aggregate_csv),
        "best_by_metric_json": str(best_json),
        "protocol": {
            "train_source_split": str(args.train_source_split),
            "eval_source_split": str(args.eval_source_split),
            "split_name": split_name,
            "individual_anomaly_fraction": float(args.individual_anomaly_fraction),
            "group_anomaly_fraction": float(args.group_anomaly_fraction),
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "smoke_max_train": int(args.smoke_max_train),
            "smoke_max_eval": int(args.smoke_max_eval),
        },
        "artifacts": {
            "all_runs_csv": _artifact_manifest(all_runs_csv),
            "aggregate_csv": _artifact_manifest(aggregate_csv),
            "best_by_metric_json": _artifact_manifest(best_json),
        },
        "git": _git_metadata(),
        "environment": _environment_metadata(),
    }


def _prepare_seed_protocol(
    args: argparse.Namespace,
    seed: int,
    protocol_root: Path,
    work_root: Path,
    split_name: str,
) -> None:
    command = [
        sys.executable,
        str(BENCHMARK_ROOT / "runners" / "prepare_vt_tiny_mot_holdout_protocol.py"),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(protocol_root),
        "--work-root",
        str(work_root),
        "--train-source-split",
        str(args.train_source_split),
        "--eval-source-split",
        str(args.eval_source_split),
        "--split-name",
        split_name,
        "--seed",
        str(seed),
        "--individual-anomaly-fraction",
        str(args.individual_anomaly_fraction),
        "--group-anomaly-fraction",
        str(args.group_anomaly_fraction),
        "--window-size",
        str(args.window_size),
        "--stride",
        str(args.stride),
    ]
    if args.smoke_max_train > 0:
        command.extend(["--smoke-max-train", str(args.smoke_max_train)])
    if args.smoke_max_eval > 0:
        command.extend(["--smoke-max-eval", str(args.smoke_max_eval)])
    _run_command(command)


def _run_matrix(matrix_path: Path, output_dir: Path) -> None:
    _run_command(
        [
            sys.executable,
            str(BENCHMARK_ROOT / "runners" / "run_benchmark_matrix.py"),
            "--config-json",
            str(matrix_path),
            "--output-dir",
            str(output_dir),
        ]
    )


def _run_command(command: list[str]) -> None:
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
            f"Command failed with exit code {result.returncode}: {' '.join(command)}\n"
            f"{result.stderr}"
        )


def _summary_rows(
    summary_csv: Path,
    seed: int,
    level: str,
    protocol_root: Path,
    result_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["seed"] = int(seed)
            row["level"] = level
            row["protocol_root"] = str(protocol_root)
            row["result_dir"] = str(result_dir)
            rows.append(row)
    return rows


def aggregate_summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("level", "")),
            str(row.get("method", "")),
            str(row.get("task", "")),
        )
        groups.setdefault(key, []).append(row)

    aggregate: list[dict[str, Any]] = []
    for (level, method, task), group_rows in sorted(groups.items()):
        output: dict[str, Any] = {
            "level": level,
            "method": method,
            "task": task,
            "num_runs": len(group_rows),
            "seeds": ",".join(str(row.get("seed", "")) for row in group_rows),
        }
        for metric in NUMERIC_METRICS:
            values = [_float_or_none(row.get(metric)) for row in group_rows]
            clean_values = [value for value in values if value is not None]
            if not clean_values:
                continue
            output[f"{metric}_mean"] = statistics.fmean(clean_values)
            output[f"{metric}_std"] = (
                statistics.stdev(clean_values) if len(clean_values) > 1 else 0.0
            )
        aggregate.append(output)
    return aggregate


def best_by_metric(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for metric in ("auprc", "auroc", "f1", "precision_at_k", "recall_at_k"):
        field = f"{metric}_mean"
        candidates = [row for row in rows if _float_or_none(row.get(field)) is not None]
        if not candidates:
            continue
        row = max(candidates, key=lambda item: float(item[field]))
        best[metric] = {
            "level": row["level"],
            "method": row["method"],
            "task": row["task"],
            "num_runs": row["num_runs"],
            f"{metric}_mean": float(row[field]),
            f"{metric}_std": float(row.get(f"{metric}_std", 0.0)),
        }
    return best


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    priority = [
        "level",
        "method",
        "task",
        "num_runs",
        "seeds",
        "source",
        "split",
        "seed",
    ]
    seen: set[str] = set()
    fieldnames: list[str] = []
    for field in priority:
        if any(field in row for row in rows):
            seen.add(field)
            fieldnames.append(field)
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    return fieldnames


def _seed_values(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        seeds = [int(seed) for seed in value]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _level_values(value: str) -> list[str]:
    levels = [part.strip() for part in value.split(",") if part.strip()]
    allowed = {"individual", "group"}
    unknown = [level for level in levels if level not in allowed]
    if unknown:
        raise ValueError(f"Unknown levels: {unknown}. Expected one of {sorted(allowed)}")
    if not levels:
        raise ValueError("At least one level is required")
    return levels


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _artifact_manifest(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
    }


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


def _git_metadata() -> dict[str, Any]:
    status = _run_git(["status", "--short"])
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
    }


def _run_git(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _environment_metadata() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
