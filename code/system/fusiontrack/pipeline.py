from __future__ import annotations

import json
import subprocess
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mtf_ba.group_interface import (
    GroupWindowConfig,
    aggregate_group_scores_by_sample,
    iter_group_windows,
    load_group_score_records_jsonl,
    write_group_score_records_jsonl,
    write_group_windows_jsonl,
)

from fusiontrack.config import FusionTrackPaths
from fusiontrack.experiment_adapter import load_experiment_result, write_scores_csv
from fusiontrack.final_dashboard import build_final_dashboard
from fusiontrack.final_results import load_final_results_dashboard
from fusiontrack.fusion import fuse_observations_csv
from fusiontrack.group_baseline import score_group_windows_jsonl
from fusiontrack.score_fusion import fuse_score_records
from fusiontrack.simple_detectors import score_fused_trajectories_simple
from fusiontrack.registration_adapter import build_registration_experiment_bundle
from fusiontrack.visualization import build_visual_report


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _as_posix(path: Path) -> str:
    return path.as_posix()


def build_extraction_command(paths: FusionTrackPaths, split: str) -> list[str]:
    return [
        "python",
        "code/anomaly_detection/individual/extract_vt_tiny_mot_trajectories.py",
        "--data-root",
        _as_posix(paths.data_root),
        "--split",
        split,
        "--output-dir",
        _as_posix(paths.trajectory_dir),
    ]


def run_command(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd or repo_root(), check=True)


def ensure_output_dirs(paths: FusionTrackPaths) -> None:
    for directory in (
        paths.trajectory_dir,
        paths.fusion_dir,
        paths.feature_dir,
        paths.model_dir,
        paths.score_dir,
        paths.group_dir,
        paths.final_dir,
        paths.heatmap_dir,
        paths.report_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _build_manifest(
    mode: str,
    paths: FusionTrackPaths,
    split: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": uuid.uuid4().hex[:16],
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split,
        "data_root": str(paths.data_root),
        "work_root": str(paths.work_root),
        "trajectory_dir": str(paths.trajectory_dir),
        "fusion_dir": str(paths.fusion_dir),
        "feature_dir": str(paths.feature_dir),
        "model_dir": str(paths.model_dir),
        "score_dir": str(paths.score_dir),
        "group_dir": str(paths.group_dir),
        "final_dir": str(paths.final_dir),
        "heatmap_dir": str(paths.heatmap_dir),
        "report_dir": str(paths.report_dir),
        "config": dict(config or {}),
    }


def _write_manifest(paths: FusionTrackPaths, mode: str, split: str, payload: dict[str, Any]) -> str:
    manifest = _build_manifest(mode=mode, paths=paths, split=split)
    manifest.update(payload)
    manifest_path = paths.work_root / f"pipeline_manifest_{mode}_{split}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path)


def _sync_remote_report(source_dir: Path, target_dir: Path) -> None:
    """Mirror generated dashboard outputs to the expected remote-result preview directory."""
    if not source_dir.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in target_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def extract_vt_tiny_mot(paths: FusionTrackPaths, split: str, force: bool = False) -> Path:
    output_path = paths.observations_csv(split)
    if output_path.exists() and not force:
        return output_path
    run_command(build_extraction_command(paths, split))
    return output_path


def export_group_windows(
    observations_csv: Path,
    output_jsonl: Path,
    window_size: int = 16,
    stride: int = 8,
) -> dict[str, Any]:
    config = GroupWindowConfig(
        sample_mode="window",
        window_size=window_size,
        stride=stride,
        min_visible_frames=2,
        require_both_modalities=False,
    )
    windows = iter_group_windows(observations_csv, config=config)
    return write_group_windows_jsonl(output_jsonl, windows)


def aggregate_group_scores(window_scores_jsonl: Path, output_jsonl: Path) -> dict[str, Any]:
    records = load_group_score_records_jsonl(window_scores_jsonl)
    aggregated = aggregate_group_scores_by_sample(records, method="max")
    count = write_group_score_records_jsonl(output_jsonl, aggregated)
    return {
        "input_jsonl": str(window_scores_jsonl),
        "output_jsonl": str(output_jsonl),
        "num_group_scores": count,
    }


def run_smoke_pipeline(
    paths: FusionTrackPaths,
    split: str = "test",
    top_sequences: int = 5,
    force: bool = False,
    skip_extraction: bool = False,
) -> dict[str, Any]:
    ensure_output_dirs(paths)

    observations_csv = paths.observations_csv(split)
    if not skip_extraction:
        observations_csv = extract_vt_tiny_mot(paths, split=split, force=force)
    if not observations_csv.exists():
        raise FileNotFoundError(f"Missing observations CSV: {observations_csv}")

    fusion_summary = fuse_observations_csv(
        observations_csv,
        paths.fused_jsonl(split),
        paths.fused_states_csv(split),
    )

    individual_jsonl = paths.score_dir / f"individual_simple_scores_{split}.jsonl"
    individual_csv = paths.score_dir / f"individual_simple_scores_{split}.csv"
    individual_summary = score_fused_trajectories_simple(
        paths.fused_jsonl(split),
        individual_jsonl,
        individual_csv,
    )

    group_windows_jsonl = paths.group_dir / f"group_windows_{split}.jsonl"
    group_window_summary = export_group_windows(observations_csv, group_windows_jsonl)
    group_window_scores_jsonl = paths.group_dir / f"group_window_scores_{split}.jsonl"
    group_window_score_summary = score_group_windows_jsonl(group_windows_jsonl, group_window_scores_jsonl)
    group_scores_jsonl = paths.group_dir / f"group_scores_{split}.jsonl"
    group_summary = aggregate_group_scores(group_window_scores_jsonl, group_scores_jsonl)

    final_jsonl = paths.final_dir / f"fused_scores_{split}.jsonl"
    final_csv = paths.final_dir / f"fused_scores_{split}.csv"
    final_summary = fuse_score_records(
        individual_jsonl,
        group_scores_jsonl,
        final_jsonl,
        final_csv,
        alpha=0.65,
    )

    report_summary = build_visual_report(
        fused_jsonl=paths.fused_jsonl(split),
        final_scores_csv=final_csv,
        data_root=paths.data_root,
        output_dir=paths.report_dir,
        top_sequences=top_sequences,
    )

    summary = {
        "mode": "smoke",
        "split": split,
        "data_root": str(paths.data_root),
        "work_root": str(paths.work_root),
        "observations_csv": str(observations_csv),
        "fusion": fusion_summary,
        "individual": individual_summary,
        "group_windows": group_window_summary,
        "group_window_scores": group_window_score_summary,
        "group_scores": group_summary,
        "final_scores": final_summary,
        "report": report_summary,
    }
    manifest_payload = {
        "manifest": {
            "smoke_inputs": {
                "skip_extraction": skip_extraction,
                "force_extraction": force,
                "top_sequences": top_sequences,
            },
            "artifacts": {
                "summary_path": str(paths.work_root / f"pipeline_summary_{split}.json"),
                "manifest_mode": "smoke",
            },
        }
    }
    summary_path = paths.work_root / f"pipeline_summary_{split}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["manifest_path"] = _write_manifest(
        paths=paths,
        mode="smoke",
        split=split,
        payload={"run_inputs": manifest_payload["manifest"], "summary_path": str(summary_path)},
    )
    return summary


def run_registration_experiment(
    paths: FusionTrackPaths,
    benchmark_summary: str | Path,
    split: str = "test",
    result_method: str | None = None,
    top_sequences: int = 5,
) -> dict[str, Any]:
    ensure_output_dirs(paths)
    bundle = build_registration_experiment_bundle(benchmark_summary, paths.work_root)
    summary = build_experiment_report(
        paths=paths,
        result_manifest=Path(bundle["manifest_path"]),
        split=split,
        result_method=result_method,
        fused_jsonl=Path(bundle["fused_jsonl"]),
        top_sequences=top_sequences,
    )
    summary["registration_bundle"] = {
        "manifest_path": bundle["manifest_path"],
        "num_methods": bundle["num_methods"],
        "num_scores": bundle["num_scores"],
        "score_files": bundle["score_files"],
    }
    return summary


def build_experiment_report(
    paths: FusionTrackPaths,
    result_manifest: str | Path,
    split: str = "test",
    result_method: str | None = None,
    fused_jsonl: str | Path | None = None,
    top_sequences: int = 5,
) -> dict[str, Any]:
    ensure_output_dirs(paths)
    result = load_experiment_result(result_manifest, method_name=result_method)
    score_csv = paths.final_dir / f"experiment_scores_{_safe_filename(result.method_name)}.csv"
    score_summary = write_scores_csv(result, score_csv)

    source_fused_jsonl = Path(fused_jsonl) if fused_jsonl is not None else paths.fused_jsonl(split)
    if not source_fused_jsonl.exists():
        raise FileNotFoundError(f"Missing fused trajectories JSONL: {source_fused_jsonl}")

    experiment_context = result.to_report_context()
    report_summary = build_visual_report(
        fused_jsonl=source_fused_jsonl,
        final_scores_csv=score_csv,
        data_root=paths.data_root,
        output_dir=paths.report_dir,
        top_sequences=top_sequences,
        experiment_context=experiment_context,
    )
    summary = {
        "mode": "experiment_report",
        "split": split,
        "data_root": str(paths.data_root),
        "work_root": str(paths.work_root),
        "result_manifest": str(result_manifest),
        "fused_jsonl": str(source_fused_jsonl),
        "score_export": score_summary,
        "experiment": report_summary.get("experiment", {}),
        "report": report_summary,
    }
    summary_path = paths.work_root / f"pipeline_summary_{split}_experiment.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["manifest_path"] = _write_manifest(
        paths=paths,
        mode="experiment_report",
        split=split,
        payload={
            "result_manifest": str(result_manifest),
            "selected_method": result.method_name,
            "result_manifest_context": {
                "method": result.method_name,
                "task": result.task,
                "split": result.split,
                "seed": result.seed,
            },
            "top_sequences": top_sequences,
            "summary_path": str(summary_path),
        },
    )
    return summary


def build_final_results_report(
    paths: FusionTrackPaths,
    final_results_root: str | Path,
    individual_label_file: str | Path,
    group_label_file: str | Path,
    score_search_roots: list[str | Path],
    registration_manifest: str | Path | None = None,
    registration_fused_jsonl: str | Path | None = None,
    fused_jsonl: str | Path | None = None,
    top_sequences: int = 5,
    top_k: int = 100,
    case_limit: int = 12,
    sync_remote_report: bool = True,
) -> dict[str, Any]:
    ensure_output_dirs(paths)
    manifest_registration = Path(registration_manifest) if registration_manifest is not None else None
    manifest_fused = Path(registration_fused_jsonl) if registration_fused_jsonl is not None else None
    dashboard = load_final_results_dashboard(
        final_results_root=final_results_root,
        individual_label_file=individual_label_file,
        group_label_file=group_label_file,
        score_search_roots=score_search_roots,
        registration_manifest=registration_manifest,
        top_k=top_k,
        case_limit=case_limit,
    )
    output_dir = paths.work_root / "final_dashboard"
    dashboard_fused_jsonl = _merge_fused_for_dashboard(
        paths=paths,
        fused_jsonl=fused_jsonl,
        registration_fused_jsonl=manifest_fused,
    )
    dashboard_summary = build_final_dashboard(
        dashboard=dashboard,
        output_dir=output_dir,
        fused_jsonl=dashboard_fused_jsonl,
        data_root=paths.data_root,
        top_sequences=top_sequences,
    )
    if sync_remote_report:
        _sync_remote_report(
            source_dir=output_dir,
            target_dir=Path("server_artifacts") / "remote_result" / "report",
        )
    summary = {
        "mode": "final_results_dashboard",
        "data_root": str(paths.data_root),
        "work_root": str(paths.work_root),
        "final_results_root": str(final_results_root),
        "individual_label_file": str(individual_label_file),
        "group_label_file": str(group_label_file),
        "score_search_roots": [str(path) for path in score_search_roots],
        "fused_jsonl": None if dashboard_fused_jsonl is None else str(dashboard_fused_jsonl),
        "registration_manifest": str(registration_manifest) if registration_manifest is not None else None,
        "registration_fused_jsonl": str(registration_fused_jsonl) if registration_fused_jsonl is not None else None,
        "dashboard": dashboard_summary,
    }
    summary_path = paths.work_root / "pipeline_summary_final_dashboard.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    summary["manifest_path"] = _write_manifest(
        paths=paths,
        mode="final_results_dashboard",
        split="all",
        payload={
            "final_results_root": str(final_results_root),
            "individual_label_file": str(individual_label_file),
            "group_label_file": str(group_label_file),
            "score_search_roots": [str(path) for path in score_search_roots],
            "registration_manifest": str(registration_manifest) if registration_manifest is not None else None,
            "registration_fused_jsonl": str(manifest_fused) if manifest_fused is not None else None,
            "top_sequences": top_sequences,
            "top_k": top_k,
            "case_limit": case_limit,
            "sync_remote_report": sync_remote_report,
            "summary_path": str(summary_path),
            "dashboard_summary": dashboard_summary,
        },
    )
    return summary


def _merge_fused_for_dashboard(
    paths: FusionTrackPaths,
    fused_jsonl: str | Path | None,
    registration_fused_jsonl: Path | None,
) -> str | Path | None:
    if registration_fused_jsonl is None:
        return fused_jsonl
    if not registration_fused_jsonl.exists():
        raise FileNotFoundError(
            f"Registration fused trajectory JSONL missing: {registration_fused_jsonl}"
        )

    if fused_jsonl is None:
        return registration_fused_jsonl

    base_fused_jsonl = Path(fused_jsonl)
    if not base_fused_jsonl.exists():
        raise FileNotFoundError(f"Missing fused trajectory JSONL: {fused_jsonl}")

    merged_rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    output_path = paths.final_dir / "merged_final_dashboard_fused.jsonl"

    for source in (base_fused_jsonl, registration_fused_jsonl):
        with source.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    continue
                sample_id = str(row.get("sample_id", ""))
                sequence = str(row.get("sequence", ""))
                track_id = str(row.get("track_id", ""))
                key = (sample_id, sequence, track_id)
                if key in seen:
                    continue
                seen.add(key)
                merged_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path
