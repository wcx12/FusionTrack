from __future__ import annotations

import json
import subprocess
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
    summary_path = paths.work_root / f"pipeline_summary_{split}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
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
    return summary


def build_final_results_report(
    paths: FusionTrackPaths,
    final_results_root: str | Path,
    individual_label_file: str | Path,
    group_label_file: str | Path,
    score_search_roots: list[str | Path],
    fused_jsonl: str | Path | None = None,
    top_sequences: int = 5,
    top_k: int = 100,
    case_limit: int = 12,
) -> dict[str, Any]:
    ensure_output_dirs(paths)
    dashboard = load_final_results_dashboard(
        final_results_root=final_results_root,
        individual_label_file=individual_label_file,
        group_label_file=group_label_file,
        score_search_roots=score_search_roots,
        top_k=top_k,
        case_limit=case_limit,
    )
    output_dir = paths.work_root / "final_dashboard"
    dashboard_summary = build_final_dashboard(
        dashboard=dashboard,
        output_dir=output_dir,
        fused_jsonl=fused_jsonl,
        data_root=paths.data_root,
        top_sequences=top_sequences,
    )
    summary = {
        "mode": "final_results_dashboard",
        "data_root": str(paths.data_root),
        "work_root": str(paths.work_root),
        "final_results_root": str(final_results_root),
        "individual_label_file": str(individual_label_file),
        "group_label_file": str(group_label_file),
        "score_search_roots": [str(path) for path in score_search_roots],
        "fused_jsonl": None if fused_jsonl is None else str(fused_jsonl),
        "dashboard": dashboard_summary,
    }
    summary_path = paths.work_root / "pipeline_summary_final_dashboard.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
