#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MTF_BA_ROOT = REPO_ROOT / "code" / "anomaly_detection" / "individual"
if str(MTF_BA_ROOT) not in sys.path:
    sys.path.insert(0, str(MTF_BA_ROOT))

from fusiontrack.config import FusionTrackPaths
from fusiontrack.export_package import build_analysis_export_package
from fusiontrack.pipeline import (
    build_experiment_report,
    build_final_results_report,
    run_registration_experiment,
    run_smoke_pipeline,
)
from fusiontrack.registration_adapter import build_registration_experiment_bundle


PATH_CONFIG_FIELDS = {
    "data_root",
    "work_root",
    "result_manifest",
    "registration_benchmark_summary",
    "registration_manifest",
    "registration_fused_jsonl",
    "fused_jsonl",
    "fused_pipeline_manifest",
    "final_results_root",
    "individual_label_file",
    "group_label_file",
    "suite_manifest",
    "holdout_manifest",
    "export_package",
}
LIST_PATH_CONFIG_FIELDS = {
    "score_search_root": {"score_search_root", "score_search_roots"},
    "protocol_manifest": {"protocol_manifest", "protocol_manifests"},
}


def _resolve_config_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base_dir / path).resolve()


def _load_run_config(config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("--run-config must point to a JSON object")
    base_value = raw.get("base_dir")
    base_dir = config_path.parent
    if base_value not in (None, ""):
        base_dir = _resolve_config_path(base_value, config_path.parent)
    config: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "base_dir":
            continue
        normalized_key = next(
            (target for target, aliases in LIST_PATH_CONFIG_FIELDS.items() if key in aliases),
            key,
        )
        if normalized_key in LIST_PATH_CONFIG_FIELDS:
            values = value if isinstance(value, list) else [value]
            config[normalized_key] = [_resolve_config_path(item, base_dir) for item in values]
        elif normalized_key in PATH_CONFIG_FIELDS and value not in (None, ""):
            config[normalized_key] = _resolve_config_path(value, base_dir)
        else:
            config[normalized_key] = value
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--run-config", type=Path, help="JSON file with default CLI options.")
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_defaults = _load_run_config(pre_args.run_config) if pre_args.run_config else {}

    parser = argparse.ArgumentParser(description="Run FusionTrack v1 offline pipeline.")
    parser.add_argument("--run-config", type=Path, default=pre_args.run_config, help="JSON file with default CLI options.")
    parser.add_argument("--data-root", type=Path, default=config_defaults.get("data_root", Path("data") / "VT-Tiny-MOT"))
    parser.add_argument("--work-root", type=Path, default=config_defaults.get("work_root", Path("runs") / "fusiontrack_v1"))
    parser.add_argument("--split", choices=["train", "test", "val"], default=config_defaults.get("split", "test"))
    parser.add_argument("--mode", choices=["smoke", "full"], default=config_defaults.get("mode", "smoke"))
    parser.add_argument("--top-sequences", type=int, default=config_defaults.get("top_sequences", 5))
    parser.add_argument("--force", action="store_true", default=bool(config_defaults.get("force", False)))
    parser.add_argument("--skip-extraction", action="store_true", default=bool(config_defaults.get("skip_extraction", False)))
    parser.add_argument("--device", default=config_defaults.get("device", "cpu"), help="Reserved for full training mode.")
    parser.add_argument("--result-manifest", type=Path, default=config_defaults.get("result_manifest"), help="Benchmark result manifest to render.")
    parser.add_argument("--result-method", default=config_defaults.get("result_method"), help="Method name inside the result manifest.")
    parser.add_argument(
        "--registration-benchmark-summary",
        type=Path,
        default=config_defaults.get("registration_benchmark_summary"),
        help="Path to non-learning registration baseline summary (run_registration_benchmark.py output).",
    )
    parser.add_argument(
        "--registration-manifest",
        type=Path,
        default=config_defaults.get("registration_manifest"),
        help="Registration experiment manifest (to add registration task into final dashboard).",
    )
    parser.add_argument(
        "--registration-fused-jsonl",
        type=Path,
        default=config_defaults.get("registration_fused_jsonl"),
        help="Registration fused trajectories JSONL for final dashboard merging.",
    )
    parser.add_argument(
        "--registration-result-method",
        default=config_defaults.get("registration_result_method"),
        help="Method to visualize from registration benchmark summary.",
    )
    parser.add_argument("--fused-jsonl", type=Path, default=config_defaults.get("fused_jsonl"), help="Fused trajectory JSONL used by the result manifest.")
    parser.add_argument("--fused-pipeline-manifest", type=Path, default=config_defaults.get("fused_pipeline_manifest"), help="Optional fused_track_pipeline_manifest_<split>.json to link into final dashboard provenance.")
    parser.add_argument("--final-results-root", type=Path, default=config_defaults.get("final_results_root"), help="Directory with final_*_summary files.")
    parser.add_argument("--individual-label-file", type=Path, default=config_defaults.get("individual_label_file"), help="Individual labels JSONL for final dashboard.")
    parser.add_argument("--group-label-file", type=Path, default=config_defaults.get("group_label_file"), help="Group labels JSONL for final dashboard.")
    parser.add_argument("--suite-manifest", type=Path, default=config_defaults.get("suite_manifest"), help="Optional run_suite.py suite_manifest.json to link into the final dashboard and export package.")
    parser.add_argument("--holdout-manifest", type=Path, default=config_defaults.get("holdout_manifest"), help="Optional holdout multiseed manifest.json to link into the final dashboard.")
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        action="append",
        default=None,
        help="Synthetic anomaly protocol manifest JSON. Can be repeated for individual/group protocols.",
    )
    parser.add_argument(
        "--score-search-root",
        type=Path,
        action="append",
        default=None,
        help="Search root for score files referenced by final summaries. Can be repeated.",
    )
    parser.add_argument("--top-k", type=int, default=config_defaults.get("top_k", 100), help="Top-K used for case and anomaly-type analysis.")
    parser.add_argument("--case-limit", type=int, default=config_defaults.get("case_limit", 12), help="Maximum TP/FP/FN cases per method.")
    parser.add_argument(
        "--export-package",
        type=Path,
        default=config_defaults.get("export_package"),
        help="Optional zip path for a portable dashboard/report export package.",
    )
    args = parser.parse_args(argv)
    if args.score_search_root is None:
        args.score_search_root = list(config_defaults.get("score_search_root", []))
    if args.protocol_manifest is None:
        args.protocol_manifest = list(config_defaults.get("protocol_manifest", []))
    return args


def main() -> None:
    args = parse_args()
    paths = FusionTrackPaths.defaults(data_root=args.data_root, work_root=args.work_root)
    if args.final_results_root:
        if args.individual_label_file is None or args.group_label_file is None:
            raise SystemExit("--individual-label-file and --group-label-file are required with --final-results-root")
        registration_manifest = args.registration_manifest
        registration_fused_jsonl = args.registration_fused_jsonl
        if args.registration_benchmark_summary is not None and registration_manifest is None:
            bundle = build_registration_experiment_bundle(
                summary_path=args.registration_benchmark_summary,
                work_root=paths.work_root,
            )
            registration_manifest = Path(bundle["manifest_path"])
            if registration_fused_jsonl is None:
                registration_fused_jsonl = Path(bundle["fused_jsonl"])
        default_root = args.final_results_root.parent
        score_roots = list(args.score_search_root) if args.score_search_root else [default_root]
        if registration_manifest is not None and paths.work_root is not None:
            work_root = paths.work_root
            normalized_work_roots = {p.resolve() for p in score_roots}
            if work_root.resolve() not in normalized_work_roots:
                score_roots.append(work_root)
            elif work_root.resolve() != work_root:
                work_root_resolved = work_root.resolve()
                if work_root_resolved not in normalized_work_roots:
                    score_roots.append(work_root_resolved)
        summary = build_final_results_report(
            paths=paths,
            final_results_root=args.final_results_root,
            individual_label_file=args.individual_label_file,
            group_label_file=args.group_label_file,
            score_search_roots=score_roots,
            registration_manifest=registration_manifest,
            registration_fused_jsonl=registration_fused_jsonl,
            fused_jsonl=args.fused_jsonl,
            fused_pipeline_manifest=args.fused_pipeline_manifest,
            suite_manifest=args.suite_manifest,
            holdout_manifest=args.holdout_manifest,
            protocol_manifests=args.protocol_manifest,
            top_sequences=args.top_sequences,
            top_k=args.top_k,
            case_limit=args.case_limit,
        )
    elif args.result_manifest:
        summary = build_experiment_report(
            paths=paths,
            result_manifest=args.result_manifest,
            split=args.split,
            result_method=args.result_method,
            fused_jsonl=args.fused_jsonl,
            top_sequences=args.top_sequences,
        )
    elif args.registration_benchmark_summary:
        summary = run_registration_experiment(
            paths=paths,
            benchmark_summary=args.registration_benchmark_summary,
            split=args.split,
            result_method=args.registration_result_method,
            top_sequences=args.top_sequences,
        )
    elif args.mode == "full":
        print("full mode is using the smoke baseline path in this v1 implementation.")
        summary = run_smoke_pipeline(
            paths=paths,
            split=args.split,
            top_sequences=args.top_sequences,
            force=args.force,
            skip_extraction=args.skip_extraction,
        )
    else:
        summary = run_smoke_pipeline(
            paths=paths,
            split=args.split,
            top_sequences=args.top_sequences,
            force=args.force,
            skip_extraction=args.skip_extraction,
        )
    if args.export_package:
        summary["export_package"] = build_analysis_export_package(summary, args.export_package)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
