#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FusionTrack v1 offline pipeline.")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "VT-Tiny-MOT")
    parser.add_argument("--work-root", type=Path, default=Path("runs") / "fusiontrack_v1")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--top-sequences", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--device", default="cpu", help="Reserved for full training mode.")
    parser.add_argument("--result-manifest", type=Path, help="Benchmark result manifest to render.")
    parser.add_argument("--result-method", help="Method name inside the result manifest.")
    parser.add_argument(
        "--registration-benchmark-summary",
        type=Path,
        help="Path to non-learning registration baseline summary (run_registration_benchmark.py output).",
    )
    parser.add_argument(
        "--registration-manifest",
        type=Path,
        help="Registration experiment manifest (to add registration task into final dashboard).",
    )
    parser.add_argument(
        "--registration-fused-jsonl",
        type=Path,
        help="Registration fused trajectories JSONL for final dashboard merging.",
    )
    parser.add_argument(
        "--registration-result-method",
        default=None,
        help="Method to visualize from registration benchmark summary.",
    )
    parser.add_argument("--fused-jsonl", type=Path, help="Fused trajectory JSONL used by the result manifest.")
    parser.add_argument("--final-results-root", type=Path, help="Directory with final_*_summary files.")
    parser.add_argument("--individual-label-file", type=Path, help="Individual labels JSONL for final dashboard.")
    parser.add_argument("--group-label-file", type=Path, help="Group labels JSONL for final dashboard.")
    parser.add_argument(
        "--score-search-root",
        type=Path,
        action="append",
        default=[],
        help="Search root for score files referenced by final summaries. Can be repeated.",
    )
    parser.add_argument("--top-k", type=int, default=100, help="Top-K used for case and anomaly-type analysis.")
    parser.add_argument("--case-limit", type=int, default=12, help="Maximum TP/FP/FN cases per method.")
    parser.add_argument(
        "--export-package",
        type=Path,
        help="Optional zip path for a portable dashboard/report export package.",
    )
    return parser.parse_args()


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
