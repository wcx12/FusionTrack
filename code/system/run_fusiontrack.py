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
from fusiontrack.pipeline import build_experiment_report, build_final_results_report, run_smoke_pipeline


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = FusionTrackPaths.defaults(data_root=args.data_root, work_root=args.work_root)
    if args.final_results_root:
        if args.individual_label_file is None or args.group_label_file is None:
            raise SystemExit("--individual-label-file and --group-label-file are required with --final-results-root")
        score_roots = args.score_search_root or [args.final_results_root.parent]
        summary = build_final_results_report(
            paths=paths,
            final_results_root=args.final_results_root,
            individual_label_file=args.individual_label_file,
            group_label_file=args.group_label_file,
            score_search_roots=score_roots,
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
