"""Utility to run non-learning registration baselines across multiple conditions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def build_cases() -> List[Dict[str, object]]:
    return [
        {
            "name": "base_crop",
            "noise_type": "crop",
            "num_points": 256,
            "partial": [0.7, 0.7],
        },
        {
            "name": "sparse_crop",
            "noise_type": "crop",
            "num_points": 256,
            "partial": [0.5, 0.5],
        },
        {
            "name": "dense_crop",
            "noise_type": "crop",
            "num_points": 256,
            "partial": [0.9, 0.9],
        },
        {
            "name": "points_512",
            "noise_type": "crop",
            "num_points": 512,
            "partial": [0.7, 0.7],
        },
        {
            "name": "jitter",
            "noise_type": "jitter",
            "num_points": 256,
            "partial": [0.7, 0.7],
        },
        {
            "name": "clean",
            "noise_type": "clean",
            "num_points": 256,
            "partial": [0.7, 0.7],
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run non-learning registration baseline benchmark across presets."
    )
    parser.add_argument(
        "--dataset_path",
        default="datasets/modelnet40_ply_hdf5_2048",
        help="Dataset path, relative to repository root.",
    )
    parser.add_argument(
        "--output_root",
        default="runs/mps_gaf_nonlearn_suite",
        help="Root directory for suite output.",
    )
    parser.add_argument(
        "--methods",
        default="icp_point_to_point,icp_point_to_plane,icp_trimmed,ransac_icp,identity",
        help="Comma-separated non-learning methods to benchmark.",
    )
    parser.add_argument("--rot_mag", type=float, default=30.0)
    parser.add_argument("--trans_mag", type=float, default=0.3)
    parser.add_argument("--num_sources_per_ref", type=int, default=2)
    parser.add_argument("--groups_per_batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=2)
    parser.add_argument("--icp_iterations", type=int, default=20)
    parser.add_argument("--icp_trim_fraction", type=float, default=0.7)
    parser.add_argument("--success_rotation_deg", type=float, default=5.0)
    parser.add_argument("--success_translation", type=float, default=0.2)
    parser.add_argument(
        "--icp_point_max_angle_deg",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--icp_point_max_translation",
        type=float,
        default=0.2,
    )
    parser.add_argument("--fpfh_voxel_size", type=float, default=0.05)
    parser.add_argument("--fpfh_normal_radius", type=float, default=0.1)
    parser.add_argument("--fpfh_feature_radius", type=float, default=0.25)
    parser.add_argument("--fpfh_normal_max_nn", type=int, default=30)
    parser.add_argument("--fpfh_feature_max_nn", type=int, default=100)
    parser.add_argument("--fpfh_max_correspondence_distance", type=float, default=0.075)
    parser.add_argument("--fpfh_ransac_n", type=int, default=4)
    parser.add_argument("--fpfh_ransac_max_iterations", type=int, default=100000)
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def validate_relative_paths(args: argparse.Namespace) -> None:
    for key in ("dataset_path", "output_root"):
        if Path(getattr(args, key)).is_absolute():
            raise ValueError(
                f"{key} is configured as an absolute path. Use a relative path per benchmark policy."
            )


def _build_case_args(args: argparse.Namespace, case: Dict[str, object]) -> List[str]:
    partial = case["partial"]
    assert isinstance(partial, Sequence)
    assert isinstance(partial[0], (int, float))
    assert isinstance(partial[1], (int, float))
    return [
        "--dataset_path",
        str(args.dataset_path),
        "--output_dir",
        f"runs/mps_gaf_nonlearn_suite/{case['name']}",
        "--methods",
        args.methods,
        "--noise_type",
        str(case["noise_type"]),
        "--num_points",
        str(case["num_points"]),
        "--partial",
        str(partial[0]),
        str(partial[1]),
        "--rot_mag",
        str(args.rot_mag),
        "--trans_mag",
        str(args.trans_mag),
        "--num_sources_per_ref",
        str(args.num_sources_per_ref),
        "--groups_per_batch",
        str(args.groups_per_batch),
        "--num_workers",
        str(args.num_workers),
        "--max_eval_batches",
        str(args.max_eval_batches),
        "--icp_iterations",
        str(args.icp_iterations),
        "--icp_trim_fraction",
        str(args.icp_trim_fraction),
        "--success_rotation_deg",
        str(args.success_rotation_deg),
        "--success_translation",
        str(args.success_translation),
        "--icp_point_max_angle_deg",
        str(args.icp_point_max_angle_deg),
        "--icp_point_max_translation",
        str(args.icp_point_max_translation),
        "--fpfh_voxel_size",
        str(args.fpfh_voxel_size),
        "--fpfh_normal_radius",
        str(args.fpfh_normal_radius),
        "--fpfh_feature_radius",
        str(args.fpfh_feature_radius),
        "--fpfh_normal_max_nn",
        str(args.fpfh_normal_max_nn),
        "--fpfh_feature_max_nn",
        str(args.fpfh_feature_max_nn),
        "--fpfh_max_correspondence_distance",
        str(args.fpfh_max_correspondence_distance),
        "--fpfh_ransac_n",
        str(args.fpfh_ransac_n),
        "--fpfh_ransac_max_iterations",
        str(args.fpfh_ransac_max_iterations),
    ]


def run_benchmark_cases(args: argparse.Namespace) -> List[Dict[str, object]]:
    cases = build_cases()
    results: List[Dict[str, object]] = []
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    script = Path("code/registration/run_registration_benchmark.py")
    base_output_dir = str(Path(args.output_root))
    for case in cases:
        run_cmd = [args.python, str(script)] + _build_case_args(args, case)
        run_cmd[run_cmd.index("--output_dir") + 1] = f"{base_output_dir}/{case['name']}"
        print(f"[RUN] {case['name']}: {' '.join(run_cmd)}")
        if not args.dry_run:
            subprocess.run(run_cmd, check=True)
        summary_path = output_root / str(case["name"]) / "baseline_summary.json"
        results.append(
            {
                "case": case["name"],
                "output_dir": str(summary_path.parent),
                "dataset_path": args.dataset_path,
                "noise_type": case["noise_type"],
                "num_points": case["num_points"],
                "partial": list(case["partial"]),
                "methods": (
                    None
                    if args.dry_run
                    else json.loads(summary_path.read_text(encoding="utf-8"))["benchmark"]
                ),
            }
        )

    return results


def main() -> None:
    args = parse_args()
    validate_relative_paths(args)
    payload = {
        "cases": run_benchmark_cases(args),
        "args": vars(args),
    }
    output_root = Path(args.output_root)
    (output_root / "suite_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Suite summary written to {output_root / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
