"""Run non-learning baseline sweeps for multiple success-threshold settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import math
import subprocess


def _fmt_name(value: float) -> str:
    text = f"{value}".replace("-", "m").replace(".", "p")
    return text.rstrip("0").rstrip(".") or "0"


def _parse_float_list(raw: str) -> List[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-learning baselines across threshold grid.")
    parser.add_argument(
        "--dataset_path",
        default="datasets/modelnet40_ply_hdf5_2048",
        help="Path to dataset (relative path).",
    )
    parser.add_argument(
        "--dataset_split",
        default="test",
        choices=["test", "train"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument("--output_root", default="runs/mps_gaf_nonlearn_threshold_sweep")
    parser.add_argument(
        "--methods",
        default="identity,icp_point_to_point,icp_point_to_plane,icp_trimmed,ransac_icp,fpfh_ransac,fpfh_fgr,gicp,cpd",
        help="Comma-separated methods.",
    )
    parser.add_argument(
        "--case_set",
        default="protocol",
        choices=["protocol", "robustness"],
        help="Case preset passed to run_registration_benchmark_suite.py.",
    )
    parser.add_argument("--rot_mag", type=float, default=45.0)
    parser.add_argument("--trans_mag", type=float, default=0.5)
    parser.add_argument("--num_sources_per_ref", type=int, default=2)
    parser.add_argument("--groups_per_batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=20)
    parser.add_argument("--icp_iterations", type=int, default=20)
    parser.add_argument("--icp_trim_fraction", type=float, default=0.7)
    parser.add_argument(
        "--success_rotation_degs",
        type=_parse_float_list,
        default=_parse_float_list("5,10,15"),
    )
    parser.add_argument(
        "--success_translations",
        type=_parse_float_list,
        default=_parse_float_list("0.2,0.3,0.5"),
    )
    parser.add_argument("--icp_point_max_angle_deg", type=float, default=10.0)
    parser.add_argument("--fpfh_voxel_size", type=float, default=0.05)
    parser.add_argument("--fpfh_normal_radius", type=float, default=0.1)
    parser.add_argument("--fpfh_feature_radius", type=float, default=0.25)
    parser.add_argument("--fpfh_normal_max_nn", type=int, default=30)
    parser.add_argument("--fpfh_feature_max_nn", type=int, default=100)
    parser.add_argument("--fpfh_max_correspondence_distance", type=float, default=0.075)
    parser.add_argument("--fpfh_ransac_n", type=int, default=4)
    parser.add_argument("--fpfh_ransac_max_iterations", type=int, default=100000)
    parser.add_argument("--icp_tolerance", type=float, default=1e-6)
    parser.add_argument("--icp_point_max_translation", type=float, default=0.2)
    parser.add_argument("--python", default="python", help="Python executable.")
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Reuse existing suite_summary.json files instead of rerunning benchmark cases.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def validate_relative_paths(args: argparse.Namespace) -> None:
    for key in ("dataset_path", "output_root"):
        if Path(getattr(args, key)).is_absolute():
            raise ValueError(f"{key} must be a relative path by benchmark policy.")


def build_suite_args(args: argparse.Namespace, run_output_dir: str, rot: float, trans: float) -> List[str]:
    return [
        "code/registration/run_registration_benchmark_suite.py",
        "--dataset_path",
        args.dataset_path,
        "--dataset_split",
        args.dataset_split,
        "--output_root",
        run_output_dir,
        "--methods",
        args.methods,
        "--case_set",
        args.case_set,
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
        str(rot),
        "--success_translation",
        str(trans),
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


def aggregate_metrics(suite_summary: Dict) -> Dict[str, Dict[str, float]]:
    methods = sorted({method for case in suite_summary["cases"] for method in case["methods"].keys()})
    out: Dict[str, Dict[str, float]] = {}
    for method in methods:
        vals = [c["methods"][method] for c in suite_summary["cases"] if method in c["methods"]]
        n = len(vals)
        success = sum(v["success_rate"] for v in vals) / n
        skip = sum(v["skip_rate"] for v in vals) / n
        runtime = sum(v["runtime_sec_mean"] for v in vals) / n
        pairs = sum(v["num_pairs"] for v in vals)
        failed = sum(v["num_failed_pairs"] for v in vals)
        good = [v for v in vals if v["num_successful_pairs"]]
        if good:
            chamfer = sum(v["chamfer_distance_mean"] for v in good) / len(good)
            rotation = sum(v["rotation_error_deg_mean"] for v in good) / len(good)
            trans = sum(v["translation_error_mean"] for v in good) / len(good)
        else:
            chamfer = math.inf
            rotation = math.inf
            trans = math.inf
        out[method] = {
            "success_rate": float(success),
            "skip_rate": float(skip),
            "rotation_error_deg_mean": float(rotation),
            "translation_error_mean": float(trans),
            "chamfer_distance_mean": float(chamfer),
            "runtime_sec_mean": float(runtime),
            "num_pairs": int(pairs),
            "num_failed_pairs": int(failed),
        }
    return out


def run_suite(args: argparse.Namespace, run_output: Path, rot: float, trans: float) -> Dict:
    cmd = [args.python, *build_suite_args(args, str(run_output), rot, trans)]
    run_output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        return {"dry_run": True, "command": cmd}
    payload_path = run_output / "suite_summary.json"
    if args.reuse_existing and payload_path.exists():
        return json.loads(payload_path.read_text(encoding="utf-8"))
    subprocess.run(cmd, check=True)
    return json.loads(payload_path.read_text(encoding="utf-8"))


def write_markdown_summary(payloads: Dict[str, Dict], output_path: Path) -> None:
    lines: List[str] = ["# Non-learning Baseline Threshold Sweep", ""]
    for threshold_key, payload in payloads.items():
        lines.append(f"## Threshold {threshold_key}")
        lines.append("")
        lines.append("| case | method | rot/deg | trans | chamfer | success | skip | runtime | failed_pairs |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for case in payload["cases"]:
            case_name = case["case"]
            for method, metrics in sorted(case["methods"].items()):
                lines.append(
                    f"|{case_name}|{method}|"
                    f"{metrics['rotation_error_deg_mean']:0.3f}|{metrics['translation_error_mean']:0.3f}|{metrics['chamfer_distance_mean']:0.3f}|"
                    f"{metrics['success_rate']:0.3f}|{metrics['skip_rate']:0.3f}|{metrics['runtime_sec_mean']:0.4f}|{metrics['num_failed_pairs']}|"
                )
        lines.append("")
        lines.append("|method|rot/deg|trans|chamfer|success|skip|runtime|pairs|failed|")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for method, metrics in sorted(payload["macro"].items(), key=lambda kv: kv[0]):
            m = metrics
            lines.append(
                f"|{method}|{m['rotation_error_deg_mean']:0.3f}|{m['translation_error_mean']:0.3f}|{m['chamfer_distance_mean']:0.3f}|"
                f"{m['success_rate']:0.3f}|{m['skip_rate']:0.3f}|{m['runtime_sec_mean']:0.4f}|{m['num_pairs']}|{m['num_failed_pairs']}|"
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_relative_paths(args)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payloads: Dict[str, Dict] = {}

    for rot in args.success_rotation_degs:
        for trans in args.success_translations:
            label = f"rot_{_fmt_name(rot)}_trans_{_fmt_name(trans)}"
            run_output = output_root / label
            suite_summary = run_suite(args, run_output, rot=rot, trans=trans)
            if args.dry_run:
                continue
            macro = aggregate_metrics(suite_summary)
            payloads[label] = {"cases": suite_summary["cases"], "macro": macro}
            (run_output / "sweep_summary.json").write_text(
                json.dumps({"label": label, "macro": macro}, indent=2),
                encoding="utf-8",
            )

    if args.dry_run:
        return

    all_payload = {
        "runs": {
            label: {"args": item["macro"]} for label, item in payloads.items()
        },
        "summary": {
            label: item["macro"] for label, item in payloads.items()
        },
        "sweep_args": {
            "success_rotation_degs": args.success_rotation_degs,
            "success_translations": args.success_translations,
            "dataset_split": args.dataset_split,
            "case_set": args.case_set,
            "methods": args.methods,
            "num_sources_per_ref": args.num_sources_per_ref,
            "groups_per_batch": args.groups_per_batch,
            "reuse_existing": args.reuse_existing,
        },
    }
    (output_root / "threshold_sweep_payload.json").write_text(
        json.dumps(all_payload, indent=2),
        encoding="utf-8",
    )
    write_markdown_summary(payloads, output_root / "threshold_sweep_summary.md")


if __name__ == "__main__":
    main()
