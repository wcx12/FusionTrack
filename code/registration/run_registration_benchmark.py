"""Run non-learning registration baseline benchmark on ModelNet40-based dataset."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "torch is required to run this benchmark script. Install a working torch build first "
        "(CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, "
        "or CUDA build matching your driver for GPU)."
    ) from exc

try:
    from .non_learning_baselines import (
        BaselineResult,
        baseline_method_names,
        parse_baseline_methods,
        run_non_learning_baseline,
        identity_transform,
    )
    from .mps_gaf_data_pipeline import (
        MPSGAFDataConfig,
        get_test_dataset,
        make_grouped_dataloader,
    )
except Exception:
    from non_learning_baselines import (
        BaselineResult,
        baseline_method_names,
        parse_baseline_methods,
        run_non_learning_baseline,
        identity_transform,
    )
    from mps_gaf_data_pipeline import (
        MPSGAFDataConfig,
        get_test_dataset,
        make_grouped_dataloader,
    )

try:
    from .mps_gaf_registration_core import transform_se3
except Exception:
    from mps_gaf_registration_core import transform_se3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-learning registration baselines")
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to modelnet40_ply_hdf5_2048 (recommend repository-relative path)",
    )
    parser.add_argument(
        "--output_dir",
        default="runs/mps_gaf_nonlearn_baselines",
        help="Output directory (recommend repository-relative path)",
    )
    parser.add_argument(
        "--methods",
        default="icp_point_to_point,icp_point_to_plane,icp_trimmed,ransac_icp",
        help="Comma-separated baseline method names",
    )
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--partial", type=float, nargs="+", default=[0.7, 0.7])
    parser.add_argument("--noise_type", default="crop", choices=["clean", "jitter", "crop"])
    parser.add_argument("--rot_mag", type=float, default=45.0)
    parser.add_argument("--trans_mag", type=float, default=0.5)
    parser.add_argument("--num_sources_per_ref", type=int, default=2)
    parser.add_argument("--groups_per_batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--icp_iterations", type=int, default=20)
    parser.add_argument("--icp_tolerance", type=float, default=1e-6)
    parser.add_argument("--icp_trim_fraction", type=float, default=0.7)
    parser.add_argument("--icp_point_max_angle_deg", type=float, default=10.0)
    parser.add_argument("--icp_point_max_translation", type=float, default=0.2)
    parser.add_argument("--fpfh_voxel_size", type=float, default=0.05)
    parser.add_argument("--fpfh_normal_radius", type=float, default=0.1)
    parser.add_argument("--fpfh_feature_radius", type=float, default=0.25)
    parser.add_argument("--fpfh_normal_max_nn", type=int, default=30)
    parser.add_argument("--fpfh_feature_max_nn", type=int, default=100)
    parser.add_argument("--fpfh_max_correspondence_distance", type=float, default=0.075)
    parser.add_argument("--fpfh_ransac_n", type=int, default=4)
    parser.add_argument("--fpfh_ransac_max_iterations", type=int, default=100000)
    parser.add_argument("--fgr_voxel_size", type=float, default=0.05)
    parser.add_argument("--fgr_normal_radius", type=float, default=0.1)
    parser.add_argument("--fgr_feature_radius", type=float, default=0.25)
    parser.add_argument("--fgr_normal_max_nn", type=int, default=30)
    parser.add_argument("--fgr_feature_max_nn", type=int, default=100)
    parser.add_argument("--fgr_max_correspondence_distance", type=float, default=0.075)
    parser.add_argument("--gicp_max_correspondence_distance", type=float, default=0.075)
    parser.add_argument("--gicp_max_iterations", type=int, default=64)
    parser.add_argument("--cpd_max_iterations", type=int, default=30)
    parser.add_argument("--cpd_tolerance", type=float, default=1e-5)
    parser.add_argument("--cpd_w", type=float, default=0.0)
    parser.add_argument("--ransac_iterations", type=int, default=500)
    parser.add_argument("--ransac_inlier_distance", type=float, default=0.05)
    parser.add_argument("--success_rotation_deg", type=float, default=5.0)
    parser.add_argument("--success_translation", type=float, default=0.2)
    return parser.parse_args()


def validate_relative_paths(args: argparse.Namespace) -> None:
    for key in ("dataset_path", "output_dir"):
        if Path(getattr(args, key)).is_absolute():
            raise ValueError(
                f"{key} is configured as an absolute path. Use a relative path per benchmark policy."
            )


def rotation_error_deg(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred[:3, :3]
    target = target[:3, :3]
    residual = target.t() @ pred
    trace = residual[0, 0] + residual[1, 1] + residual[2, 2]
    value = 0.5 * (trace - 1.0)
    value = max(min(float(value), 1.0 - 1e-6), -1.0 + 1e-6)
    return float(math.degrees(math.acos(value)))


def translation_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.norm(pred[:3, 3] - target[:3, 3], p=2))


def chamfer_distance_numpy(src: torch.Tensor, ref: torch.Tensor) -> float:
    if hasattr(src, "detach"):
        src_t = src.detach().cpu().numpy()
    else:
        src_t = src
    if hasattr(ref, "detach"):
        ref_t = ref.detach().cpu().numpy()
    else:
        ref_t = ref

    src_t = np.asarray(src_t)[:, :3]
    ref_t = np.asarray(ref_t)[:, :3]
    diff = src_t[:, None, :] - ref_t[None, :, :]
    dist2 = (diff * diff).sum(axis=-1)
    src_to_ref = dist2.min(axis=1).mean()
    ref_to_src = dist2.min(axis=0).mean()
    return float(src_to_ref + ref_to_src)


def build_metric_accumulator() -> Dict[str, float]:
    return {
        "attempts": 0.0,
        "count": 0.0,
        "rotation_error_deg_sum": 0.0,
        "rotation_error_deg_sq_sum": 0.0,
        "translation_error_sum": 0.0,
        "translation_error_sq_sum": 0.0,
        "chamfer_sum": 0.0,
        "chamfer_sq_sum": 0.0,
        "runtime_sum": 0.0,
        "success_sum": 0.0,
        "failures": 0.0,
    }


def update_metrics(
    acc: Dict[str, float],
    rot: float,
    trans: float,
    chamfer: float,
    runtime: float,
    success: int,
    valid: bool,
) -> None:
    acc["attempts"] += 1.0
    acc["runtime_sum"] += runtime
    acc["success_sum"] += float(success)
    if not valid:
        acc["failures"] += 1.0
        return

    acc["count"] += 1.0
    acc["rotation_error_deg_sum"] += rot
    acc["rotation_error_deg_sq_sum"] += rot * rot
    acc["translation_error_sum"] += trans
    acc["translation_error_sq_sum"] += trans * trans
    acc["chamfer_sum"] += chamfer
    acc["chamfer_sq_sum"] += chamfer * chamfer


def finalize_metrics(acc: Dict[str, float]) -> Dict[str, float]:
    attempts = acc["attempts"]
    if attempts <= 0.0:
        raise ValueError("No samples were evaluated")

    count = acc["count"]
    failures = int(acc["failures"])

    if count <= 0.0:
        return {
            "num_pairs": int(attempts),
            "num_successful_pairs": 0,
            "num_failed_pairs": failures,
            "rotation_error_deg_mean": float("inf"),
            "rotation_error_deg_rmse": float("inf"),
            "translation_error_mean": float("inf"),
            "translation_error_rmse": float("inf"),
            "chamfer_distance_mean": float("inf"),
            "chamfer_distance_rmse": float("inf"),
            "runtime_sec_mean": acc["runtime_sum"] / attempts,
            "success_rate": 0.0,
            "skip_rate": float(failures) / attempts,
            "failures": failures,
        }

    return {
        "num_pairs": int(attempts),
        "num_successful_pairs": int(count),
        "num_failed_pairs": failures,
        "rotation_error_deg_mean": acc["rotation_error_deg_sum"] / count,
        "rotation_error_deg_rmse": math.sqrt(acc["rotation_error_deg_sq_sum"] / count),
        "translation_error_mean": acc["translation_error_sum"] / count,
        "translation_error_rmse": math.sqrt(acc["translation_error_sq_sum"] / count),
        "chamfer_distance_mean": acc["chamfer_sum"] / count,
        "chamfer_distance_rmse": math.sqrt(acc["chamfer_sq_sum"] / count),
        "runtime_sec_mean": acc["runtime_sum"] / attempts,
        "success_rate": acc["success_sum"] / attempts,
        "skip_rate": acc["failures"] / attempts,
        "failures": failures,
    }


def _to_torch_matrix(result: BaselineResult) -> torch.Tensor:
    return torch.tensor(result.transform, dtype=torch.float32)


def _benchmark_methods_case(
    method: str,
    args: argparse.Namespace,
    src: np.ndarray,
    ref: np.ndarray,
    src_n: np.ndarray,
    ref_n: np.ndarray,
) -> tuple[BaselineResult, dict[str, float] | None, str, bool]:
    kwargs = {}

    try:
        if method in {"icp_point_to_point", "icp"}:
            kwargs = {
                "iterations": args.icp_iterations,
                "tolerance": args.icp_tolerance,
                "trim_fraction": 1.0,
            }
            result = run_non_learning_baseline(method, src, ref, **kwargs)
        elif method in {"icp_point_to_plane", "icp_point_to_plane_refine"}:
            kwargs = {
                "iterations": args.icp_iterations,
                "tolerance": args.icp_tolerance,
                "max_angle_deg": args.icp_point_max_angle_deg,
                "max_translation": args.icp_point_max_translation,
            }
            result = run_non_learning_baseline(
                method, src, ref, src_normals=src_n, ref_normals=ref_n, **kwargs
            )
        elif method in {"icp_trimmed", "trimmed_icp"}:
            kwargs = {
                "iterations": args.icp_iterations,
                "tolerance": args.icp_tolerance,
                "trim_fraction": args.icp_trim_fraction,
            }
            result = run_non_learning_baseline(method, src, ref, **kwargs)
        elif method == "ransac_icp":
            kwargs = {
                "iterations": args.ransac_iterations,
                "inlier_distance": args.ransac_inlier_distance,
                "random_seed": args.seed,
                "refine_iterations": 10,
            }
            result = run_non_learning_baseline(method, src, ref, **kwargs)
        elif method in {"fpfh_ransac", "fpfh_fgr", "fgr", "fpfh_fgr_icp"}:
            if method == "fpfh_ransac":
                kwargs = {
                    "voxel_size": args.fpfh_voxel_size,
                    "normal_radius": args.fpfh_normal_radius,
                    "feature_radius": args.fpfh_feature_radius,
                    "normal_max_nn": args.fpfh_normal_max_nn,
                    "feature_max_nn": args.fpfh_feature_max_nn,
                    "max_correspondence_distance": args.fpfh_max_correspondence_distance,
                    "ransac_n": args.fpfh_ransac_n,
                    "ransac_max_iterations": args.fpfh_ransac_max_iterations,
                }
            else:
                kwargs = {
                    "voxel_size": args.fgr_voxel_size,
                    "normal_radius": args.fgr_normal_radius,
                    "feature_radius": args.fgr_feature_radius,
                    "normal_max_nn": args.fgr_normal_max_nn,
                    "feature_max_nn": args.fgr_feature_max_nn,
                    "max_correspondence_distance": args.fgr_max_correspondence_distance,
                }
            result = run_non_learning_baseline(method, src, ref, **kwargs)
        elif method in {"gicp", "generalized_icp"}:
            kwargs = {
                "max_correspondence_distance": args.gicp_max_correspondence_distance,
                "max_iterations": args.gicp_max_iterations,
            }
            result = run_non_learning_baseline(method, src, ref, **kwargs)
        elif method in {"cpd", "cpd_rigid"}:
            kwargs = {
                "max_iterations": args.cpd_max_iterations,
                "tolerance": args.cpd_tolerance,
                "w": args.cpd_w,
            }
            result = run_non_learning_baseline(method, src, ref, **kwargs)
        elif method in {"teaserpp", "teaser", "super4pcs", "goicp", "go_icp"}:
            result = run_non_learning_baseline(method, src, ref)
        else:
            result = run_non_learning_baseline(method, src, ref)
        return result, kwargs, "", True
    except Exception as exc:  # noqa: BLE001
        return (
            BaselineResult(
                transform=identity_transform(),
                runtime_sec=0.0,
                meta={"status": "failed", "error": str(exc), "attempted_method": method},
            ),
            kwargs,
            str(exc),
            False,
        )


def benchmark_methods(args: argparse.Namespace) -> Dict[str, object]:
    data_config = MPSGAFDataConfig(
        dataset_path=args.dataset_path,
        num_points=args.num_points,
        noise_type=args.noise_type,
        rot_mag=args.rot_mag,
        trans_mag=args.trans_mag,
        partial=tuple(args.partial),
        num_sources_per_ref=args.num_sources_per_ref,
        seed=args.seed,
    )
    test_dataset = get_test_dataset(data_config)
    test_loader = make_grouped_dataloader(
        test_dataset,
        groups_per_batch=args.groups_per_batch,
        shuffle_groups=False,
        num_workers=args.num_workers,
    )

    methods = parse_baseline_methods(args.methods)
    metric_accumulators = {method: build_metric_accumulator() for method in methods}
    pair_results: List[dict] = []

    for batch_idx, batch in enumerate(test_loader, start=1):
        points_src = batch["points_src"]
        points_ref = batch["points_ref"]
        transform_gt = batch["transform_gt"]
        group_refs = batch.get("group_ref_idx", torch.zeros(points_src.shape[0], dtype=torch.int64))

        if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
            break

        for sample_idx in range(points_src.shape[0]):
            src = points_src[sample_idx].cpu().numpy()
            ref = points_ref[sample_idx].cpu().numpy()
            src_n = src[:, 3:6]
            ref_n = ref[:, 3:6]
            gt = transform_gt[sample_idx]

            for method in methods:
                result, call_kwargs, error_msg, valid = _benchmark_methods_case(
                    method, args, src, ref, src_n, ref_n
                )
                pred = _to_torch_matrix(result)
                if valid:
                    pred_tgt = transform_se3(
                        pred.unsqueeze(0), points_src[sample_idx][..., :3].unsqueeze(0)
                    )[0]
                    gt_tgt = transform_se3(gt.unsqueeze(0), points_src[sample_idx][..., :3].unsqueeze(0))[0]
                    chamfer = chamfer_distance_numpy(pred_tgt, points_ref[sample_idx][:, :3])
                    rot = rotation_error_deg(pred, gt)
                    trans = translation_error(pred, gt)
                    success = int(rot <= args.success_rotation_deg and trans <= args.success_translation)
                else:
                    pred_tgt = points_ref[sample_idx][:, :3]
                    rot = 0.0
                    trans = 0.0
                    chamfer = 0.0
                    success = 0

                update_metrics(
                    metric_accumulators[method], rot, trans, chamfer, result.runtime_sec, success, valid
                )

                pair_results.append(
                    {
                        "batch_idx": int(batch_idx),
                        "sample_idx": int(sample_idx),
                        "group_ref_idx": int(group_refs[sample_idx].item()),
                        "method": method,
                        "rotation_error_deg": None if not valid else float(rot),
                        "translation_error": None if not valid else float(trans),
                        "success": bool(success),
                        "chamfer_distance": None if not valid else float(chamfer),
                        "runtime_sec": float(result.runtime_sec),
                        "skipped": bool(not valid),
                        "error": error_msg,
                        "meta": {**result.meta, "call_kwargs": call_kwargs},
                    }
                )

    summary: Dict[str, dict] = {
        method: finalize_metrics(metric_accumulators[method]) for method in methods
    }
    return {
        "methods": methods,
        "supported_methods": baseline_method_names(),
        "benchmark": summary,
        "pair_results": pair_results,
    }


def main() -> None:
    args = parse_args()
    validate_relative_paths(args)
    available = set(baseline_method_names())
    requested = parse_baseline_methods(args.methods)
    unknown = set(requested) - available
    if unknown:
        raise ValueError(f"Unsupported method(s): {sorted(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload = benchmark_methods(args)
    payload["benchmark_time_sec"] = float(time.perf_counter() - started)
    payload["args"] = vars(args)

    (output_dir / "baseline_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["benchmark"], indent=2))


if __name__ == "__main__":
    main()
