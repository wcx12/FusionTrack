"""Convert GeoTransformer feature dumps to the registration comparison schema.

GeoTransformer writes one ``.npz`` file per pair under
``features/<benchmark>/<scene>/<pair>.npz``.  Each file contains the predicted
``estimated_transform`` and ground-truth ``transform`` mapping source points to
reference points.  This script aggregates those pair files into the same compact
schema used by the non-learning and learned baseline runners.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


POINT_LEVEL_KEYS = {
    "full": ("src_points", "ref_points"),
    "fine": ("src_points_f", "ref_points_f"),
    "coarse": ("src_points_c", "ref_points_c"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features_dir",
        required=True,
        help="GeoTransformer features directory, e.g. output/.../features/3DMatch.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method_key", default="geotransformer")
    parser.add_argument("--pose_trans_weight", type=float, default=50.0)
    parser.add_argument(
        "--points_level",
        choices=sorted(POINT_LEVEL_KEYS),
        default="fine",
        help="Point resolution used for Chamfer. RRE/RTE are independent of this.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional debug limit. Omit for the full benchmark split.",
    )
    return parser.parse_args()


def _is_policy_absolute_path(path_value: str) -> bool:
    return (
        Path(path_value).is_absolute()
        or PurePosixPath(path_value).is_absolute()
        or PureWindowsPath(path_value).is_absolute()
    )


def validate_relative_paths(args: argparse.Namespace) -> None:
    for key in ("features_dir", "output_dir"):
        value = getattr(args, key, None)
        if value is not None and _is_policy_absolute_path(str(value)):
            raise ValueError(f"{key} must be a relative path")


def iter_pair_files(features_dir: Path, max_pairs: int | None = None) -> Iterable[Path]:
    pair_files = sorted(features_dir.glob("*/*.npz"))
    if max_pairs is not None:
        pair_files = pair_files[:max_pairs]
    return pair_files


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation)
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


def rotation_error_deg(pred: np.ndarray, target: np.ndarray) -> float:
    pred_rot = project_rotation(pred[:3, :3])
    target_rot = project_rotation(target[:3, :3])
    residual = target_rot.T @ pred_rot
    trace = float(np.trace(residual))
    value = max(min(0.5 * (trace - 1.0), 1.0), -1.0)
    return math.degrees(math.acos(value))


def translation_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred[:3, 3] - target[:3, 3]))


def chamfer_distance(src: np.ndarray, ref: np.ndarray) -> float:
    ref_tree = cKDTree(ref)
    src_tree = cKDTree(src)
    src_to_ref = ref_tree.query(src, k=1, workers=-1)[0]
    ref_to_src = src_tree.query(ref, k=1, workers=-1)[0]
    return float(np.mean(src_to_ref**2) + np.mean(ref_to_src**2))


def read_pair_metrics(path: Path, points_level: str) -> Dict[str, float]:
    src_key, ref_key = POINT_LEVEL_KEYS[points_level]
    with np.load(path) as data:
        pred = data["estimated_transform"].astype(np.float64)
        target = data["transform"].astype(np.float64)
        src = data[src_key].astype(np.float64)
        ref = data[ref_key].astype(np.float64)
    aligned_src = transform_points(src, pred)
    return {
        "rotation_error_deg": rotation_error_deg(pred, target),
        "translation_error": translation_error(pred, target),
        "chamfer_distance": chamfer_distance(aligned_src, ref),
    }


def mean_and_rmse(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(np.sqrt(np.mean(arr * arr)))


def aggregate_metrics(pair_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    rot_mean, rot_rmse = mean_and_rmse([m["rotation_error_deg"] for m in pair_metrics])
    trans_mean, trans_rmse = mean_and_rmse([m["translation_error"] for m in pair_metrics])
    chamfer_mean, chamfer_rmse = mean_and_rmse([m["chamfer_distance"] for m in pair_metrics])
    return {
        "rotation_error_deg_mean": rot_mean,
        "rotation_error_deg_rmse": rot_rmse,
        "translation_error_mean": trans_mean,
        "translation_error_rmse": trans_rmse,
        "chamfer_distance_mean": chamfer_mean,
        "chamfer_distance_rmse": chamfer_rmse,
        "num_pairs": len(pair_metrics),
    }


def to_comparison_schema(metrics: Dict[str, float], pose_trans_weight: float) -> Dict[str, float]:
    out = dict(metrics)
    out["pose_metric"] = out["rotation_error_deg_mean"] + pose_trans_weight * out["translation_error_mean"]
    return out


def main() -> None:
    args = parse_args()
    validate_relative_paths(args)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    pair_files = list(iter_pair_files(features_dir, args.max_pairs))
    if not pair_files:
        raise FileNotFoundError(f"No GeoTransformer pair files found under {features_dir}")

    pair_metrics = [read_pair_metrics(path, args.points_level) for path in pair_files]
    metrics = aggregate_metrics(pair_metrics)
    comparison = {args.method_key: to_comparison_schema(metrics, args.pose_trans_weight)}
    payload = {
        "features_dir": str(features_dir),
        "method_key": args.method_key,
        "points_level": args.points_level,
        "pose_trans_weight": args.pose_trans_weight,
        "metrics": metrics,
        "comparison_schema": comparison,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "comparison_schema_summary.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
