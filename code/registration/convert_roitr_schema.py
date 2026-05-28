"""Convert RoITr official registration logs into the project comparison schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Iterable, List, Tuple

import numpy as np


MatrixEntry = Tuple[Tuple[int, int], np.ndarray]


def _reject_absolute(path_text: str, name: str) -> None:
    if (
        Path(path_text).is_absolute()
        or PurePosixPath(path_text).is_absolute()
        or PureWindowsPath(path_text).is_absolute()
    ):
        raise ValueError(f"{name} must be a relative path")


def parse_registration_log(path: Path) -> List[MatrixEntry]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    entries: List[MatrixEntry] = []
    index = 0
    while index < len(lines):
        header = lines[index].split()
        if len(header) < 2:
            raise ValueError(f"Malformed registration header in {path}: {lines[index]}")
        try:
            pair = (int(header[0]), int(header[1]))
        except ValueError as exc:
            raise ValueError(f"Malformed fragment ids in {path}: {lines[index]}") from exc
        matrix_lines = lines[index + 1 : index + 5]
        if len(matrix_lines) != 4:
            raise ValueError(f"Incomplete matrix for pair {pair} in {path}")
        matrix = np.array([[float(value) for value in row.split()] for row in matrix_lines], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix for pair {pair} in {path} is not 4x4")
        entries.append((pair, matrix))
        index += 5
    return entries


def project_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1
        projected = u @ vt
    return projected


def rotation_error_deg(pred: np.ndarray, target: np.ndarray) -> float:
    pred_r = project_rotation(pred[:3, :3])
    target_r = project_rotation(target[:3, :3])
    relative = target_r.T @ pred_r
    trace_value = float(np.trace(relative))
    cos_theta = max(-1.0, min(1.0, (trace_value - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta))


def translation_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred[:3, 3] - target[:3, 3]))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _rmse(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2))) if values else float("nan")


def summarize_roitr(est_root: Path, gt_root: Path, translation_weight: float = 50.0) -> Dict[str, object]:
    scene_summaries: Dict[str, Dict[str, float]] = {}
    rotation_errors: List[float] = []
    translation_errors: List[float] = []

    for scene_dir in sorted(path for path in est_root.iterdir() if path.is_dir()):
        est_log = scene_dir / "est.log"
        gt_log = gt_root / scene_dir.name / "gt.log"
        if not est_log.exists() or not gt_log.exists():
            continue
        est_entries = dict(parse_registration_log(est_log))
        gt_entries = dict(parse_registration_log(gt_log))
        common_pairs = sorted(set(est_entries) & set(gt_entries))
        scene_rre = [rotation_error_deg(est_entries[pair], gt_entries[pair]) for pair in common_pairs]
        scene_rte = [translation_error(est_entries[pair], gt_entries[pair]) for pair in common_pairs]
        rotation_errors.extend(scene_rre)
        translation_errors.extend(scene_rte)
        scene_summaries[scene_dir.name] = {
            "pairs": len(common_pairs),
            "rotation_error_deg_mean": _mean(scene_rre),
            "translation_error_mean": _mean(scene_rte),
            "pose_metric": _mean(scene_rre) + translation_weight * _mean(scene_rte),
        }

    rre = _mean(rotation_errors)
    rte = _mean(translation_errors)
    return {
        "pairs": len(rotation_errors),
        "num_pairs": len(rotation_errors),
        "rotation_error_deg_mean": rre,
        "rotation_error_deg_rmse": _rmse(rotation_errors),
        "translation_error_mean": rte,
        "translation_error_rmse": _rmse(translation_errors),
        "chamfer_distance_mean": None,
        "pose_metric": rre + translation_weight * rte,
        "translation_weight": translation_weight,
        "scenes": scene_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--est_root", required=True, help="RoITr est_traj benchmark/n_points directory")
    parser.add_argument("--gt_root", required=True, help="RoITr configs/benchmarks benchmark directory")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--translation_weight", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key in ("est_root", "gt_root", "output"):
        value = getattr(args, key)
        if value is not None:
            _reject_absolute(value, key)

    summary = summarize_roitr(Path(args.est_root), Path(args.gt_root), args.translation_weight)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
