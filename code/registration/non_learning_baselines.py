"""Non-learning registration baselines for point cloud alignment.

This module implements a practical subset of classical registration baselines that
can be used as a first-stage comparison suite for MPS-GAF.

Baseline methods:
- identity
- point-to-point ICP
- point-to-plane ICP
- trimmed point-to-point ICP
- RANSAC + point-to-point ICP refinement
- FPFH + RANSAC (Open3D optional)
- FPFH + Fast Global Registration (Open3D optional)
- Generalized ICP (Open3D optional)
- CPD via Gaussian Mixture Model (pycpd optional)
- FPFH + MAC / SC2-PCR style spatial compatibility filtering (Open3D optional)
- KISS-Matcher robust global registration (kiss-matcher optional)
- TEASER++ / TurboReg / Super4PCS / Go-ICP (optional external dependencies)
"""

from __future__ import annotations

import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree


Transform3x4 = np.ndarray
PointCloud = np.ndarray


@dataclass(frozen=True)
class BaselineResult:
    """Return value for one baseline registration call."""

    transform: Transform3x4
    runtime_sec: float
    meta: Dict[str, float | int | bool | str]


def _as_float32(points: np.ndarray) -> np.ndarray:
    if not isinstance(points, np.ndarray):
        raise TypeError("Point cloud input must be a NumPy array")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Point cloud must have shape [N, 3+]")
    return points.astype(np.float32)


def _as_rigid_transform(matrix: Transform3x4) -> Transform3x4:
    if matrix.shape != (3, 4):
        raise ValueError(f"Rigid transform must be 3x4, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Transform contains NaN/inf")
    return matrix.astype(np.float32)


def identity_transform() -> Transform3x4:
    """Return a 3x4 identity SE(3) transform."""

    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)


def compose_transforms(a: Transform3x4, b: Transform3x4) -> Transform3x4:
    """Compose two row-vector transforms: apply ``b`` then ``a``.
    """

    a_r = a[:, :3]
    b_r = b[:, :3]
    a_t = a[:, 3]
    b_t = b[:, 3]
    out_r = a_r @ b_r
    out_t = b_t @ a_r.T + a_t
    return np.concatenate([out_r, out_t[:, None]], axis=1).astype(np.float32)


def apply_transform(points: PointCloud, transform: Transform3x4) -> PointCloud:
    """Apply transform to points with shape [N, 3]."""

    points_f = _as_float32(points)
    if points_f.shape[1] < 3:
        raise ValueError("points must contain xyz columns")
    return points_f[:, :3] @ transform[:3, :3].T + transform[:, 3][None, :]


def weighted_kabsch(src: PointCloud, dst: PointCloud, weights: np.ndarray | None = None) -> Transform3x4:
    """Rigid SVD alignment from ``src`` to ``dst``.

    Returns transform in row-vector form: ``x_t = x @ R^T + t``.
    """

    src = _as_float32(src)
    dst = _as_float32(dst)
    if src.shape != dst.shape:
        raise ValueError("src and dst must have identical shape")
    if src.shape[0] < 3:
        raise ValueError("At least 3 correspondences are required")

    if weights is None:
        weights = np.ones((src.shape[0],), dtype=np.float32)
    else:
        weights = np.clip(weights.astype(np.float32), 0.0, None)
        if weights.shape != (src.shape[0],):
            raise ValueError("weights must be a vector of length N")

    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        raise ValueError("sum(weights) must be > 0")
    ws = weights / w_sum

    centroid_src = np.sum(src[:, :3] * ws[:, None], axis=0)
    centroid_dst = np.sum(dst[:, :3] * ws[:, None], axis=0)
    src_centered = src[:, :3] - centroid_src[None, :]
    dst_centered = dst[:, :3] - centroid_dst[None, :]
    covariance = src_centered.T @ (dst_centered * ws[:, None])

    u, _, vt = np.linalg.svd(covariance, full_matrices=True)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T

    trans = centroid_dst - centroid_src @ rot.T
    return np.concatenate([rot.astype(np.float32), trans[None, :].astype(np.float32).T], axis=1)


def _load_open3d() -> object:
    """Import Open3D lazily so environments without this optional dependency still work."""

    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Open3D is not installed. Install it to enable Open3D baselines: "
            "`pip install open3d`."
        ) from exc
    return o3d


def _load_pycpd() -> object:
    """Import pycpd lazily for CPD-based baseline."""

    try:
        from pycpd import RigidRegistration  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pycpd is not installed. Install it to enable CPD baselines: `pip install pycpd`."
        ) from exc
    return RigidRegistration


def _load_kiss_matcher() -> object:
    """Import KISS-Matcher lazily for its optional robust registration baseline."""

    try:
        import kiss_matcher as km  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "kiss-matcher is not installed. Install it to enable KISS-Matcher: "
            "`pip install kiss-matcher`."
        ) from exc
    return km


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_policy_absolute_path(path_value: str) -> bool:
    return (
        Path(path_value).is_absolute()
        or PurePosixPath(path_value).is_absolute()
        or PureWindowsPath(path_value).is_absolute()
    )


def _relative_path_candidates(path_value: str) -> List[Path]:
    path = Path(path_value)
    if _is_policy_absolute_path(path_value):
        raise RuntimeError(
            f"External dependency path must be relative to the repository root: {path_value}"
        )
    return [Path.cwd() / path, _repo_root() / path]


def _resolve_existing_relative_path(
    path_value: str | None,
    env_key: str,
    default_relative_path: str,
) -> Path:
    configured = os.environ.get(env_key) or path_value or default_relative_path
    for candidate in _relative_path_candidates(configured):
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"Could not find {env_key} target. Expected a relative path such as "
        f"`{default_relative_path}` or set {env_key} to a repository-relative path."
    )


def _build_o3d_point_cloud(points: PointCloud, o3d: object) -> "open3d.geometry.PointCloud":
    """Build an Open3D point cloud from raw xyz coordinates."""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)[:, :3])
    return pcd


def _estimate_fpfh_features(
    points_xyz: PointCloud,
    o3d: object,
    normal_radius: float,
    feature_radius: float,
    normal_max_nn: int,
    feature_max_nn: int,
) -> object:
    """Estimate FPFH features on normalized point cloud scale.

    The scale is tuned using voxel/neighbor radii that are usually robust across
    ModelNet40 and similar unit-normalized datasets.
    """

    pcd = _build_o3d_point_cloud(points_xyz, o3d)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn,
        )
    )
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=feature_radius,
            max_nn=feature_max_nn,
        ),
    )


def _build_downsampled_o3d_pair(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    o3d: object,
    voxel_size: float,
) -> tuple[object, object]:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    src_pcd = _build_o3d_point_cloud(src_xyz, o3d).voxel_down_sample(voxel_size)
    ref_pcd = _build_o3d_point_cloud(ref_xyz, o3d).voxel_down_sample(voxel_size)
    if len(src_pcd.points) < 3 or len(ref_pcd.points) < 3:
        raise ValueError("Voxel down-sampling removed too few points")
    return src_pcd, ref_pcd


def _convert_open3d_to_row_transform(transform_4x4: np.ndarray) -> Transform3x4:
    """Convert Open3D 4x4 column-vector transform to row-vector 3x4 format."""

    if transform_4x4.shape != (4, 4):
        raise ValueError("Expected a 4x4 transform")
    rot = transform_4x4[:3, :3].astype(np.float32)
    trans = transform_4x4[:3, 3].astype(np.float32)
    return np.concatenate([rot, trans[:, None]], axis=1).astype(np.float32)


def _normalize_pair_for_external_solver(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
) -> tuple[PointCloud, PointCloud, np.ndarray, np.ndarray, float]:
    src = _as_float32(src_xyz)[:, :3]
    ref = _as_float32(ref_xyz)[:, :3]
    src_center = src.mean(axis=0).astype(np.float32)
    ref_center = ref.mean(axis=0).astype(np.float32)
    src_centered = src - src_center[None, :]
    ref_centered = ref - ref_center[None, :]
    scale = float(max(np.max(np.abs(src_centered)), np.max(np.abs(ref_centered)), 1e-6))
    return (
        (src_centered / scale).astype(np.float32),
        (ref_centered / scale).astype(np.float32),
        src_center,
        ref_center,
        scale,
    )


def _denormalize_external_transform(
    rotation: np.ndarray,
    translation_normalized: np.ndarray,
    src_center: np.ndarray,
    ref_center: np.ndarray,
    scale: float,
) -> Transform3x4:
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    translation_normalized = np.asarray(translation_normalized, dtype=np.float32).reshape(3)
    translation = ref_center + scale * translation_normalized - rotation @ src_center
    return _as_rigid_transform(np.concatenate([rotation, translation[:, None]], axis=1))


def _write_goicp_points(path: Path, points_xyz: PointCloud) -> None:
    points = _as_float32(points_xyz)[:, :3]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{points.shape[0]}\n")
        for x, y, z in points:
            handle.write(f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n")


def _write_obj_vertices(path: Path, points_xyz: PointCloud) -> None:
    points = _as_float32(points_xyz)[:, :3]
    with path.open("w", encoding="utf-8") as handle:
        for x, y, z in points:
            handle.write(f"v {float(x):.8f} {float(y):.8f} {float(z):.8f}\n")


def _parse_first_4x4_matrix(text: str) -> np.ndarray:
    rows: List[List[float]] = []
    in_transform_block = False
    for line in text.splitlines():
        if "Transformation from" in line:
            in_transform_block = True
            rows = []
            continue
        if not in_transform_block:
            continue
        values = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", line)]
        if len(values) == 4:
            rows.append(values)
            if len(rows) == 4:
                return np.asarray(rows, dtype=np.float32)
        elif rows:
            rows = []
    raise RuntimeError("Could not parse 4x4 transform matrix from solver output")


def _parse_goicp_output(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values: List[List[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        numbers = [float(item) for item in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", line)]
        if numbers:
            values.append(numbers)
    if len(values) < 7:
        raise RuntimeError("Go-ICP output did not contain a full rotation and translation")
    rotation = np.asarray(values[1:4], dtype=np.float32)
    translation = np.asarray([values[4][0], values[5][0], values[6][0]], dtype=np.float32)
    return rotation, translation


def _tail_text(text: str, max_chars: int = 1200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _fpfh_correspondences(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    voxel_size: float,
    normal_radius: float,
    feature_radius: float,
    normal_max_nn: int,
    feature_max_nn: int,
    max_correspondences: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    o3d = _load_open3d()
    src_down, ref_down = _build_downsampled_o3d_pair(src_xyz, ref_xyz, o3d, voxel_size)
    src_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    ref_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    src_feat = o3d.pipelines.registration.compute_fpfh_feature(
        src_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=feature_max_nn),
    )
    ref_feat = o3d.pipelines.registration.compute_fpfh_feature(
        ref_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=feature_max_nn),
    )

    src_desc = np.asarray(src_feat.data, dtype=np.float32).T
    ref_desc = np.asarray(ref_feat.data, dtype=np.float32).T
    src_points = np.asarray(src_down.points, dtype=np.float32)
    ref_points = np.asarray(ref_down.points, dtype=np.float32)
    if src_desc.shape[0] < 3 or ref_desc.shape[0] < 3:
        raise RuntimeError("FPFH downsampling produced too few descriptors")

    feature_tree = cKDTree(ref_desc)
    distances, nn_idx = feature_tree.query(src_desc, k=1)
    order = np.argsort(distances)
    if max_correspondences > 0:
        order = order[: max(3, min(int(max_correspondences), order.shape[0]))]
    return src_points[order], ref_points[nn_idx[order]], np.asarray(distances[order], dtype=np.float32)


def _spatial_compatibility(
    src_corr: PointCloud,
    ref_corr: PointCloud,
    compatibility_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if compatibility_distance <= 0.0:
        raise ValueError("compatibility_distance must be > 0")
    src = _as_float32(src_corr)[:, :3]
    ref = _as_float32(ref_corr)[:, :3]
    if src.shape != ref.shape or src.shape[0] < 3:
        raise ValueError("Spatial compatibility requires at least 3 paired correspondences")

    src_dist = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=-1)
    ref_dist = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
    dist_diff = np.abs(src_dist - ref_dist)
    adjacency = dist_diff <= float(compatibility_distance)
    np.fill_diagonal(adjacency, False)
    return adjacency, dist_diff.astype(np.float32)


def _greedy_largest_clique(
    adjacency: np.ndarray,
    descriptor_distances: np.ndarray,
    max_seeds: int,
) -> np.ndarray:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    n_corr = adjacency.shape[0]
    if n_corr < 3:
        return np.arange(n_corr, dtype=np.int64)

    degrees = adjacency.sum(axis=1)
    descriptor_distances = np.asarray(descriptor_distances, dtype=np.float32).reshape(-1)
    if descriptor_distances.shape[0] != n_corr:
        descriptor_distances = np.zeros((n_corr,), dtype=np.float32)

    ranked = sorted(range(n_corr), key=lambda idx: (-int(degrees[idx]), float(descriptor_distances[idx])))
    descriptor_ranked = sorted(range(n_corr), key=lambda idx: float(descriptor_distances[idx]))
    seeds = []
    for idx in [*ranked, *descriptor_ranked]:
        if idx not in seeds:
            seeds.append(idx)
        if len(seeds) >= max(1, int(max_seeds)):
            break

    best: List[int] = []
    best_descriptor_mean = float("inf")
    for seed in seeds:
        clique = [int(seed)]
        candidates = np.flatnonzero(adjacency[seed])
        while candidates.size:
            local_degree = adjacency[np.ix_(candidates, candidates)].sum(axis=1)
            order = sorted(
                range(candidates.size),
                key=lambda pos: (
                    -int(local_degree[pos]),
                    -int(degrees[candidates[pos]]),
                    float(descriptor_distances[candidates[pos]]),
                ),
            )
            next_idx = int(candidates[order[0]])
            clique.append(next_idx)
            candidates = candidates[adjacency[next_idx, candidates]]

        descriptor_mean = float(np.mean(descriptor_distances[clique])) if clique else float("inf")
        if len(clique) > len(best) or (len(clique) == len(best) and descriptor_mean < best_descriptor_mean):
            best = clique
            best_descriptor_mean = descriptor_mean

    return np.asarray(best, dtype=np.int64)


def mac_baseline(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    voxel_size: float = 0.05,
    normal_radius: float = 0.1,
    feature_radius: float = 0.25,
    normal_max_nn: int = 30,
    feature_max_nn: int = 100,
    max_correspondences: int = 512,
    compatibility_distance: float = 0.05,
    max_seeds: int = 64,
    refine_iterations: int = 10,
    refine_trim_fraction: float = 0.7,
) -> BaselineResult:
    """FPFH + maximal-clique spatial compatibility registration.

    This is a dependency-light project implementation inspired by MAC-style
    robust correspondence filtering. It uses FPFH nearest-neighbor matches as
    input correspondences and estimates a pose from a greedy maximal clique, so
    results should be reported as `mac_fpfh`, not as official MAC with learned
    descriptors.
    """

    start = time.perf_counter()
    src_corr, ref_corr, desc_dist = _fpfh_correspondences(
        src_xyz,
        ref_xyz,
        voxel_size=voxel_size,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        normal_max_nn=normal_max_nn,
        feature_max_nn=feature_max_nn,
        max_correspondences=max_correspondences,
    )
    adjacency, _ = _spatial_compatibility(src_corr, ref_corr, compatibility_distance)
    clique = _greedy_largest_clique(adjacency, desc_dist, max_seeds=max_seeds)
    fallback = bool(clique.shape[0] < 3)
    if fallback:
        clique = np.argsort(desc_dist)[:3]
    transform = weighted_kabsch(src_corr[clique], ref_corr[clique])

    if refine_iterations > 0:
        result = point_to_point_icp(
            src_xyz,
            ref_xyz,
            iterations=refine_iterations,
            trim_fraction=refine_trim_fraction,
            init_transform=transform,
        )
        transform = result.transform
        refine_meta = result.meta
    else:
        refine_meta = {}

    meta: Dict[str, float | int | bool | str] = {
        "method": "mac_fpfh",
        "num_correspondences": int(src_corr.shape[0]),
        "clique_size": int(clique.shape[0]),
        "compatibility_distance": float(compatibility_distance),
        "max_seeds": int(max_seeds),
        "fallback": fallback,
        "descriptor_distance_mean": float(np.mean(desc_dist)),
    }
    meta.update({f"refine_{key}": value for key, value in refine_meta.items()})
    return BaselineResult(transform=transform, runtime_sec=float(time.perf_counter() - start), meta=meta)


def sc2_pcr_baseline(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    voxel_size: float = 0.05,
    normal_radius: float = 0.1,
    feature_radius: float = 0.25,
    normal_max_nn: int = 30,
    feature_max_nn: int = 100,
    max_correspondences: int = 512,
    compatibility_distance: float = 0.05,
    max_selected_correspondences: int = 96,
    power_iterations: int = 10,
    refine_iterations: int = 10,
    refine_trim_fraction: float = 0.7,
) -> BaselineResult:
    """FPFH + second-order spatial compatibility registration.

    This follows the SC2-PCR idea of using second-order spatial compatibility to
    score correspondences, but keeps the implementation lightweight and tied to
    FPFH correspondences for this benchmark.
    """

    start = time.perf_counter()
    src_corr, ref_corr, desc_dist = _fpfh_correspondences(
        src_xyz,
        ref_xyz,
        voxel_size=voxel_size,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        normal_max_nn=normal_max_nn,
        feature_max_nn=feature_max_nn,
        max_correspondences=max_correspondences,
    )
    _, dist_diff = _spatial_compatibility(src_corr, ref_corr, compatibility_distance)
    first_order = np.clip(1.0 - (dist_diff / float(compatibility_distance)) ** 2, 0.0, 1.0)
    np.fill_diagonal(first_order, 0.0)
    second_order = (first_order @ first_order) * first_order
    scores = np.ones((second_order.shape[0],), dtype=np.float32) / max(1, second_order.shape[0])
    for _ in range(max(1, int(power_iterations))):
        scores = second_order @ scores
        norm = float(np.linalg.norm(scores))
        if norm <= 1e-12:
            break
        scores = (scores / norm).astype(np.float32)
    if not np.isfinite(scores).all() or float(scores.max(initial=0.0)) <= 0.0:
        scores = 1.0 / (desc_dist + 1e-6)

    keep = max(3, min(int(max_selected_correspondences), src_corr.shape[0]))
    selected = np.argsort(-scores)[:keep]
    weights = np.clip(scores[selected], 1e-6, None).astype(np.float32)
    transform = weighted_kabsch(src_corr[selected], ref_corr[selected], weights=weights)

    if refine_iterations > 0:
        result = point_to_point_icp(
            src_xyz,
            ref_xyz,
            iterations=refine_iterations,
            trim_fraction=refine_trim_fraction,
            init_transform=transform,
        )
        transform = result.transform
        refine_meta = result.meta
    else:
        refine_meta = {}

    meta: Dict[str, float | int | bool | str] = {
        "method": "sc2_pcr_fpfh",
        "num_correspondences": int(src_corr.shape[0]),
        "selected_correspondences": int(selected.shape[0]),
        "compatibility_distance": float(compatibility_distance),
        "power_iterations": int(power_iterations),
        "score_mean": float(np.mean(scores)),
        "descriptor_distance_mean": float(np.mean(desc_dist)),
    }
    meta.update({f"refine_{key}": value for key, value in refine_meta.items()})
    return BaselineResult(transform=transform, runtime_sec=float(time.perf_counter() - start), meta=meta)


def kiss_matcher_baseline(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    voxel_size: float = 0.05,
) -> BaselineResult:
    """KISS-Matcher robust global point-cloud registration.

    KISS-Matcher estimates a rigid transform directly from the source and
    reference coordinates. It is exposed as an optional dependency because the
    package ships native extensions.
    """

    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be > 0")

    start = time.perf_counter()
    km = _load_kiss_matcher()
    config = km.KISSMatcherConfig(float(voxel_size))
    matcher = km.KISSMatcher(config)
    result = matcher.estimate(_as_float32(src_xyz)[:, :3], _as_float32(ref_xyz)[:, :3])

    rotation = np.asarray(result.rotation, dtype=np.float32)
    translation = np.asarray(result.translation, dtype=np.float32).reshape(3)
    transform = _as_rigid_transform(np.concatenate([rotation, translation[:, None]], axis=1))
    meta: Dict[str, float | int | bool | str] = {
        "method": "kiss_matcher",
        "voxel_size": float(voxel_size),
    }
    if hasattr(matcher, "get_num_rotation_inliers"):
        meta["num_rotation_inliers"] = int(matcher.get_num_rotation_inliers())
    if hasattr(matcher, "get_num_final_inliers"):
        meta["num_final_inliers"] = int(matcher.get_num_final_inliers())
    return BaselineResult(transform=transform, runtime_sec=float(time.perf_counter() - start), meta=meta)


def _rotation_matrix_from_so3(axis_angle: np.ndarray, max_angle: float) -> np.ndarray:
    """Convert small rotation vector to rotation matrix, with angle clamping."""

    axis_angle = axis_angle.astype(np.float32)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float32)

    angle = min(angle, max_angle)
    axis = axis_angle / angle
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float32,
    )
    return (
        np.eye(3, dtype=np.float32) * c
        + (1.0 - c) * np.outer(axis, axis)
        + s * skew
    )


def _trim_indices(dist2: np.ndarray, trim_fraction: float) -> np.ndarray:
    if trim_fraction >= 1.0:
        return np.arange(dist2.shape[0])
    if not (0.0 < trim_fraction < 1.0):
        raise ValueError("trim_fraction must be in (0, 1]")

    keep = max(3, math.ceil(dist2.shape[0] * trim_fraction))
    keep = min(keep, dist2.shape[0])
    return np.argpartition(dist2, keep - 1)[:keep]


def _point_to_point_step(
    src: PointCloud,
    ref: PointCloud,
    nn_indices: np.ndarray,
    trim_fraction: float,
) -> Transform3x4:
    src_sel = _as_float32(src[nn_indices]) if src.shape[0] != nn_indices.shape[0] else _as_float32(src)
    if src_sel.shape[0] != nn_indices.shape[0]:
        raise ValueError("shape mismatch")
    dst_sel = _as_float32(ref[nn_indices, :3])

    # src_sel here is expected to be transformed source points aligned to matched ref.
    ref_tree = cKDTree(dst_sel[:, :3])
    _, reverse_idx = ref_tree.query(src_sel[:, :3], k=1)
    nn_src = src_sel[reverse_idx, :3]
    nn_dst = dst_sel[:, :3]
    dist2 = np.sum((nn_src - nn_dst) ** 2, axis=1)
    keep_idx = _trim_indices(dist2, trim_fraction)
    return weighted_kabsch(nn_src[keep_idx], nn_dst[keep_idx])


def point_to_point_icp(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    iterations: int = 20,
    tolerance: float = 1e-6,
    trim_fraction: float = 1.0,
    init_transform: Transform3x4 | None = None,
) -> BaselineResult:
    """Point-to-point ICP with optional trimming."""

    src = _as_float32(src_xyz)[:, :3]
    ref = _as_float32(ref_xyz)[:, :3]
    if src.shape[0] < 3 or ref.shape[0] < 3:
        raise ValueError("src and ref require at least 3 points")
    transform = _as_rigid_transform(identity_transform() if init_transform is None else init_transform.copy())

    tree = cKDTree(ref)
    start = time.perf_counter()
    prev_error = float("inf")
    it_used = 0
    mean_errors: List[float] = []
    for it in range(max(1, int(iterations))):
        src_t = apply_transform(src, transform)
        nn_dist2, nn_idx = tree.query(src_t, k=1)
        src_corr = src_t
        ref_corr = ref[nn_idx]
        dist2 = (src_t - ref_corr) ** 2
        dist2 = np.sum(dist2, axis=1)

        keep_idx = _trim_indices(dist2, trim_fraction)
        delta = weighted_kabsch(src_corr[keep_idx], ref_corr[keep_idx])
        transform = compose_transforms(delta, transform)
        error = float(np.mean(dist2[keep_idx]))
        mean_errors.append(error)
        it_used = it + 1
        if abs(prev_error - error) < tolerance * (1.0 + prev_error):
            break
        prev_error = error

    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta={"iterations_used": int(it_used), "final_error": float(mean_errors[-1])},
    )


def point_to_plane_icp(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    ref_normals: PointCloud,
    iterations: int = 20,
    tolerance: float = 1e-6,
    max_angle_deg: float = 10.0,
    max_translation: float = 0.2,
    init_transform: Transform3x4 | None = None,
) -> BaselineResult:
    """Point-to-plane ICP using the normal equations with hard trimming disabled."""

    src = _as_float32(src_xyz)[:, :3]
    ref = _as_float32(ref_xyz)[:, :3]
    normals = _as_float32(ref_normals)[:, :3]
    if src.shape[0] < 3 or ref.shape[0] < 3:
        raise ValueError("src and ref require at least 3 points")
    if normals.shape != ref.shape:
        raise ValueError("ref_normals must match ref shape")
    transform = _as_rigid_transform(identity_transform() if init_transform is None else init_transform.copy())

    tree = cKDTree(ref)
    start = time.perf_counter()
    prev_error = float("inf")
    it_used = 0
    last_error = float("inf")

    max_angle = math.radians(max(0.0, max_angle_deg))
    for it in range(max(1, int(iterations))):
        src_t = apply_transform(src, transform)
        nn_dist2, nn_idx = tree.query(src_t, k=1)
        matched_ref = ref[nn_idx]
        matched_n = normals[nn_idx]
        residual = np.sum((src_t - matched_ref) * matched_n, axis=1)
        A = np.empty((src_t.shape[0], 6), dtype=np.float32)
        A[:, :3] = np.cross(src_t, matched_n, axis=1)
        A[:, 3:] = matched_n
        b = -residual[:, None]

        # Solve normal equations: A x = b.
        # If the system is ill-conditioned, skip update for stability.
        at_a = A.T @ A
        at_b = A.T @ b
        try:
            x = np.linalg.solve(at_a, at_b).ravel()
        except np.linalg.LinAlgError:
            break

        omega = x[:3]
        trans = x[3:]
        if np.linalg.norm(omega) > max_angle:
            omega *= max_angle / (np.linalg.norm(omega) + 1e-12)
        trans_norm = float(np.linalg.norm(trans))
        if trans_norm > max_translation:
            trans *= max_translation / (trans_norm + 1e-12)

        rot = _rotation_matrix_from_so3(omega, max_angle=max_angle)
        delta = np.concatenate([rot, trans[:, None]], axis=1).astype(np.float32)
        transform = compose_transforms(delta, transform)

        it_used = it + 1
        mean_error = float(np.mean(np.abs(residual)))
        last_error = mean_error
        if abs(prev_error - mean_error) < tolerance * (1.0 + prev_error):
            break
        prev_error = mean_error

    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta={"iterations_used": int(it_used), "final_error": float(last_error)},
    )


def trimmed_icp(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    iterations: int = 20,
    trim_fraction: float = 0.7,
    tolerance: float = 1e-6,
    init_transform: Transform3x4 | None = None,
) -> BaselineResult:
    """Trimmed ICP keeps only a fixed portion of closest correspondences every step."""

    src = _as_float32(src_xyz)[:, :3]
    ref = _as_float32(ref_xyz)[:, :3]
    if not (0.0 < trim_fraction <= 1.0):
        raise ValueError("trim_fraction must be in (0, 1]")

    return point_to_point_icp(
        src,
        ref,
        iterations=iterations,
        tolerance=tolerance,
        trim_fraction=trim_fraction,
        init_transform=identity_transform() if init_transform is None else init_transform,
    )


def ransac_icp(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    iterations: int = 500,
    inlier_distance: float = 0.05,
    inlier_ratio_min: float = 0.0,
    random_seed: int = 0,
    refine_iterations: int = 10,
) -> BaselineResult:
    """Simple RANSAC + point-to-point ICP style robust baseline.

    Notes
    -----
    This baseline intentionally implements a dependency-light consensus strategy:
    sample 3 correspondences, estimate transform, validate by nearest-neighbor inlier
    count, then refine the best hypothesis with few ICP iterations.
    """

    src = _as_float32(src_xyz)[:, :3]
    ref = _as_float32(ref_xyz)[:, :3]
    if src.shape[0] < 3 or ref.shape[0] < 3:
        raise ValueError("src and ref require at least 3 points")
    if inlier_distance <= 0.0:
        raise ValueError("inlier_distance must be positive")

    start = time.perf_counter()
    tree = cKDTree(ref)
    corr_d2, corr_idx = tree.query(src, k=1)
    if corr_idx.size < 3:
        raise ValueError("Nearest-neighbor correspondences failed")

    rng = random.Random(random_seed)
    best_inlier_ratio = float(-1.0)
    best_transform = identity_transform()
    best_inliers = 0

    corr_dist2 = corr_d2.astype(np.float32)
    correspondences = np.asarray(corr_idx, dtype=np.int64)

    for _ in range(max(1, int(iterations))):
        sample_idx = rng.sample(range(src.shape[0]), k=min(3, src.shape[0]))
        src_sample = src[sample_idx]
        ref_sample = ref[correspondences[sample_idx]]

        if np.linalg.matrix_rank(src_sample - src_sample.mean(axis=0)) < 2:
            continue

        try:
            candidate = weighted_kabsch(src_sample, ref_sample)
        except ValueError:
            continue

        src_t = apply_transform(src, candidate)
        d2, _ = tree.query(src_t, k=1)
        inliers = int(np.sum(d2 <= inlier_distance ** 2))
        ratio = inliers / src.shape[0]
        if ratio > best_inlier_ratio:
            best_inlier_ratio = ratio
            best_transform = candidate
            best_inliers = inliers

    # Reject weak models if ratio is extremely poor; fallback to identity.
    if best_inlier_ratio < inlier_ratio_min:
        best_transform = identity_transform()

    # Light local refinement.
    result = point_to_point_icp(
        src,
        ref,
        iterations=refine_iterations,
        trim_fraction=1.0,
        init_transform=best_transform,
    )

    meta = dict(result.meta)
    meta.update({"inlier_ratio": float(best_inlier_ratio), "inlier_count": int(best_inliers)})
    return BaselineResult(
        transform=result.transform,
        runtime_sec=float(time.perf_counter() - start),
        meta=meta,
    )


def fpfh_ransac_icp(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    voxel_size: float = 0.05,
    normal_radius: float = 0.10,
    feature_radius: float = 0.25,
    normal_max_nn: int = 30,
    feature_max_nn: int = 100,
    max_correspondence_distance: float = 0.075,
    ransac_n: int = 4,
    ransac_max_iterations: int = 100_000,
) -> BaselineResult:
    """FPFH + RANSAC registration using Open3D global registration module.

    This is a classical non-learning global registration baseline for low-overlap
    partial matching. It does not rely on the local ICP objective and is often
    used as a baseline on ModelNet-like data.
    """

    src = _as_float32(src_xyz)[:, :3]
    ref = _as_float32(ref_xyz)[:, :3]
    if src.shape[0] < 3 or ref.shape[0] < 3:
        raise ValueError("src and ref require at least 3 points")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    start = time.perf_counter()
    o3d = _load_open3d()

    src_pcd = _build_o3d_point_cloud(src, o3d).voxel_down_sample(voxel_size)
    ref_pcd = _build_o3d_point_cloud(ref, o3d).voxel_down_sample(voxel_size)
    if len(src_pcd.points) < 3 or len(ref_pcd.points) < 3:
        raise ValueError("Voxel down-sampling removed too many points for FPFH+RANSAC")

    src_fpfh = _estimate_fpfh_features(
        np.asarray(src_pcd.points),
        o3d=o3d,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        normal_max_nn=normal_max_nn,
        feature_max_nn=feature_max_nn,
    )
    ref_fpfh = _estimate_fpfh_features(
        np.asarray(ref_pcd.points),
        o3d=o3d,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        normal_max_nn=normal_max_nn,
        feature_max_nn=feature_max_nn,
    )

    reg_result = None
    try:
        reg_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=src_pcd,
            target=ref_pcd,
            source_feature=src_fpfh,
            target_feature=ref_fpfh,
            mutual_filter=False,
            max_correspondence_distance=max_correspondence_distance,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=ransac_n,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=ransac_max_iterations,
                confidence=0.999,
            ),
        )
    except TypeError:
        reg_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_pcd,
            ref_pcd,
            src_fpfh,
            ref_fpfh,
            False,
            max_correspondence_distance,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_iterations, 0.999),
        )

    meta: Dict[str, float | int | bool | str] = {
        "fallback": False,
        "voxel_size": float(voxel_size),
        "method": "fpfh_ransac",
        "max_correspondence_distance": float(max_correspondence_distance),
        "ransac_n": int(ransac_n),
        "ransac_max_iterations": int(ransac_max_iterations),
    }
    if reg_result is None or getattr(reg_result, "transformation", None) is None:
        transform = identity_transform()
        meta.update(
            {
                "fitness": 0.0,
                "inlier_rmse": float("inf"),
                "fallback": True,
                "failure": "Open3D registration failed" if reg_result is None else "No transform returned",
                "raw_inliers": 0,
            }
        )
    else:
        transform = _convert_open3d_to_row_transform(np.asarray(reg_result.transformation))
        meta.update(
            {
                "fitness": float(reg_result.fitness),
                "inlier_rmse": float(reg_result.inlier_rmse),
                "correspondence_count": int(len(reg_result.correspondence_set)),
            }
        )

    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta=meta,
    )


def fpfh_fgr_icp(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    voxel_size: float = 0.05,
    normal_radius: float = 0.10,
    feature_radius: float = 0.25,
    normal_max_nn: int = 30,
    feature_max_nn: int = 100,
    max_correspondence_distance: float = 0.075,
) -> BaselineResult:
    """FPFH + Fast Global Registration using Open3D."""

    src = _as_float32(src_xyz)[:, :3]
    ref = _as_float32(ref_xyz)[:, :3]
    if src.shape[0] < 3 or ref.shape[0] < 3:
        raise ValueError("src and ref require at least 3 points")

    start = time.perf_counter()
    o3d = _load_open3d()
    src_pcd, ref_pcd = _build_downsampled_o3d_pair(src, ref, o3d, voxel_size)
    src_fpfh = _estimate_fpfh_features(
        np.asarray(src_pcd.points),
        o3d=o3d,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        normal_max_nn=normal_max_nn,
        feature_max_nn=feature_max_nn,
    )
    ref_fpfh = _estimate_fpfh_features(
        np.asarray(ref_pcd.points),
        o3d=o3d,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        normal_max_nn=normal_max_nn,
        feature_max_nn=feature_max_nn,
    )

    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=max_correspondence_distance
    )
    reg_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source=src_pcd,
        target=ref_pcd,
        source_feature=src_fpfh,
        target_feature=ref_fpfh,
        option=option,
    )

    meta: Dict[str, float | int | str | bool] = {
        "method": "fpfh_fgr",
        "voxel_size": float(voxel_size),
        "max_correspondence_distance": float(max_correspondence_distance),
    }
    transform = _convert_open3d_to_row_transform(np.asarray(reg_result.transformation))
    meta.update(
        {
            "fitness": float(reg_result.fitness),
            "inlier_rmse": float(reg_result.inlier_rmse),
            "correspondence_count": int(len(reg_result.correspondence_set)),
            "fallback": False,
        }
    )
    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta=meta,
    )


def gicp_icp(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    max_correspondence_distance: float = 0.075,
    init_transform: Transform3x4 | None = None,
    max_iterations: int = 64,
) -> BaselineResult:
    """Generalized ICP using Open3D."""

    src = _as_float32(src_xyz)[:, :3]
    ref = _as_float32(ref_xyz)[:, :3]
    if src.shape[0] < 3 or ref.shape[0] < 3:
        raise ValueError("src and ref require at least 3 points")
    if max_correspondence_distance <= 0:
        raise ValueError("max_correspondence_distance must be > 0")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")

    start = time.perf_counter()
    o3d = _load_open3d()
    src_pcd = _build_o3d_point_cloud(src, o3d)
    ref_pcd = _build_o3d_point_cloud(ref, o3d)
    init_4x4 = np.eye(4, dtype=np.float64)
    if init_transform is not None:
        init_4x4[:3, :3] = init_transform[:, :3].T
        init_4x4[:3, 3] = init_transform[:, 3]
    estimator_type = getattr(
        o3d.pipelines.registration, "TransformationEstimationForGeneralizedICP", None
    )
    if estimator_type is None:
        raise RuntimeError("Open3D build does not expose generalized ICP estimator")
    reg_result = o3d.pipelines.registration.registration_generalized_icp(
        source=src_pcd,
        target=ref_pcd,
        max_correspondence_distance=max_correspondence_distance,
        init=init_4x4,
        estimation_method=estimator_type(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )
    transform = _convert_open3d_to_row_transform(np.asarray(reg_result.transformation))
    meta: Dict[str, float | int | str | bool] = {
        "method": "gicp",
        "max_correspondence_distance": float(max_correspondence_distance),
        "max_iterations": int(max_iterations),
        "fitness": float(reg_result.fitness),
        "inlier_rmse": float(reg_result.inlier_rmse),
    }
    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta=meta,
    )


def cpd_rigid(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
    w: float = 0.0,
) -> BaselineResult:
    """Coherent Point Drift point-set registration (rigid) using pycpd."""

    src = _as_float32(src_xyz)[:, :3]
    ref = _as_float32(ref_xyz)[:, :3]
    if src.shape[0] < 3 or ref.shape[0] < 3:
        raise ValueError("src and ref require at least 3 points")

    start = time.perf_counter()
    RigidRegistration = _load_pycpd()
    reg = RigidRegistration(
        X=ref,
        Y=src,
        w=w,
        max_iterations=max(1, int(max_iterations)),
        tolerance=max(0.0, float(tolerance)),
        scale=False,
    )
    _, (scale, rot, trans) = reg.register()
    if not np.isfinite(rot).all() or not np.isfinite(trans).all():
        raise RuntimeError("CPD produced invalid transform")
    # pycpd uses column-vector convention: x' = x @ R + t.
    transform = np.concatenate([rot.T.astype(np.float32), trans.astype(np.float32)[:, None]], axis=1)
    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta={"method": "cpd", "scale": float(scale), "scale_disabled": True, "w": float(w)},
    )


def teaserpp_baseline(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    noise_bound: float = 0.05,
    voxel_size: float = 0.05,
    normal_radius: float = 0.1,
    feature_radius: float = 0.25,
    normal_max_nn: int = 30,
    feature_max_nn: int = 100,
    max_correspondences: int = 512,
    rotation_max_iterations: int = 100,
) -> BaselineResult:
    """TEASER++ robust registration over FPFH nearest-neighbor correspondences."""

    default_python_path = "external_src/TEASER-plusplus/build/python"
    env_python_path = os.environ.get("TEASERPP_PYTHONPATH")
    if env_python_path:
        for candidate in _relative_path_candidates(env_python_path):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                break
    else:
        for candidate in _relative_path_candidates(default_python_path):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                break

    try:
        try:
            from teaserpp_python import _teaserpp as teaserpp_python  # type: ignore
        except ImportError:
            import teaserpp_python  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "TEASER++ Python binding is not importable. Build it under "
            "`external_src/TEASER-plusplus/build/python` or set TEASERPP_PYTHONPATH "
            "to a repository-relative path."
        ) from exc

    start = time.perf_counter()
    src_corr, ref_corr, desc_dist = _fpfh_correspondences(
        src_xyz,
        ref_xyz,
        voxel_size=voxel_size,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        normal_max_nn=normal_max_nn,
        feature_max_nn=feature_max_nn,
        max_correspondences=max_correspondences,
    )
    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.noise_bound = float(noise_bound)
    params.cbar2 = 1.0
    params.estimate_scaling = False
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = int(rotation_max_iterations)
    params.rotation_cost_threshold = 1e-12
    params.rotation_estimation_algorithm = teaserpp_python.RotationEstimationAlgorithm.GNC_TLS
    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(src_corr.T.astype(np.float64), ref_corr.T.astype(np.float64))
    solution = solver.getSolution()
    rotation = np.asarray(solution.rotation, dtype=np.float32)
    translation = np.asarray(solution.translation, dtype=np.float32).reshape(3)
    transform = _as_rigid_transform(np.concatenate([rotation, translation[:, None]], axis=1))
    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta={
            "method": "teaserpp",
            "num_correspondences": int(src_corr.shape[0]),
            "noise_bound": float(noise_bound),
            "descriptor_distance_mean": float(np.mean(desc_dist)),
        },
    )


def turboreg_baseline(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    voxel_size: float = 0.05,
    normal_radius: float = 0.1,
    feature_radius: float = 0.25,
    normal_max_nn: int = 30,
    feature_max_nn: int = 100,
    max_correspondences: int = 6000,
    max_n: int = 7000,
    tau_length_consis: float = 0.012,
    num_pivot: int = 2000,
    radiu_nms: float = 0.15,
    tau_inlier: float = 0.1,
    metric: str = "IN",
    device: str | None = None,
) -> BaselineResult:
    """TurboReg robust estimator over FPFH nearest-neighbor correspondences."""

    try:
        import torch
        import turboreg_gpu  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "TurboReg Python binding is not importable. Install it from "
            "`external_src/new_baselines/TurboReg/bindings` first."
        ) from exc

    metric = str(metric).upper()
    if metric not in {"IN", "MAE", "MSE"}:
        raise ValueError("TurboReg metric must be one of IN, MAE, or MSE")

    start = time.perf_counter()
    src_corr, ref_corr, desc_dist = _fpfh_correspondences(
        src_xyz,
        ref_xyz,
        voxel_size=voxel_size,
        normal_radius=normal_radius,
        feature_radius=feature_radius,
        normal_max_nn=normal_max_nn,
        feature_max_nn=feature_max_nn,
        max_correspondences=max_correspondences,
    )
    max_n = max(int(max_n), int(src_corr.shape[0]))
    run_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    src_tensor = torch.as_tensor(src_corr, dtype=torch.float32, device=run_device)
    ref_tensor = torch.as_tensor(ref_corr, dtype=torch.float32, device=run_device)

    solver = turboreg_gpu.TurboRegGPU(
        max_n,
        float(tau_length_consis),
        int(num_pivot),
        float(radiu_nms),
        float(tau_inlier),
        metric,
    )
    transform_4x4 = solver.run_reg(src_tensor, ref_tensor).detach().cpu().numpy()
    transform = _convert_open3d_to_row_transform(np.asarray(transform_4x4, dtype=np.float32))
    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta={
            "method": "turboreg",
            "num_correspondences": int(src_corr.shape[0]),
            "max_n": int(max_n),
            "tau_length_consis": float(tau_length_consis),
            "num_pivot": int(num_pivot),
            "radiu_nms": float(radiu_nms),
            "tau_inlier": float(tau_inlier),
            "metric": metric,
            "device": str(run_device),
            "descriptor_distance_mean": float(np.mean(desc_dist)),
        },
    )


def super4pcs_baseline(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    binary_path: str | None = None,
    overlap: float = 0.4,
    delta: float = 0.05,
    n_points: int = 300,
    max_time_seconds: int = 10,
    timeout_seconds: int = 20,
) -> BaselineResult:
    """Super4PCS CLI wrapper using temporary OBJ vertex files."""

    binary = _resolve_existing_relative_path(
        binary_path,
        "SUPER4PCS_BINARY",
        "external_src/Super4PCS/build/demos/Super4PCS/Super4PCS",
    )
    src_norm, ref_norm, src_center, ref_center, scale = _normalize_pair_for_external_solver(src_xyz, ref_xyz)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="super4pcs_") as tmp_name:
        tmp_dir = Path(tmp_name)
        ref_path = tmp_dir / "ref.obj"
        src_path = tmp_dir / "src.obj"
        _write_obj_vertices(ref_path, ref_norm)
        _write_obj_vertices(src_path, src_norm)
        command = [
            str(binary),
            "-i",
            str(ref_path),
            str(src_path),
            "-o",
            str(float(overlap)),
            "-d",
            str(float(delta)),
            "-n",
            str(int(n_points)),
            "-t",
            str(int(max_time_seconds)),
        ]
        completed = subprocess.run(
            command,
            cwd=tmp_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Super4PCS failed: {_tail_text(completed.stderr or completed.stdout)}")
    matrix = _parse_first_4x4_matrix(completed.stdout)
    transform = _denormalize_external_transform(
        matrix[:3, :3],
        matrix[:3, 3],
        src_center,
        ref_center,
        scale,
    )
    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta={
            "method": "super4pcs",
            "overlap": float(overlap),
            "delta": float(delta),
            "n_points": int(n_points),
            "stdout_tail": _tail_text(completed.stdout),
        },
    )


def go_icp_baseline(
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    binary_path: str | None = None,
    num_points: int = 300,
    mse_threshold: float = 0.001,
    trim_fraction: float = 0.3,
    dist_trans_size: int = 150,
    dist_trans_expand_factor: float = 2.0,
    timeout_seconds: int = 30,
) -> BaselineResult:
    """Go-ICP CLI wrapper using normalized temporary point-set files."""

    binary = _resolve_existing_relative_path(
        binary_path,
        "GOICP_BINARY",
        "external_src/Go-ICP/build/GoICP",
    )
    src_norm, ref_norm, src_center, ref_center, scale = _normalize_pair_for_external_solver(src_xyz, ref_xyz)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="goicp_") as tmp_name:
        tmp_dir = Path(tmp_name)
        model_path = tmp_dir / "model.txt"
        data_path = tmp_dir / "data.txt"
        config_path = tmp_dir / "config.txt"
        output_path = tmp_dir / "output.txt"
        _write_goicp_points(model_path, ref_norm)
        _write_goicp_points(data_path, src_norm)
        config_path.write_text(
            "\n".join(
                [
                    "# Config file for GO-ICP",
                    f"MSEThresh={float(mse_threshold)}",
                    "rotMinX=-3.1416",
                    "rotMinY=-3.1416",
                    "rotMinZ=-3.1416",
                    "rotWidth=6.2832",
                    "transMinX=-0.5",
                    "transMinY=-0.5",
                    "transMinZ=-0.5",
                    "transWidth=1.0",
                    f"trimFraction={float(trim_fraction)}",
                    f"distTransSize={int(dist_trans_size)}",
                    f"distTransExpandFactor={float(dist_trans_expand_factor)}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        command = [
            str(binary),
            str(model_path),
            str(data_path),
            str(int(num_points)),
            str(config_path),
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            cwd=tmp_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Go-ICP failed: {_tail_text(completed.stderr or completed.stdout)}")
        rotation, translation = _parse_goicp_output(output_path)
    transform = _denormalize_external_transform(rotation, translation, src_center, ref_center, scale)
    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta={
            "method": "goicp",
            "num_points": int(num_points),
            "mse_threshold": float(mse_threshold),
            "trim_fraction": float(trim_fraction),
            "dist_trans_size": int(dist_trans_size),
            "stdout_tail": _tail_text(completed.stdout),
        },
    )


def run_non_learning_baseline(
    name: str,
    src_xyz: PointCloud,
    ref_xyz: PointCloud,
    src_normals: PointCloud | None = None,
    ref_normals: PointCloud | None = None,
    **kwargs: float | int | str | bool | None,
) -> BaselineResult:
    """Dispatch a baseline registration method by name."""

    if name == "identity":
        return BaselineResult(identity_transform(), runtime_sec=0.0, meta={})
    if name in {"icp", "point_to_point_icp", "icp_point_to_point"}:
        return point_to_point_icp(src_xyz, ref_xyz, **kwargs)
    if name in {"point_to_plane_icp", "icp_point_to_plane"}:
        if ref_normals is None:
            raise ValueError("point_to_plane_icp requires ref_normals")
        return point_to_plane_icp(src_xyz, ref_xyz, ref_normals, **kwargs)
    if name in {"trimmed_icp", "icp_trimmed"}:
        return trimmed_icp(src_xyz, ref_xyz, **kwargs)
    if name == "ransac_icp":
        return ransac_icp(src_xyz, ref_xyz, **kwargs)
    if name == "fpfh_ransac":
        return fpfh_ransac_icp(src_xyz, ref_xyz, **kwargs)
    if name in {"fgr", "fpfh_fgr", "fpfh_fgr_icp"}:
        return fpfh_fgr_icp(src_xyz, ref_xyz, **kwargs)
    if name in {"gicp", "generalized_icp"}:
        return gicp_icp(src_xyz, ref_xyz, **kwargs)
    if name in {"cpd", "cpd_rigid", "cpd_rigid_registration"}:
        return cpd_rigid(src_xyz, ref_xyz, **kwargs)
    if name in {"teaserpp", "teaser++", "teaser"}:
        return teaserpp_baseline(src_xyz, ref_xyz, **kwargs)
    if name in {"turboreg", "turbo_reg"}:
        return turboreg_baseline(src_xyz, ref_xyz, **kwargs)
    if name in {"mac", "mac_fpfh", "maximal_clique"}:
        return mac_baseline(src_xyz, ref_xyz, **kwargs)
    if name in {"sc2_pcr", "sc2pcr", "sc2_pcr_fpfh"}:
        return sc2_pcr_baseline(src_xyz, ref_xyz, **kwargs)
    if name in {"kiss_matcher", "kissmatcher", "kiss"}:
        return kiss_matcher_baseline(src_xyz, ref_xyz, **kwargs)
    if name in {"super4pcs", "super4pcs_ransac", "super4pcs_baseline"}:
        return super4pcs_baseline(src_xyz, ref_xyz, **kwargs)
    if name in {"goicp", "go_icp"}:
        return go_icp_baseline(src_xyz, ref_xyz, **kwargs)
    raise ValueError(f"Unknown baseline name: {name}")


def baseline_method_names() -> List[str]:
    return [
        "identity",
        "icp_point_to_point",
        "icp_point_to_plane",
        "icp_trimmed",
        "ransac_icp",
        "fpfh_ransac",
        "fpfh_fgr",
        "fgr",
        "gicp",
        "generalized_icp",
        "cpd",
        "cpd_rigid",
        "teaserpp",
        "turboreg",
        "mac",
        "mac_fpfh",
        "maximal_clique",
        "sc2_pcr",
        "sc2pcr",
        "sc2_pcr_fpfh",
        "kiss_matcher",
        "kissmatcher",
        "kiss",
        "super4pcs",
        "goicp",
    ]


def parse_baseline_methods(spec: str) -> List[str]:
    methods = [item.strip().lower() for item in spec.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one baseline method must be specified")
    unknown = [item for item in methods if item not in baseline_method_names()]
    if unknown:
        raise ValueError(f"Unknown baseline methods: {unknown}")
    return methods
