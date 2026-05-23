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
- TEASER++ / Super4PCS / Go-ICP (optional external optional dependencies)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
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
    return np.concatenate([rot.T, trans[:, None]], axis=1).astype(np.float32)


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
    )
    _, (scale, rot, trans) = reg.register()
    if not np.isfinite(rot).all() or not np.isfinite(trans).all():
        raise RuntimeError("CPD produced invalid transform")
    # pycpd uses column-vector convention: x' = s * x @ R + t
    transform = np.concatenate([scale * rot.T.astype(np.float32), trans.astype(np.float32)[:, None]], axis=1)
    return BaselineResult(
        transform=transform,
        runtime_sec=float(time.perf_counter() - start),
        meta={"method": "cpd", "scale": float(scale), "w": float(w)},
    )


def teaserpp_baseline(*args: object, **kwargs: object) -> BaselineResult:
    """Optional TEASER++ placeholder implementation."""

    raise RuntimeError(
        "TEASER++ dependency is not installed. Available Python wrappers may be added later."
    )


def super4pcs_baseline(*args: object, **kwargs: object) -> BaselineResult:
    """Optional Super4PCS placeholder implementation."""

    raise RuntimeError(
        "Super4PCS dependency is not installed. Available Python wrappers may be added later."
    )


def go_icp_baseline(*args: object, **kwargs: object) -> BaselineResult:
    """Optional Go-ICP placeholder implementation."""

    raise RuntimeError(
        "Go-ICP dependency is not installed. Available Python wrappers may be added later."
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
