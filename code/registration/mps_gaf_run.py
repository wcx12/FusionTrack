"""Train, evaluate, or inspect the MPS-GAF registration pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .mps_gaf_data_pipeline import (
        MPSGAFDataConfig,
        get_test_dataset,
        get_train_datasets,
        make_grouped_dataloader,
    )
    from .mps_gaf_registration_core import (
        MPSGAFConfig,
        MPSGAFRegistration,
        compute_rigid_transform,
        transform_se3,
    )
except ImportError:
    from mps_gaf_data_pipeline import (
        MPSGAFDataConfig,
        get_test_dataset,
        get_train_datasets,
        make_grouped_dataloader,
    )
    from mps_gaf_registration_core import MPSGAFConfig, MPSGAFRegistration, compute_rigid_transform, transform_se3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPS-GAF registration runner")
    parser.add_argument("--mode", choices=["inspect", "train", "eval"], required=True)
    parser.add_argument("--dataset_path", required=True, help="Path to modelnet40_ply_hdf5_2048")
    parser.add_argument("--output_dir", default="runs/mps_gaf", help="Directory for checkpoints and metrics")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for eval or training resume")

    parser.add_argument("--noise_type", default="crop", choices=["clean", "jitter", "crop"])
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--partial", type=float, nargs="+", default=[0.7, 0.7])
    parser.add_argument("--rot_mag", type=float, default=45.0)
    parser.add_argument("--trans_mag", type=float, default=0.5)
    parser.add_argument("--num_sources_per_ref", type=int, default=10)
    parser.add_argument("--groups_per_batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--features", nargs="+", default=["ppf", "dxyz", "xyz"], choices=["ppf", "dxyz", "xyz"])
    parser.add_argument("--feat_dim", type=int, default=96)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--num_neighbors", type=int, default=64)
    parser.add_argument("--num_sk_iter", type=int, default=5)
    parser.add_argument("--no_slack", action="store_true")
    parser.add_argument(
        "--fusion_mode",
        choices=["none", "self_only", "target_only", "source_source_only", "full"],
        default="full",
    )
    parser.add_argument("--fusion_start_iter", type=int, default=1)
    parser.add_argument("--fusion_logit_init", type=float, default=-4.0)
    parser.add_argument("--freeze_fusion_gate", action="store_true")
    parser.add_argument("--no_self_corr", action="store_true")
    parser.add_argument("--self_corr_mode", choices=["replace", "residual"], default="replace")
    parser.add_argument("--self_corr_logit_init", type=float, default=-8.0)
    parser.add_argument("--freeze_self_corr_gate", action="store_true")
    parser.add_argument(
        "--svd_weight_mode",
        choices=["row_sum", "rowmax", "entropy", "mutual", "learned", "learned_entropy"],
        default="row_sum",
    )
    parser.add_argument("--svd_weight_power", type=float, default=1.0)
    parser.add_argument("--svd_topk_fraction", type=float, default=1.0)
    parser.add_argument("--learned_svd_logit_init", type=float, default=2.0)

    parser.add_argument("--num_train_iter", type=int, default=2)
    parser.add_argument("--num_eval_iter", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional smoke-test limit for train batches")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Optional smoke-test limit for eval batches")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wt_inliers", type=float, default=1e-2)
    parser.add_argument("--wt_transform", type=float, default=0.0)
    parser.add_argument("--wt_pose", type=float, default=0.0)
    parser.add_argument("--pose_loss_trans_weight", type=float, default=1.0)
    parser.add_argument("--wt_chamfer", type=float, default=0.0)
    parser.add_argument("--wt_chamfer_warmup_start_epoch", type=int, default=1)
    parser.add_argument("--wt_chamfer_warmup_epochs", type=int, default=0)
    parser.add_argument("--train_chamfer_trim_fraction", type=float, default=1.0)
    parser.add_argument("--wt_plane", type=float, default=0.0)
    parser.add_argument("--wt_plane_warmup_start_epoch", type=int, default=1)
    parser.add_argument("--wt_plane_warmup_epochs", type=int, default=0)
    parser.add_argument("--train_plane_trim_fraction", type=float, default=0.7)
    parser.add_argument("--wt_corr", type=float, default=0.0)
    parser.add_argument("--wt_corr_warmup_start_epoch", type=int, default=1)
    parser.add_argument("--wt_corr_warmup_epochs", type=int, default=0)
    parser.add_argument("--corr_overlap_radius", type=float, default=0.05)
    parser.add_argument("--corr_outlier_weight", type=float, default=0.25)
    parser.add_argument("--wt_svd_inlier", type=float, default=0.0)
    parser.add_argument("--wt_svd_inlier_warmup_start_epoch", type=int, default=1)
    parser.add_argument("--wt_svd_inlier_warmup_epochs", type=int, default=0)
    parser.add_argument("--svd_inlier_radius", type=float, default=0.07)
    parser.add_argument("--svd_inlier_pos_weight", type=float, default=1.0)
    parser.add_argument("--lr_plateau_patience", type=int, default=None)
    parser.add_argument("--lr_plateau_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--best_metric", choices=["chamfer", "rotation", "translation", "pose"], default="chamfer")
    parser.add_argument("--pose_trans_weight", type=float, default=50.0)
    parser.add_argument("--icp_refine_steps", type=int, default=0)
    parser.add_argument("--icp_trim_fraction", type=float, default=0.7)
    parser.add_argument("--icp_mode", choices=["point", "plane"], default="point")
    parser.add_argument("--icp_damping", type=float, default=1e-4)
    parser.add_argument("--icp_max_angle_deg", type=float, default=10.0)
    parser.add_argument("--icp_max_translation", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_category_file", default=None)
    parser.add_argument("--val_category_file", default=None)
    parser.add_argument("--test_category_file", default=None)
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> Tuple[MPSGAFDataConfig, MPSGAFConfig]:
    data_config = MPSGAFDataConfig(
        dataset_path=args.dataset_path,
        num_points=args.num_points,
        noise_type=args.noise_type,
        rot_mag=args.rot_mag,
        trans_mag=args.trans_mag,
        partial=tuple(args.partial),
        num_sources_per_ref=args.num_sources_per_ref,
        train_category_file=args.train_category_file,
        val_category_file=args.val_category_file,
        test_category_file=args.test_category_file,
        seed=args.seed,
    )
    model_config = MPSGAFConfig(
        features=tuple(args.features),
        feat_dim=args.feat_dim,
        radius=args.radius,
        num_neighbors=args.num_neighbors,
        num_sources=args.num_sources_per_ref,
        num_sk_iter=args.num_sk_iter,
        no_slack=args.no_slack,
        fusion_mode=args.fusion_mode,
        fusion_start_iter=args.fusion_start_iter,
        enable_self_corr=not args.no_self_corr,
        self_corr_mode=args.self_corr_mode,
        self_corr_logit_init=args.self_corr_logit_init,
        freeze_self_corr_gate=args.freeze_self_corr_gate,
        fusion_logit_init=args.fusion_logit_init,
        freeze_fusion_gate=args.freeze_fusion_gate,
        svd_weight_mode=args.svd_weight_mode,
        svd_weight_power=args.svd_weight_power,
        svd_topk_fraction=args.svd_topk_fraction,
        learned_svd_logit_init=args.learned_svd_logit_init,
    )
    return data_config, model_config


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


def rotation_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return rotation_error_rad(pred, target) * 180.0 / math.pi


def rotation_error_rad(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = target[:, :3, :3].transpose(1, 2) @ pred[:, :3, :3]
    trace = residual[:, 0, 0] + residual[:, 1, 1] + residual[:, 2, 2]
    return torch.acos(torch.clamp(0.5 * (trace - 1.0), min=-1.0 + 1e-6, max=1.0 - 1e-6))


def chamfer_distance(src: torch.Tensor, ref: torch.Tensor, trim_fraction: float = 1.0) -> torch.Tensor:
    dist = torch.cdist(src, ref, p=2.0) ** 2
    src_to_ref = torch.min(dist, dim=2)[0]
    ref_to_src = torch.min(dist, dim=1)[0]
    if trim_fraction >= 1.0:
        return src_to_ref.mean(dim=1) + ref_to_src.mean(dim=1)
    if trim_fraction <= 0.0:
        raise ValueError("trim_fraction must be in (0, 1]")
    src_keep = max(1, math.ceil(src_to_ref.shape[1] * trim_fraction))
    ref_keep = max(1, math.ceil(ref_to_src.shape[1] * trim_fraction))
    src_trimmed = torch.topk(src_to_ref, src_keep, dim=1, largest=False)[0].mean(dim=1)
    ref_trimmed = torch.topk(ref_to_src, ref_keep, dim=1, largest=False)[0].mean(dim=1)
    return src_trimmed + ref_trimmed


def scheduled_weight(target_weight: float, epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    if target_weight == 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return target_weight
    if epoch < start_epoch:
        return 0.0
    progress = min(1.0, (epoch - start_epoch + 1) / warmup_epochs)
    return target_weight * progress


def selection_metric(metrics: Dict[str, float], best_metric: str, pose_trans_weight: float) -> float:
    if best_metric == "chamfer":
        return metrics["chamfer_distance_mean"]
    if best_metric == "rotation":
        return metrics["rotation_error_deg_mean"]
    if best_metric == "translation":
        return metrics["translation_error_mean"]
    if best_metric == "pose":
        return metrics["rotation_error_deg_mean"] + pose_trans_weight * metrics["translation_error_mean"]
    raise ValueError(f"Unsupported best_metric: {best_metric}")


def compose_transforms(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Compose two source-to-target transforms as delta after base."""

    rotation = delta[:, :3, :3] @ base[:, :3, :3]
    translation = delta[:, :3, :3] @ base[:, :3, 3:4] + delta[:, :3, 3:4]
    return torch.cat((rotation, translation), dim=2)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    batch_size = axis_angle.shape[0]
    device = axis_angle.device
    dtype = axis_angle.dtype
    theta = torch.linalg.norm(axis_angle, dim=1, keepdim=True).clamp_min(1e-12)
    axis = axis_angle / theta
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros(batch_size, device=device, dtype=dtype)
    skew = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=1,
    ).reshape(batch_size, 3, 3)
    eye = torch.eye(3, device=device, dtype=dtype).expand(batch_size, -1, -1)
    theta = theta[:, None]
    return eye + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)


def trim_weights(distances: torch.Tensor, trim_fraction: float) -> torch.Tensor:
    if trim_fraction < 1.0:
        keep = max(3, math.ceil(distances.shape[1] * trim_fraction))
        keep = min(keep, distances.shape[1])
        _, keep_idx = torch.topk(distances, keep, dim=1, largest=False)
        return torch.zeros_like(distances).scatter_(1, keep_idx, 1.0)
    return torch.ones_like(distances)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def point_to_plane_loss(
    pred_src: torch.Tensor,
    ref_xyz: torch.Tensor,
    ref_normals: torch.Tensor,
    trim_fraction: float,
) -> torch.Tensor:
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError("train_plane_trim_fraction must be in (0, 1]")

    distances = torch.cdist(pred_src, ref_xyz, p=2.0) ** 2
    nn_dist, nn_idx = torch.min(distances, dim=2)
    matched_ref = torch.gather(ref_xyz, 1, nn_idx[..., None].expand(-1, -1, 3))
    matched_normals = torch.gather(ref_normals, 1, nn_idx[..., None].expand(-1, -1, 3))
    matched_normals = F.normalize(matched_normals, dim=-1, eps=1e-6)
    residual = torch.sum((pred_src - matched_ref) * matched_normals, dim=-1)
    weights = trim_weights(nn_dist.detach(), trim_fraction)
    return (F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none") * weights).sum() / (
        weights.sum().clamp_min(1.0)
    )


def correspondence_supervision_loss(
    perm: torch.Tensor,
    gt_src: torch.Tensor,
    ref_xyz: torch.Tensor,
    overlap_radius: float,
    outlier_weight: float,
) -> torch.Tensor:
    if overlap_radius <= 0:
        raise ValueError("corr_overlap_radius must be positive")
    if outlier_weight < 0:
        raise ValueError("corr_outlier_weight must be non-negative")

    with torch.no_grad():
        gt_dist = torch.cdist(gt_src, ref_xyz, p=2.0) ** 2
        nn_dist, nn_idx = torch.min(gt_dist, dim=2)
        overlap = nn_dist <= overlap_radius**2
        non_overlap = ~overlap

    matched_prob = torch.gather(perm, 2, nn_idx[..., None]).squeeze(-1)
    match_nll = -torch.log(matched_prob.clamp_min(1e-8))
    match_loss = masked_mean(match_nll, overlap)

    if outlier_weight == 0.0:
        return match_loss

    row_sum = torch.sum(perm, dim=2).clamp(max=1.0 - 1e-8)
    outlier_nll = -torch.log((1.0 - row_sum).clamp_min(1e-8))
    outlier_loss = masked_mean(outlier_nll, non_overlap)
    return match_loss + outlier_weight * outlier_loss


def svd_inlier_supervision_loss(
    logits: torch.Tensor,
    gt_src: torch.Tensor,
    ref_xyz: torch.Tensor,
    inlier_radius: float,
    pos_weight: float,
) -> torch.Tensor:
    if inlier_radius <= 0:
        raise ValueError("svd_inlier_radius must be positive")
    with torch.no_grad():
        gt_dist = torch.cdist(gt_src, ref_xyz, p=2.0) ** 2
        labels = (torch.min(gt_dist, dim=2)[0] <= inlier_radius**2).to(logits.dtype)
    weights = torch.ones_like(labels)
    if pos_weight != 1.0:
        weights = torch.where(labels > 0.5, weights * pos_weight, weights)
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weights)


def point_to_plane_delta(
    src_t: torch.Tensor,
    matched_ref: torch.Tensor,
    matched_normals: torch.Tensor,
    weights: torch.Tensor,
    damping: float,
    max_angle_deg: float,
    max_translation: float,
) -> torch.Tensor:
    normals = F.normalize(matched_normals, dim=-1, eps=1e-6)
    cross = torch.cross(src_t, normals, dim=-1)
    system = torch.cat((cross, normals), dim=-1)
    residual = torch.sum(normals * (src_t - matched_ref), dim=-1, keepdim=True)
    weights = weights[..., None]
    lhs = system.transpose(1, 2) @ (system * weights)
    rhs = system.transpose(1, 2) @ (-residual * weights)
    eye = torch.eye(6, device=src_t.device, dtype=src_t.dtype).expand(src_t.shape[0], -1, -1)
    update = torch.linalg.solve(lhs + damping * eye, rhs).squeeze(-1)

    omega = update[:, :3]
    translation = update[:, 3:6]
    max_angle = math.radians(max_angle_deg)
    angle_norm = torch.linalg.norm(omega, dim=1, keepdim=True).clamp_min(1e-12)
    angle_scale = torch.clamp(max_angle / angle_norm, max=1.0)
    omega = omega * angle_scale
    trans_norm = torch.linalg.norm(translation, dim=1, keepdim=True).clamp_min(1e-12)
    trans_scale = torch.clamp(max_translation / trans_norm, max=1.0)
    translation = translation * trans_scale

    rotation = axis_angle_to_matrix(omega)
    return torch.cat((rotation, translation[:, :, None]), dim=2)


def icp_refine_transform(
    transform: torch.Tensor,
    src_xyz: torch.Tensor,
    ref_xyz: torch.Tensor,
    steps: int,
    trim_fraction: float,
    mode: str = "point",
    ref_normals: torch.Tensor | None = None,
    damping: float = 1e-4,
    max_angle_deg: float = 10.0,
    max_translation: float = 0.2,
) -> torch.Tensor:
    if steps <= 0:
        return transform
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError("icp_trim_fraction must be in (0, 1]")

    refined = transform
    for _ in range(steps):
        src_t = transform_se3(refined, src_xyz)
        distances = torch.cdist(src_t, ref_xyz, p=2.0) ** 2
        nn_dist, nn_idx = torch.min(distances, dim=2)
        matched_ref = torch.gather(ref_xyz, 1, nn_idx[..., None].expand(-1, -1, 3))
        weights = trim_weights(nn_dist, trim_fraction)

        if mode == "point":
            delta = compute_rigid_transform(src_t, matched_ref, weights=weights)
        elif mode == "plane":
            if ref_normals is None:
                raise ValueError("ref_normals is required when icp_mode='plane'")
            matched_normals = torch.gather(ref_normals, 1, nn_idx[..., None].expand(-1, -1, 3))
            delta = point_to_plane_delta(
                src_t,
                matched_ref,
                matched_normals,
                weights=weights,
                damping=damping,
                max_angle_deg=max_angle_deg,
                max_translation=max_translation,
            )
        else:
            raise ValueError(f"Unsupported icp_mode: {mode}")
        refined = compose_transforms(delta, refined)
    return refined


def compute_losses(
    batch: Dict,
    pred_transforms: List[torch.Tensor],
    endpoints: Dict,
    wt_inliers: float,
    wt_transform: float = 0.0,
    wt_pose: float = 0.0,
    pose_loss_trans_weight: float = 1.0,
    wt_chamfer: float = 0.0,
    train_chamfer_trim_fraction: float = 1.0,
    wt_plane: float = 0.0,
    train_plane_trim_fraction: float = 0.7,
    wt_corr: float = 0.0,
    corr_overlap_radius: float = 0.05,
    corr_outlier_weight: float = 0.25,
    wt_svd_inlier: float = 0.0,
    svd_inlier_radius: float = 0.07,
    svd_inlier_pos_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    gt_src = transform_se3(batch["transform_gt"], batch["points_src"][..., :3])
    ref_xyz = batch["points_ref"][..., :3]
    ref_normals = batch["points_ref"][..., 3:6]
    criterion = nn.L1Loss()
    losses: Dict[str, torch.Tensor] = {}

    total = 0.0
    discount_factor = 0.5
    num_iter = len(pred_transforms)
    for idx, pred_transform in enumerate(pred_transforms):
        pred_src = transform_se3(pred_transform, batch["points_src"][..., :3])
        reg_loss = criterion(pred_src, gt_src)

        perm = endpoints["perm_matrices"][idx]
        ref_outlier = 1.0 - torch.sum(perm, dim=1)
        src_outlier = 1.0 - torch.sum(perm, dim=2)
        inlier_loss = wt_inliers * (ref_outlier.mean() + src_outlier.mean())

        transform_loss = pred_transform.new_tensor(0.0)
        if wt_transform:
            rot_loss = F.mse_loss(pred_transform[:, :3, :3], batch["transform_gt"][:, :3, :3])
            trans_loss = F.l1_loss(pred_transform[:, :3, 3], batch["transform_gt"][:, :3, 3])
            transform_loss = rot_loss + trans_loss

        pose_loss = pred_transform.new_tensor(0.0)
        if wt_pose:
            pose_rot = rotation_error_rad(pred_transform, batch["transform_gt"]).mean()
            pose_trans = torch.norm(
                pred_transform[:, :3, 3] - batch["transform_gt"][:, :3, 3],
                dim=1,
            ).mean()
            pose_loss = pose_rot + pose_loss_trans_weight * pose_trans

        chamfer_loss = pred_transform.new_tensor(0.0)
        if wt_chamfer:
            chamfer_loss = chamfer_distance(
                pred_src,
                ref_xyz,
                trim_fraction=train_chamfer_trim_fraction,
            ).mean()

        plane_loss = pred_transform.new_tensor(0.0)
        if wt_plane:
            plane_loss = point_to_plane_loss(
                pred_src,
                ref_xyz,
                ref_normals,
                trim_fraction=train_plane_trim_fraction,
            )

        corr_loss = pred_transform.new_tensor(0.0)
        if wt_corr:
            corr_loss = correspondence_supervision_loss(
                perm,
                gt_src.detach(),
                ref_xyz.detach(),
                overlap_radius=corr_overlap_radius,
                outlier_weight=corr_outlier_weight,
            )

        svd_inlier_loss = pred_transform.new_tensor(0.0)
        svd_logits = endpoints.get("svd_logits", [None] * num_iter)[idx]
        if wt_svd_inlier and svd_logits is not None:
            svd_inlier_loss = svd_inlier_supervision_loss(
                svd_logits,
                gt_src.detach(),
                ref_xyz.detach(),
                inlier_radius=svd_inlier_radius,
                pos_weight=svd_inlier_pos_weight,
            )

        iter_loss = (
            reg_loss
            + inlier_loss
            + wt_transform * transform_loss
            + wt_pose * pose_loss
            + wt_chamfer * chamfer_loss
            + wt_plane * plane_loss
            + wt_corr * corr_loss
            + wt_svd_inlier * svd_inlier_loss
        )
        weight = discount_factor ** (num_iter - idx - 1)
        total = total + weight * iter_loss
        losses[f"reg_{idx}"] = reg_loss.detach()
        losses[f"inlier_{idx}"] = inlier_loss.detach()
        losses[f"transform_{idx}"] = transform_loss.detach()
        losses[f"pose_{idx}"] = pose_loss.detach()
        losses[f"chamfer_{idx}"] = chamfer_loss.detach()
        losses[f"plane_{idx}"] = plane_loss.detach()
        losses[f"corr_{idx}"] = corr_loss.detach()
        losses[f"svd_inlier_{idx}"] = svd_inlier_loss.detach()

    losses["total"] = total
    return losses


@torch.no_grad()
def evaluate_model(
    model: MPSGAFRegistration,
    loader: Iterable[Dict],
    device: torch.device,
    num_iter: int,
    max_batches: int | None = None,
    icp_refine_steps: int = 0,
    icp_trim_fraction: float = 0.7,
    icp_mode: str = "point",
    icp_damping: float = 1e-4,
    icp_max_angle_deg: float = 10.0,
    icp_max_translation: float = 0.2,
) -> Dict[str, float]:
    model.eval()
    all_rot = []
    all_trans = []
    all_chamfer = []

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_to_device(batch, device)
        pred_transforms, _ = model(batch, num_iter=num_iter)
        pred = pred_transforms[-1]
        pred = icp_refine_transform(
            pred,
            batch["points_src"][..., :3],
            batch["points_ref"][..., :3],
            steps=icp_refine_steps,
            trim_fraction=icp_trim_fraction,
            mode=icp_mode,
            ref_normals=batch["points_ref"][..., 3:6],
            damping=icp_damping,
            max_angle_deg=icp_max_angle_deg,
            max_translation=icp_max_translation,
        )
        gt = batch["transform_gt"]

        transformed_src = transform_se3(pred, batch["points_src"][..., :3])
        all_rot.append(rotation_error_deg(pred, gt).detach().cpu())
        all_trans.append(torch.norm(pred[:, :3, 3] - gt[:, :3, 3], dim=1).detach().cpu())
        all_chamfer.append(chamfer_distance(transformed_src, batch["points_ref"][..., :3]).detach().cpu())
        if max_batches is not None and batch_idx >= max_batches:
            break

    if not all_rot:
        raise RuntimeError("Evaluation loader produced no batches")

    rot = torch.cat(all_rot)
    trans = torch.cat(all_trans)
    chamfer = torch.cat(all_chamfer)
    return {
        "rotation_error_deg_mean": float(rot.mean()),
        "rotation_error_deg_rmse": float(torch.sqrt(torch.mean(rot**2))),
        "translation_error_mean": float(trans.mean()),
        "translation_error_rmse": float(torch.sqrt(torch.mean(trans**2))),
        "chamfer_distance_mean": float(chamfer.mean()),
        "num_pairs": int(rot.numel()),
    }


def save_checkpoint(
    output_dir: Path,
    model: MPSGAFRegistration,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    filename: str = "mps_gaf_latest.pt",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str,
    model: MPSGAFRegistration,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    if optimizer is not None and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except ValueError:
            pass
    return int(checkpoint.get("epoch", 0))


def inspect_pipeline(args: argparse.Namespace) -> None:
    data_config, _ = build_configs(args)
    dataset = get_test_dataset(data_config)
    loader = make_grouped_dataloader(
        dataset,
        groups_per_batch=args.groups_per_batch,
        shuffle_groups=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))
    expected_batch = args.groups_per_batch * args.num_sources_per_ref

    if batch["points_src"].shape[0] != expected_batch:
        raise RuntimeError(f"Unexpected batch size: {batch['points_src'].shape[0]} != {expected_batch}")
    if batch["points_src"].shape[-1] != 6 or batch["points_ref"].shape[-1] != 6:
        raise RuntimeError("points_src and points_ref must contain xyz plus normals")

    group_ref = batch["group_ref_idx"].reshape(args.groups_per_batch, args.num_sources_per_ref)
    if not torch.all(group_ref == group_ref[:, :1]):
        raise RuntimeError("A grouped batch mixes different references inside one source group")

    print(
        json.dumps(
            {
                "status": "ok",
                "batch_size": int(batch["points_src"].shape[0]),
                "points_src_shape": list(batch["points_src"].shape),
                "points_ref_shape": list(batch["points_ref"].shape),
                "num_groups": int(args.groups_per_batch),
                "num_sources_per_ref": int(args.num_sources_per_ref),
            },
            indent=2,
        )
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    data_config, model_config = build_configs(args)

    train_dataset, val_dataset = get_train_datasets(data_config)
    train_loader = make_grouped_dataloader(
        train_dataset,
        groups_per_batch=args.groups_per_batch,
        shuffle_groups=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = make_grouped_dataloader(
        val_dataset,
        groups_per_batch=args.groups_per_batch,
        shuffle_groups=False,
        num_workers=args.num_workers,
    )

    model = MPSGAFRegistration(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_plateau_patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            min_lr=args.min_lr,
        )
    start_epoch = 0
    best_value = float("inf")
    best_metrics: Dict[str, float] | None = None
    epochs_since_best = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(args.checkpoint, model, optimizer, map_location=device)

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        steps_run = 0
        for step, batch in enumerate(train_loader, start=1):
            steps_run = step
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred_transforms, endpoints = model(batch, num_iter=args.num_train_iter)
            wt_chamfer_current = scheduled_weight(
                args.wt_chamfer,
                epoch + 1,
                args.wt_chamfer_warmup_start_epoch,
                args.wt_chamfer_warmup_epochs,
            )
            wt_plane_current = scheduled_weight(
                args.wt_plane,
                epoch + 1,
                args.wt_plane_warmup_start_epoch,
                args.wt_plane_warmup_epochs,
            )
            wt_corr_current = scheduled_weight(
                args.wt_corr,
                epoch + 1,
                args.wt_corr_warmup_start_epoch,
                args.wt_corr_warmup_epochs,
            )
            wt_svd_inlier_current = scheduled_weight(
                args.wt_svd_inlier,
                epoch + 1,
                args.wt_svd_inlier_warmup_start_epoch,
                args.wt_svd_inlier_warmup_epochs,
            )
            losses = compute_losses(
                batch,
                pred_transforms,
                endpoints,
                wt_inliers=args.wt_inliers,
                wt_transform=args.wt_transform,
                wt_pose=args.wt_pose,
                pose_loss_trans_weight=args.pose_loss_trans_weight,
                wt_chamfer=wt_chamfer_current,
                train_chamfer_trim_fraction=args.train_chamfer_trim_fraction,
                wt_plane=wt_plane_current,
                train_plane_trim_fraction=args.train_plane_trim_fraction,
                wt_corr=wt_corr_current,
                corr_overlap_radius=args.corr_overlap_radius,
                corr_outlier_weight=args.corr_outlier_weight,
                wt_svd_inlier=wt_svd_inlier_current,
                svd_inlier_radius=args.svd_inlier_radius,
                svd_inlier_pos_weight=args.svd_inlier_pos_weight,
            )
            losses["total"].backward()
            optimizer.step()
            running_loss += float(losses["total"].detach().cpu())
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
        if steps_run == 0:
            raise RuntimeError("Training loader produced no batches")

        metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            num_iter=args.num_eval_iter,
            max_batches=args.max_eval_batches,
            icp_refine_steps=args.icp_refine_steps,
            icp_trim_fraction=args.icp_trim_fraction,
            icp_mode=args.icp_mode,
            icp_damping=args.icp_damping,
            icp_max_angle_deg=args.icp_max_angle_deg,
            icp_max_translation=args.icp_max_translation,
        )
        current_value = selection_metric(metrics, args.best_metric, args.pose_trans_weight)
        if scheduler is not None:
            scheduler.step(current_value)
        checkpoint_path = save_checkpoint(output_dir, model, optimizer, epoch + 1, args)
        best_checkpoint_path = None
        improved = current_value < best_value - args.early_stop_min_delta
        if improved:
            best_value = current_value
            best_metrics = metrics.copy()
            epochs_since_best = 0
            best_checkpoint_path = save_checkpoint(output_dir, model, optimizer, epoch + 1, args, "mps_gaf_best.pt")
        else:
            epochs_since_best += 1
        summary = {
            "epoch": epoch + 1,
            "train_loss": running_loss / steps_run,
            "validation": metrics,
            "checkpoint": str(checkpoint_path),
            "best_metric": args.best_metric,
            "selection_metric": current_value,
            "best_selection_metric": best_value,
            "best_validation": best_metrics,
            "best_chamfer_distance_mean": best_metrics["chamfer_distance_mean"] if best_metrics is not None else None,
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
            "epochs_since_best": epochs_since_best,
            "lr": optimizer.param_groups[0]["lr"],
            "icp_refine_steps": args.icp_refine_steps,
            "icp_trim_fraction": args.icp_trim_fraction,
            "icp_mode": args.icp_mode,
            "wt_chamfer_current": scheduled_weight(
                args.wt_chamfer,
                epoch + 1,
                args.wt_chamfer_warmup_start_epoch,
                args.wt_chamfer_warmup_epochs,
            ),
            "wt_plane_current": scheduled_weight(
                args.wt_plane,
                epoch + 1,
                args.wt_plane_warmup_start_epoch,
                args.wt_plane_warmup_epochs,
            ),
            "wt_corr_current": scheduled_weight(
                args.wt_corr,
                epoch + 1,
                args.wt_corr_warmup_start_epoch,
                args.wt_corr_warmup_epochs,
            ),
            "wt_svd_inlier_current": scheduled_weight(
                args.wt_svd_inlier,
                epoch + 1,
                args.wt_svd_inlier_warmup_start_epoch,
                args.wt_svd_inlier_warmup_epochs,
            ),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "last_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        if args.early_stop_patience is not None and epochs_since_best >= args.early_stop_patience:
            print(
                json.dumps(
                    {
                        "status": "early_stopped",
                        "epoch": epoch + 1,
                        "best_metric": args.best_metric,
                        "best_selection_metric": best_value,
                        "epochs_since_best": epochs_since_best,
                    },
                    indent=2,
                )
            )
            break


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for eval mode")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data_config, model_config = build_configs(args)
    test_dataset = get_test_dataset(data_config)
    test_loader = make_grouped_dataloader(
        test_dataset,
        groups_per_batch=args.groups_per_batch,
        shuffle_groups=False,
        num_workers=args.num_workers,
    )

    model = MPSGAFRegistration(model_config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        num_iter=args.num_eval_iter,
        max_batches=args.max_eval_batches,
        icp_refine_steps=args.icp_refine_steps,
        icp_trim_fraction=args.icp_trim_fraction,
        icp_mode=args.icp_mode,
        icp_damping=args.icp_damping,
        icp_max_angle_deg=args.icp_max_angle_deg,
        icp_max_translation=args.icp_max_translation,
    )
    (output_dir / "eval_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "inspect":
        inspect_pipeline(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
