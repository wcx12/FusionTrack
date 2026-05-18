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

try:
    from .mps_gaf_data_pipeline import (
        MPSGAFDataConfig,
        get_test_dataset,
        get_train_datasets,
        make_grouped_dataloader,
    )
    from .mps_gaf_registration_core import MPSGAFConfig, MPSGAFRegistration, transform_se3
except ImportError:
    from mps_gaf_data_pipeline import (
        MPSGAFDataConfig,
        get_test_dataset,
        get_train_datasets,
        make_grouped_dataloader,
    )
    from mps_gaf_registration_core import MPSGAFConfig, MPSGAFRegistration, transform_se3


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

    parser.add_argument("--num_train_iter", type=int, default=1)
    parser.add_argument("--num_eval_iter", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wt_inliers", type=float, default=1e-2)
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
    )
    model_config = MPSGAFConfig(
        features=tuple(args.features),
        feat_dim=args.feat_dim,
        radius=args.radius,
        num_neighbors=args.num_neighbors,
        num_sources=args.num_sources_per_ref,
        num_sk_iter=args.num_sk_iter,
        no_slack=args.no_slack,
    )
    return data_config, model_config


def move_to_device(batch: Dict, device: torch.device) -> Dict:
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


def rotation_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = target[:, :3, :3].transpose(1, 2) @ pred[:, :3, :3]
    trace = residual[:, 0, 0] + residual[:, 1, 1] + residual[:, 2, 2]
    return torch.acos(torch.clamp(0.5 * (trace - 1.0), min=-1.0, max=1.0)) * 180.0 / math.pi


def chamfer_distance(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(src, ref, p=2.0) ** 2
    return torch.min(dist, dim=2)[0].mean(dim=1) + torch.min(dist, dim=1)[0].mean(dim=1)


def compute_losses(
    batch: Dict,
    pred_transforms: List[torch.Tensor],
    endpoints: Dict,
    wt_inliers: float,
) -> Dict[str, torch.Tensor]:
    gt_src = transform_se3(batch["transform_gt"], batch["points_src"][..., :3])
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

        iter_loss = reg_loss + inlier_loss
        weight = discount_factor ** (num_iter - idx - 1)
        total = total + weight * iter_loss
        losses[f"reg_{idx}"] = reg_loss.detach()
        losses[f"inlier_{idx}"] = inlier_loss.detach()

    losses["total"] = total
    return losses


@torch.no_grad()
def evaluate_model(
    model: MPSGAFRegistration,
    loader: Iterable[Dict],
    device: torch.device,
    num_iter: int,
) -> Dict[str, float]:
    model.eval()
    all_rot = []
    all_trans = []
    all_chamfer = []

    for batch in loader:
        batch = move_to_device(batch, device)
        pred_transforms, _ = model(batch, num_iter=num_iter)
        pred = pred_transforms[-1]
        gt = batch["transform_gt"]

        transformed_src = transform_se3(pred, batch["points_src"][..., :3])
        all_rot.append(rotation_error_deg(pred, gt).detach().cpu())
        all_trans.append(torch.norm(pred[:, :3, 3] - gt[:, :3, 3], dim=1).detach().cpu())
        all_chamfer.append(chamfer_distance(transformed_src, batch["points_ref"][..., :3]).detach().cpu())

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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "mps_gaf_latest.pt"
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
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
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
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(args.checkpoint, model, optimizer, map_location=device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred_transforms, endpoints = model(batch, num_iter=args.num_train_iter)
            losses = compute_losses(batch, pred_transforms, endpoints, wt_inliers=args.wt_inliers)
            losses["total"].backward()
            optimizer.step()
            running_loss += float(losses["total"].detach().cpu())

        metrics = evaluate_model(model, val_loader, device=device, num_iter=args.num_eval_iter)
        checkpoint_path = save_checkpoint(output_dir, model, optimizer, epoch + 1, args)
        summary = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader)),
            "validation": metrics,
            "checkpoint": str(checkpoint_path),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "last_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


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
    metrics = evaluate_model(model, test_loader, device=device, num_iter=args.num_eval_iter)
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
