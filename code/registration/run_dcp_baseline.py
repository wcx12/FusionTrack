"""Train/evaluate adapted DCP-family baselines with the MPS-GAF data protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .mps_gaf_data_pipeline import MPSGAFDataConfig, get_test_dataset, get_train_datasets, make_grouped_dataloader
    from .mps_gaf_run import chamfer_distance, rotation_error_deg
except ImportError:
    from mps_gaf_data_pipeline import MPSGAFDataConfig, get_test_dataset, get_train_datasets, make_grouped_dataloader
    from mps_gaf_run import chamfer_distance, rotation_error_deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DCP-family models under the MPS-GAF evaluation schema")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--dataset_name",
        default="modelnet40",
        choices=["modelnet40", "modellonet40", "3dmatch", "3dlomatch", "kitti", "eth"],
    )
    parser.add_argument("--split_root", default=None)
    parser.add_argument("--pair_list", default=None)
    parser.add_argument("--no_estimate_normals", action="store_true")
    parser.add_argument("--output_dir", default="runs/dcp_source2_crop_eval20")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_family", choices=["dcp", "prnet", "idam", "rpmnet", "pointnetlk", "omnet"], default="dcp")
    parser.add_argument("--external_repo", default=None)
    parser.add_argument("--dcp_repo", default="external_src/learned_baselines/DCP")
    parser.add_argument("--prnet_repo", default="external_src/learned_baselines/PRNet")
    parser.add_argument("--idam_repo", default="external_src/learned_baselines/IDAM")
    parser.add_argument("--rpmnet_repo", default="external_src/learned_baselines/RPMNet")
    parser.add_argument("--pointnetlk_repo", default="external_src/learned_baselines/PointNetLK")
    parser.add_argument("--omnet_repo", default="external_src/new_baselines/OMNet_Pytorch")
    parser.add_argument("--dataset_split", default="test", choices=["test", "train"])
    parser.add_argument("--noise_type", default="crop", choices=["clean", "jitter", "crop"])
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--partial", type=float, nargs="+", default=[0.7, 0.7])
    parser.add_argument("--rot_mag", type=float, default=45.0)
    parser.add_argument("--trans_mag", type=float, default=0.5)
    parser.add_argument("--num_sources_per_ref", type=int, default=2)
    parser.add_argument("--train_category_file", default=None)
    parser.add_argument("--val_category_file", default=None)
    parser.add_argument("--test_category_file", default=None)
    parser.add_argument("--groups_per_batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--max_train_steps", type=int, default=10)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--emb_nn", default="pointnet")
    parser.add_argument("--emb_dims", type=int, default=512)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ff_dims", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_iters", type=int, default=3)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--n_keypoints", type=int, default=512)
    parser.add_argument("--n_subsampled_points", type=int, default=512)
    parser.add_argument("--cat_sampler", choices=["softmax", "gumbel_softmax"], default="gumbel_softmax")
    parser.add_argument("--temp_factor", type=float, default=100.0)
    parser.add_argument("--feature_alignment_loss", type=float, default=0.1)
    parser.add_argument("--cycle_consistency_loss", type=float, default=0.1)
    parser.add_argument("--features", nargs="+", default=["ppf", "dxyz", "xyz"])
    parser.add_argument("--feat_dim", type=int, default=96)
    parser.add_argument("--radius", type=float, default=0.3)
    parser.add_argument("--num_neighbors", type=int, default=64)
    parser.add_argument("--num_sk_iter", type=int, default=5)
    parser.add_argument("--no_slack", action="store_true")
    parser.add_argument("--wt_pose", type=float, default=1.0)
    parser.add_argument("--wt_chamfer", type=float, default=0.1)
    parser.add_argument("--wt_aux", type=float, default=0.01)
    parser.add_argument("--pose_trans_weight", type=float, default=50.0)
    parser.add_argument("--overlap_dist", type=float, default=0.1)
    parser.add_argument("--loss_alpha1", type=float, default=1.0)
    parser.add_argument("--loss_alpha2", type=float, default=4.0)
    parser.add_argument("--transform_type", default="modelnet_os_prnet_clean")
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--no_checkpoint_model_args", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_relative_paths(args: argparse.Namespace) -> None:
    for key in (
        "dataset_path",
        "output_dir",
        "checkpoint",
        "split_root",
        "pair_list",
        "external_repo",
        "dcp_repo",
        "prnet_repo",
        "idam_repo",
        "rpmnet_repo",
        "pointnetlk_repo",
        "omnet_repo",
        "train_category_file",
        "val_category_file",
        "test_category_file",
    ):
        value = getattr(args, key, None)
        if value is not None and _is_policy_absolute_path(str(value)):
            raise ValueError(f"{key} must be a relative path")


def validate_learned_runner_paths(args: argparse.Namespace) -> None:
    validate_relative_paths(args)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_policy_absolute_path(path_value: str) -> bool:
    return (
        Path(path_value).is_absolute()
        or PurePosixPath(path_value).is_absolute()
        or PureWindowsPath(path_value).is_absolute()
    )


def _resolve_relative_dir(value: str) -> Path:
    if _is_policy_absolute_path(value):
        raise ValueError(f"External repository path must be relative: {value}")
    path = Path(value)
    for candidate in (Path.cwd() / path, _repo_root() / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find relative directory: {value}")


def _patch_open3d_legacy_aliases() -> None:
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return
    geometry = getattr(getattr(o3d, "open3d", o3d), "geometry", o3d.geometry)
    if not hasattr(geometry, "estimate_normals"):
        def estimate_normals(pcd, search_param):  # type: ignore[no-untyped-def]
            return pcd.estimate_normals(search_param)

        geometry.estimate_normals = estimate_normals
    if not hasattr(o3d, "registration") and hasattr(o3d, "pipelines"):
        o3d.registration = o3d.pipelines.registration
    sys.modules.setdefault("open3d.open3d", getattr(o3d, "open3d", o3d))
    sys.modules.setdefault("open3d.open3d.geometry", geometry)


def load_external_model_module(repo: str):
    repo_path = _resolve_relative_dir(repo)
    old_model = sys.modules.pop("model", None)
    old_util = sys.modules.pop("util", None)
    sys.path.insert(0, str(repo_path))
    try:
        return importlib.import_module("model")
    finally:
        if sys.path and sys.path[0] == str(repo_path):
            sys.path.pop(0)
        if old_model is not None:
            sys.modules["model"] = old_model
        else:
            sys.modules.pop("model", None)
        if old_util is not None:
            sys.modules["util"] = old_util
        else:
            sys.modules.pop("util", None)


def load_external_model_class(repo: str, class_name: str):
    module = load_external_model_module(repo)
    return getattr(module, class_name)


def selected_external_repo(args: argparse.Namespace) -> str:
    if args.external_repo:
        return args.external_repo
    if args.model_family == "rpmnet":
        return args.rpmnet_repo
    if args.model_family == "omnet":
        return args.omnet_repo
    if args.model_family == "pointnetlk":
        return args.pointnetlk_repo
    if args.model_family == "idam":
        return args.idam_repo
    if args.model_family == "prnet":
        return args.prnet_repo
    return args.dcp_repo


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    repo = selected_external_repo(args)
    if args.model_family == "omnet":
        repo_path = _resolve_relative_dir(repo)
        old_model = sys.modules.pop("model", None)
        old_common = sys.modules.pop("common", None)
        sys.path.insert(0, str(repo_path))
        try:
            omnet_module = importlib.import_module("model.net")
            omnet_args = SimpleNamespace(
                net_type="omnet",
                titer=args.n_iters,
                overlap_dist=args.overlap_dist,
                loss_type="omnet",
                loss_alpha1=args.loss_alpha1,
                loss_alpha2=args.loss_alpha2,
                transform_type=args.transform_type,
            )
            model = omnet_module.fetch_net(omnet_args)
            model._mps_gaf_omnet_params = omnet_args
            return model
        finally:
            if sys.path and sys.path[0] == str(repo_path):
                sys.path.pop(0)
            if old_model is not None:
                sys.modules["model"] = old_model
            if old_common is not None:
                sys.modules["common"] = old_common

    if args.model_family == "pointnetlk":
        repo_path = _resolve_relative_dir(repo)
        sys.path.insert(0, str(repo_path))
        try:
            from ptlk import pointlk, pointnet

            features = pointnet.PointNet_features(dim_k=args.emb_dims, use_tnet=False)
            return pointlk.PointLK(features, delta=1.0e-2, learn_delta=False)
        finally:
            if sys.path and sys.path[0] == str(repo_path):
                sys.path.pop(0)

    if args.model_family == "rpmnet":
        repo_path = _resolve_relative_dir(repo)
        src_path = repo_path / "src"
        sys.path.insert(0, str(src_path))
        try:
            rpmnet = importlib.import_module("models.rpmnet")
            rpm_args = SimpleNamespace(
                no_slack=args.no_slack,
                num_sk_iter=args.num_sk_iter,
                features=args.features,
                feat_dim=args.feat_dim,
                radius=args.radius,
                num_neighbors=args.num_neighbors,
            )
            return rpmnet.get_model(rpm_args)
        finally:
            if sys.path and sys.path[0] == str(src_path):
                sys.path.pop(0)

    if args.model_family == "prnet":
        PRNet = load_external_model_class(repo, "PRNet")
        exp_name = Path(args.output_dir).name
        checkpoint_dir = Path("checkpoints") / exp_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prnet_args = SimpleNamespace(
            exp_name=exp_name,
            emb_nn=args.emb_nn,
            attention="transformer",
            head="svd",
            n_emb_dims=args.emb_dims,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            n_ff_dims=args.ff_dims,
            dropout=args.dropout,
            n_iters=args.n_iters,
            discount_factor=args.discount_factor,
            n_keypoints=args.n_keypoints,
            n_subsampled_points=args.n_subsampled_points,
            cat_sampler=args.cat_sampler,
            temp_factor=args.temp_factor,
            model_path="",
            feature_alignment_loss=args.feature_alignment_loss,
            cycle_consistency_loss=args.cycle_consistency_loss,
        )
        return PRNet(prnet_args)

    if args.model_family == "idam":
        _patch_open3d_legacy_aliases()
        module = load_external_model_module(repo)
        idam_args = SimpleNamespace(
            emb_dims=args.emb_dims,
            num_iter=args.n_iters,
        )
        emb_name = args.emb_nn.upper()
        if emb_name == "GNN":
            emb_nn = module.GNN(args.emb_dims)
        elif emb_name == "FPFH":
            idam_args.emb_dims = 33
            emb_nn = module.FPFH()
        else:
            raise ValueError("IDAM --emb_nn must be GNN or FPFH")
        return module.IDAM(emb_nn, idam_args)

    DCP = load_external_model_class(repo, "DCP")
    dcp_args = SimpleNamespace(
        emb_nn=args.emb_nn,
        pointer="transformer",
        head="svd",
        emb_dims=args.emb_dims,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        ff_dims=args.ff_dims,
        dropout=args.dropout,
        cycle=False,
    )
    return DCP(dcp_args)


def build_data_config(args: argparse.Namespace) -> MPSGAFDataConfig:
    return MPSGAFDataConfig(
        dataset_path=args.dataset_path,
        dataset_name=getattr(args, "dataset_name", "modelnet40"),
        num_points=args.num_points,
        noise_type=args.noise_type,
        rot_mag=args.rot_mag,
        trans_mag=args.trans_mag,
        partial=tuple(args.partial),
        num_sources_per_ref=args.num_sources_per_ref,
        split_root=getattr(args, "split_root", None),
        pair_list=getattr(args, "pair_list", None),
        estimate_normals=not getattr(args, "no_estimate_normals", False),
        train_category_file=args.train_category_file,
        val_category_file=args.val_category_file,
        test_category_file=args.test_category_file,
        seed=args.seed,
    )


def rotation_matrix_to_quaternion_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    r00 = rotation[:, 0, 0]
    r01 = rotation[:, 0, 1]
    r02 = rotation[:, 0, 2]
    r10 = rotation[:, 1, 0]
    r11 = rotation[:, 1, 1]
    r12 = rotation[:, 1, 2]
    r20 = rotation[:, 2, 0]
    r21 = rotation[:, 2, 1]
    r22 = rotation[:, 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + r00 + r11 + r22, min=1e-8))
    qx = 0.5 * torch.copysign(torch.sqrt(torch.clamp(1.0 + r00 - r11 - r22, min=1e-8)), r21 - r12)
    qy = 0.5 * torch.copysign(torch.sqrt(torch.clamp(1.0 - r00 + r11 - r22, min=1e-8)), r02 - r20)
    qz = 0.5 * torch.copysign(torch.sqrt(torch.clamp(1.0 - r00 - r11 + r22, min=1e-8)), r10 - r01)
    quat = torch.stack((qw, qx, qy, qz), dim=1)
    return F.normalize(quat, dim=1, eps=1e-8)


def compute_omnet_aux_loss(endpoints: Dict, loss_alpha1: float, loss_alpha2: float) -> torch.Tensor:
    losses = []
    weight = None
    for src_pair, ref_pair, pose_pair in zip(
        endpoints["all_src_cls_pair"],
        endpoints["all_ref_cls_pair"],
        endpoints["all_pose_pair"],
    ):
        src_gt, src_pred = src_pair
        ref_gt, ref_pred = ref_pair
        if weight is None:
            weight = torch.tensor([0.7, 0.3], dtype=src_pred.dtype, device=src_pred.device)
        losses.append(F.cross_entropy(src_pred, src_gt.long(), weight=weight))
        losses.append(F.cross_entropy(ref_pred, ref_gt.long(), weight=weight))
        pose_gt, pose_pred = pose_pair
        losses.append(F.l1_loss(pose_pred[:, :4], pose_gt[:, :4]) * loss_alpha1)
        losses.append(F.mse_loss(pose_pred[:, 4:], pose_gt[:, 4:]) * loss_alpha2)
    return torch.stack([loss.float() for loss in losses]).sum()


def make_loader(args: argparse.Namespace, split: str):
    data_config = build_data_config(args)
    if split == "train":
        dataset, _ = get_train_datasets(data_config)
        shuffle = True
    elif split == "val":
        _, dataset = get_train_datasets(data_config)
        shuffle = False
    else:
        dataset = get_test_dataset(data_config)
        shuffle = False
    return dataset, make_grouped_dataloader(
        dataset,
        groups_per_batch=args.groups_per_batch,
        shuffle_groups=shuffle,
        num_workers=args.num_workers,
    )


def predict_transform(
    model: torch.nn.Module,
    batch: Dict,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    src = batch["points_src"][..., :3].to(device).transpose(1, 2).contiguous()
    ref = batch["points_ref"][..., :3].to(device).transpose(1, 2).contiguous()
    if args.model_family == "omnet":
        src_points = batch["points_src"][..., :3].to(device)
        ref_points = batch["points_ref"][..., :3].to(device)
        gt = batch["transform_gt"].to(device)
        pose_gt = torch.cat((rotation_matrix_to_quaternion_wxyz(gt[:, :3, :3]), gt[:, :3, 3]), dim=1)
        omnet_batch = {
            "points_src": src_points,
            "points_ref": ref_points,
            "transform_gt": gt,
            "pose_gt": pose_gt,
        }
        endpoints = model(omnet_batch)
        transform = endpoints["transform_pair"][1].contiguous()
        pred_points = src_points @ transform[:, :3, :3].transpose(1, 2) + transform[:, :3, 3].unsqueeze(1)
        aux_loss = None
        if model.training and args.wt_aux > 0.0:
            aux_loss = compute_omnet_aux_loss(endpoints, args.loss_alpha1, args.loss_alpha2)
        return transform, pred_points, aux_loss

    if args.model_family == "pointnetlk":
        src_points = batch["points_src"][..., :3].to(device)
        ref_points = batch["points_ref"][..., :3].to(device)
        repo_path = _resolve_relative_dir(selected_external_repo(args))
        sys.path.insert(0, str(repo_path))
        try:
            from ptlk import pointlk

            pointlk.PointLK.do_forward(model, ref_points, src_points, maxiter=args.n_iters)
        finally:
            if sys.path and sys.path[0] == str(repo_path):
                sys.path.pop(0)
        transform = model.g[:, :3, :].contiguous()
        pred_points = src_points @ transform[:, :3, :3].transpose(1, 2) + transform[:, :3, 3].unsqueeze(1)
        return transform, pred_points, None

    if args.model_family == "rpmnet":
        rpm_batch = {
            "points_src": batch["points_src"].to(device),
            "points_ref": batch["points_ref"].to(device),
        }
        transforms, _ = model(rpm_batch, args.n_iters)
        transform = transforms[-1]
        src_xyz = rpm_batch["points_src"][..., :3]
        pred_points = src_xyz @ transform[:, :3, :3].transpose(1, 2) + transform[:, :3, 3].unsqueeze(1)
        return transform, pred_points, None

    if args.model_family == "idam":
        if model.training:
            gt = batch["transform_gt"].to(device)
            rotation, translation, aux_loss = model(src, ref, gt[:, :3, :3], gt[:, :3, 3])
        else:
            rotation, translation, aux_loss = model(src, ref)
    elif hasattr(model, "predict"):
        rotation, translation = model.predict(src, ref)
        aux_loss = None
    else:
        rotation, translation, _, _ = model(src, ref)
        aux_loss = None
    transform = torch.cat([rotation, translation.unsqueeze(2)], dim=2)
    pred_points = (rotation @ src + translation.unsqueeze(2)).transpose(1, 2).contiguous()
    return transform, pred_points, aux_loss


def compute_train_loss(
    pred_transform: torch.Tensor,
    pred_points: torch.Tensor,
    batch: Dict,
    args: argparse.Namespace,
    device: torch.device,
    aux_loss: torch.Tensor | None = None,
) -> torch.Tensor:
    gt = batch["transform_gt"].to(device)
    ref = batch["points_ref"][..., :3].to(device)
    pose_loss = F.mse_loss(pred_transform[:, :3, :3], gt[:, :3, :3]) + F.mse_loss(
        pred_transform[:, :3, 3],
        gt[:, :3, 3],
    )
    chamfer = chamfer_distance(pred_points, ref).mean()
    total = args.wt_pose * pose_loss + args.wt_chamfer * chamfer
    if aux_loss is not None:
        total = total + args.wt_aux * aux_loss
    return total


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    rot_values = []
    trans_values = []
    chamfer_values = []
    for batch_idx, batch in enumerate(loader, start=1):
        if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
            break
        pred_transform, pred_points, _ = predict_transform(model, batch, device, args)
        gt = batch["transform_gt"].to(device)
        ref = batch["points_ref"][..., :3].to(device)
        rot_values.append(rotation_error_deg(pred_transform, gt).detach().cpu())
        trans_values.append(torch.linalg.norm(pred_transform[:, :3, 3] - gt[:, :3, 3], dim=1).detach().cpu())
        chamfer_values.append(chamfer_distance(pred_points, ref).detach().cpu())

    if not rot_values:
        raise RuntimeError("Evaluation loader produced no batches")
    rot = torch.cat(rot_values)
    trans = torch.cat(trans_values)
    chamfer = torch.cat(chamfer_values)
    return {
        "rotation_error_deg_mean": float(rot.mean().item()),
        "rotation_error_deg_rmse": float(torch.sqrt((rot * rot).mean()).item()),
        "translation_error_mean": float(trans.mean().item()),
        "translation_error_rmse": float(torch.sqrt((trans * trans).mean()).item()),
        "chamfer_distance_mean": float(chamfer.mean().item()),
        "num_pairs": int(rot.numel()),
    }


def pose_metric(metrics: Dict[str, float], pose_trans_weight: float) -> float:
    return metrics["rotation_error_deg_mean"] + pose_trans_weight * metrics["translation_error_mean"]


def schema_method_key(args: argparse.Namespace) -> str:
    if args.model_family == "dcp":
        return "dcp"
    return args.model_family


CHECKPOINT_MODEL_ARG_KEYS = (
    "model_family",
    "emb_nn",
    "emb_dims",
    "n_blocks",
    "n_heads",
    "ff_dims",
    "dropout",
    "n_iters",
    "discount_factor",
    "n_keypoints",
    "n_subsampled_points",
    "cat_sampler",
    "temp_factor",
    "feature_alignment_loss",
    "cycle_consistency_loss",
    "features",
    "feat_dim",
    "radius",
    "num_neighbors",
    "num_sk_iter",
    "no_slack",
    "overlap_dist",
    "loss_alpha1",
    "loss_alpha2",
    "transform_type",
)


def apply_checkpoint_model_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.checkpoint or args.no_checkpoint_model_args:
        return args
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    stored_args = checkpoint.get("args", {})
    for key in CHECKPOINT_MODEL_ARG_KEYS:
        if key in stored_args:
            setattr(args, key, stored_args[key])
    return args


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint.get("epoch", 0))


def load_previous_selection_state(output_dir: Path) -> tuple[float, Dict[str, float] | None, int]:
    summary_path = output_dir / "last_train_summary.json"
    if not summary_path.exists():
        return float("inf"), None, 0
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return float("inf"), None, 0
    best_metric = summary.get("best_selection_metric", float("inf"))
    if not isinstance(best_metric, (int, float)) or not math.isfinite(float(best_metric)):
        return float("inf"), None, 0
    best_metrics = summary.get("best_validation")
    if not isinstance(best_metrics, dict):
        best_metrics = None
    epochs_since_best = summary.get("epochs_since_best", 0)
    if not isinstance(epochs_since_best, int) or epochs_since_best < 0:
        epochs_since_best = 0
    return float(best_metric), best_metrics, epochs_since_best


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args = apply_checkpoint_model_args(args)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, train_loader = make_loader(args, "train")
    _, val_loader = make_loader(args, "val")
    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = load_checkpoint(args.checkpoint, model, optimizer, device) if args.checkpoint else 0
    best_metric = float("inf")
    best_metrics: Dict[str, float] | None = None
    epochs_since_best = 0
    if args.checkpoint:
        best_metric, best_metrics, epochs_since_best = load_previous_selection_state(output_dir)

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        steps = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            if args.max_train_steps is not None and batch_idx > args.max_train_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            pred_transform, pred_points, aux_loss = predict_transform(model, batch, device, args)
            loss = compute_train_loss(pred_transform, pred_points, batch, args, device, aux_loss=aux_loss)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())
            steps += 1
        if steps == 0:
            raise RuntimeError("Training loader produced no batches")

        metrics = evaluate_model(model, val_loader, args, device)
        current_metric = pose_metric(metrics, args.pose_trans_weight)
        checkpoint_prefix = args.model_family
        latest_path = output_dir / f"{checkpoint_prefix}_latest.pt"
        save_checkpoint(latest_path, model, optimizer, epoch + 1, args)
        improved = current_metric < best_metric - args.early_stop_min_delta
        if improved:
            best_metric = current_metric
            best_metrics = metrics.copy()
            epochs_since_best = 0
            save_checkpoint(output_dir / f"{checkpoint_prefix}_best.pt", model, optimizer, epoch + 1, args)
        else:
            epochs_since_best += 1
        summary = {
            "epoch": epoch + 1,
            "train_loss": running_loss / steps,
            "validation": metrics,
            "best_validation": best_metrics,
            "best_selection_metric": best_metric,
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(output_dir / f"{checkpoint_prefix}_best.pt") if best_metrics is not None else None,
            "epochs_since_best": epochs_since_best,
        }
        (output_dir / "last_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if args.early_stop_patience > 0 and epochs_since_best >= args.early_stop_patience:
            break


@torch.no_grad()
def eval_checkpoint(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in eval mode")
    args = apply_checkpoint_model_args(args)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, loader = make_loader(args, args.dataset_split)
    model = build_model(args).to(device)
    load_checkpoint(args.checkpoint, model, None, device)
    metrics = evaluate_model(model, loader, args, device)
    comparison = metrics.copy()
    comparison["pose_metric"] = pose_metric(metrics, args.pose_trans_weight)
    (output_dir / "eval_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "comparison_schema_summary.json").write_text(
        json.dumps({schema_method_key(args): comparison}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    validate_relative_paths(args)
    if args.mode == "train":
        train(args)
    else:
        eval_checkpoint(args)


if __name__ == "__main__":
    main()
