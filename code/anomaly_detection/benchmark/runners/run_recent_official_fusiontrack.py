from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from training.convergence import write_convergence_artifacts


METHODS = ("catch", "sensitive_hue", "cutaddpaste", "timemixer")


@dataclass
class SequenceSample:
    sample_id: str
    sequence: str
    track_id: str
    values: np.ndarray
    metadata: dict[str, Any]


class SequenceDataset(Dataset):
    def __init__(self, samples: list[SequenceSample], channel_first: bool = False) -> None:
        self.samples = samples
        self.channel_first = bool(channel_first)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        values = self.samples[index].values
        if self.channel_first:
            values = values.T
        return torch.tensor(values, dtype=torch.float32), index


class CapDataset(Dataset):
    def __init__(self, values: np.ndarray, labels: np.ndarray) -> None:
        self.values = values.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.values[index].T, dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run recent official-source TSAD baselines on FusionTrack protocol files."
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--task", required=True, choices=["individual", "group"])
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--val-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--win-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=25000)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    started_at = time.perf_counter()
    args = parse_args(argv)
    _set_seed(args.seed)
    if str(args.official_root) not in sys.path:
        sys.path.insert(0, str(args.official_root))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(args.device).startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    win_size = int(args.win_size or (64 if args.task == "individual" else 16))
    args.win_size = win_size
    train_samples = _load_samples(args.train_jsonl, args.task, win_size)
    val_samples = _load_samples(args.val_jsonl, args.task, win_size)
    if len(train_samples) > int(args.max_train_samples):
        train_samples = random.Random(args.seed).sample(train_samples, int(args.max_train_samples))
    if not train_samples or not val_samples:
        raise ValueError("Need non-empty train and validation samples")
    _standardize(train_samples, val_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.method == "catch":
        history, score_rows, model_manifest = _run_catch(args, train_samples, val_samples, device)
    elif args.method == "sensitive_hue":
        history, score_rows, model_manifest = _run_sensitive_hue(
            args, train_samples, val_samples, device
        )
    elif args.method == "cutaddpaste":
        history, score_rows, model_manifest = _run_cutaddpaste(
            args, train_samples, val_samples, device
        )
    elif args.method == "timemixer":
        history, score_rows, model_manifest = _run_timemixer(
            args, train_samples, val_samples, device
        )
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(args.method)

    convergence = write_convergence_artifacts(
        args.output_dir,
        history,
        requested_epochs=int(args.epochs),
        monitor="val_loss",
        extra={
            "gpu_name": _gpu_name(device),
            "wall_time_seconds": time.perf_counter() - started_at,
        },
    )
    score_jsonl = args.output_dir / f"official_{args.method}_{args.task}_scores.jsonl"
    _write_jsonl(score_jsonl, score_rows)
    score_csv = args.output_dir / f"official_{args.method}_{args.task}_scores.csv"
    _write_score_csv(score_csv, score_rows)

    manifest = {
        "method": args.method,
        "task": args.task,
        "official_root": str(args.official_root),
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "score_jsonl": str(score_jsonl),
        "score_csv": str(score_csv),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "win_size": win_size,
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "e_layers": int(args.e_layers),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "history": history,
        "convergence": convergence,
        "model_manifest": model_manifest,
        "adapter_note": (
            "External runner converts FusionTrack trajectories/group-object windows "
            "into fixed-length multivariate series and imports the official method "
            "modules from --official-root."
        ),
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _run_catch(
    args: argparse.Namespace,
    train_samples: list[SequenceSample],
    val_samples: list[SequenceSample],
    device: torch.device,
) -> tuple[list[dict[str, float | int]], list[dict[str, Any]], dict[str, Any]]:
    _prepare_catch_imports(args.official_root)
    from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
    from ts_benchmark.baselines.catch.utils.fre_rec_loss import (
        frequency_criterion,
        frequency_loss,
    )

    input_dim = int(train_samples[0].values.shape[1])
    config = _catch_config(args, input_dim=input_dim)
    model = CATCHModel(config).to(device)
    rec_criterion = nn.MSELoss()
    point_criterion = nn.MSELoss(reduction="none")
    aux_loss = frequency_loss(config)
    freq_criterion = frequency_criterion(config)
    main_params = [param for name, param in model.named_parameters() if "mask_generator" not in name]
    mask_params = list(model.mask_generator.parameters())
    optimizer = torch.optim.Adam(main_params, lr=float(args.lr))
    optimizer_mask = torch.optim.Adam(mask_params, lr=float(args.lr) * 0.1)
    train_loader = DataLoader(
        SequenceDataset(train_samples),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        SequenceDataset(val_samples),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for step, (batch, _) in enumerate(train_loader, start=1):
            inputs = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            optimizer_mask.zero_grad(set_to_none=True)
            outputs, output_complex, dc_loss = model(inputs)
            rec_loss = rec_criterion(outputs, inputs)
            norm_input = model.revin_layer(inputs, "transform")
            loss = rec_loss + config.dc_lambda * dc_loss + config.auxi_lambda * aux_loss(
                output_complex, norm_input
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % max(1, len(train_loader) // 10) == 0:
                optimizer_mask.step()
            total += float(loss.item()) * int(inputs.shape[0])
            count += int(inputs.shape[0])
        train_loss = total / max(count, 1)
        val_loss = _catch_val_loss(
            model, val_loader, device, rec_criterion, aux_loss, config
        )
        row = {
            "epoch": epoch,
            "loss": train_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    score_rows = _score_reconstruction_method(
        args,
        model,
        val_loader,
        val_samples,
        device,
        lambda inputs, outputs: torch.mean(point_criterion(inputs, outputs), dim=-1)
        + config.score_lambda * torch.mean(freq_criterion(inputs, outputs), dim=-1),
    )
    return history, score_rows, {
        "official_components": [
            "ts_benchmark.baselines.catch.models.CATCH_model.CATCHModel",
            "ts_benchmark.baselines.catch.utils.fre_rec_loss.frequency_loss",
        ],
        "input_dim": input_dim,
        "patch_size": config.patch_size,
        "patch_stride": config.patch_stride,
    }


def _prepare_catch_imports(official_root: Path) -> None:
    """Load CATCH model modules without executing its broad benchmark package imports."""

    package_paths = {
        "ts_benchmark": official_root / "ts_benchmark",
        "ts_benchmark.baselines": official_root / "ts_benchmark" / "baselines",
        "ts_benchmark.baselines.catch": official_root
        / "ts_benchmark"
        / "baselines"
        / "catch",
        "ts_benchmark.baselines.catch.layers": official_root
        / "ts_benchmark"
        / "baselines"
        / "catch"
        / "layers",
        "ts_benchmark.baselines.catch.models": official_root
        / "ts_benchmark"
        / "baselines"
        / "catch"
        / "models",
        "ts_benchmark.baselines.catch.utils": official_root
        / "ts_benchmark"
        / "baselines"
        / "catch"
        / "utils",
    }
    for name in list(package_paths):
        module = types.ModuleType(name)
        module.__path__ = [str(package_paths[name])]  # type: ignore[attr-defined]
        sys.modules[name] = module


@torch.no_grad()
def _catch_val_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rec_criterion: nn.Module,
    aux_loss: nn.Module,
    config: SimpleNamespace,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch, _ in loader:
        inputs = batch.to(device)
        outputs, output_complex, dc_loss = model(inputs)
        rec_loss = rec_criterion(outputs, inputs)
        norm_input = model.revin_layer(inputs, "transform")
        loss = rec_loss + config.dc_lambda * dc_loss + config.auxi_lambda * aux_loss(
            output_complex, norm_input
        )
        total += float(loss.item()) * int(inputs.shape[0])
        count += int(inputs.shape[0])
    return total / max(count, 1)


def _catch_config(args: argparse.Namespace, input_dim: int) -> SimpleNamespace:
    patch_size, patch_stride = _patch_config(int(args.win_size or 64))
    inference_patch_size = min(max(4, patch_size * 2), int(args.win_size or 64))
    return SimpleNamespace(
        lr=float(args.lr),
        Mlr=float(args.lr) * 0.1,
        e_layers=int(args.e_layers),
        n_heads=int(args.n_heads),
        cf_dim=int(args.d_model),
        d_ff=int(args.d_model) * 4,
        d_model=int(args.d_model),
        head_dim=max(8, int(args.d_model) // max(1, int(args.n_heads))),
        individual=0,
        dropout=float(args.dropout),
        head_dropout=float(args.dropout),
        auxi_loss="MAE",
        auxi_type="complex",
        auxi_mode="fft",
        auxi_lambda=0.005,
        score_lambda=0.05,
        regular_lambda=0.5,
        temperature=0.07,
        patch_stride=patch_stride,
        patch_size=patch_size,
        inference_patch_stride=1,
        inference_patch_size=inference_patch_size,
        dc_lambda=0.005,
        module_first=True,
        mask=False,
        seq_len=int(args.win_size or 64),
        pred_len=int(args.win_size or 64),
        c_in=int(input_dim),
        enc_in=int(input_dim),
        dec_in=int(input_dim),
        c_out=int(input_dim),
        affine=0,
        subtract_last=0,
        task_name="anomaly_detection",
    )


def _run_sensitive_hue(
    args: argparse.Namespace,
    train_samples: list[SequenceSample],
    val_samples: list[SequenceSample],
    device: torch.device,
) -> tuple[list[dict[str, float | int]], list[dict[str, Any]], dict[str, Any]]:
    from sensitive_hue.model import SensitiveHUE

    input_dim = int(train_samples[0].values.shape[1])
    model = SensitiveHUE(
        step_num_in=int(args.win_size or 64),
        f_in=input_dim,
        dim_model=int(args.d_model),
        head_num=int(args.n_heads),
        dim_hidden_fc=int(args.d_model) * 4,
        encode_layer_num=int(args.e_layers),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    train_loader = DataLoader(
        SequenceDataset(train_samples),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        SequenceDataset(val_samples),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for batch, _ in train_loader:
            inputs = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            rec, log_var_recip = model(inputs)
            loss = _sensitive_hue_loss(rec, inputs, log_var_recip, with_weight=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * int(inputs.shape[0])
            count += int(inputs.shape[0])
        train_loss = total / max(count, 1)
        val_loss = _sensitive_hue_val_loss(model, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": train_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    score_rows = _score_sensitive_hue(args, model, val_loader, val_samples, device)
    return history, score_rows, {
        "official_components": ["sensitive_hue.model.SensitiveHUE"],
        "input_dim": input_dim,
        "loss": "MTS-NLL-style loss from official sensitive_hue.trainer.Trainer.loss_func",
    }


def _sensitive_hue_loss(
    rec: torch.Tensor,
    inputs: torch.Tensor,
    log_var_recip: torch.Tensor,
    with_weight: bool,
    alpha: float = 0.1,
) -> torch.Tensor:
    rec_loss = nn.functional.mse_loss(rec, inputs, reduction="none")
    sigma_loss = rec_loss * log_var_recip.exp() - log_var_recip
    if not with_weight:
        return sigma_loss.mean()
    var = (-log_var_recip).exp().detach()
    mean_var = var.mean(dim=(0, 1)).clamp_min(1e-6) ** float(alpha)
    return (var * sigma_loss / mean_var).mean()


@torch.no_grad()
def _sensitive_hue_val_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch, _ in loader:
        inputs = batch.to(device)
        rec, log_var_recip = model(inputs)
        loss = _sensitive_hue_loss(rec, inputs, log_var_recip, with_weight=False)
        total += float(loss.item()) * int(inputs.shape[0])
        count += int(inputs.shape[0])
    return total / max(count, 1)


@torch.no_grad()
def _score_sensitive_hue(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    samples: list[SequenceSample],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch, indices in loader:
        inputs = batch.to(device)
        rec, log_var_recip = model(inputs)
        scores = nn.functional.mse_loss(rec, inputs, reduction="none")
        scores = scores * log_var_recip.exp() - log_var_recip
        scores = scores.max(dim=-1).values
        sample_scores = _aggregate_time_scores(scores, float(args.top_fraction)).cpu().tolist()
        for sample_index, score in zip(indices.tolist(), sample_scores):
            rows.append(_score_row(args, samples[int(sample_index)], score))
    return rows


def _run_cutaddpaste(
    args: argparse.Namespace,
    train_samples: list[SequenceSample],
    val_samples: list[SequenceSample],
    device: torch.device,
) -> tuple[list[dict[str, float | int]], list[dict[str, Any]], dict[str, Any]]:
    generate_negative = _load_official_module(
        "cutaddpaste_generate_negative",
        args.official_root / "dataloader" / "generate_negative.py",
    )
    model_module = _load_official_module(
        "cutaddpaste_model",
        args.official_root / "models" / "CutAddPaste" / "network" / "model.py",
    )
    cut_add_paste_outlier = generate_negative.cut_add_paste_outlier
    base_Model = model_module.base_Model

    input_dim = int(train_samples[0].values.shape[1])
    win_size = int(args.win_size or 64)
    config = SimpleNamespace(
        dataset="FusionTrack",
        input_channels=input_dim,
        kernel_size=4,
        stride=1,
        final_out_channels=32,
        project=2,
        dropout=0.45,
        features_len=_cap_features_len(win_size),
        window_size=win_size,
        time_step=win_size,
        num_epoch=int(args.epochs),
        beta1=0.9,
        beta2=0.99,
        lr=float(args.lr),
        weight=5e-4,
        drop_last=False,
        batch_size=int(args.batch_size),
        trend_rate=1.0,
        rate=0.6,
        cut_rate=max(2, win_size // 4),
        dim=input_dim,
    )
    train_values = np.stack([sample.values for sample in train_samples]).astype(np.float32)
    augmented = cut_add_paste_outlier(train_values.copy(), config)
    augmented_count = max(1, int(round(float(config.rate) * len(train_values))))
    augmented = augmented[:augmented_count]
    train_x = np.concatenate([train_values, augmented], axis=0)
    train_y = np.concatenate(
        [np.zeros(len(train_values), dtype=np.int64), np.ones(len(augmented), dtype=np.int64)]
    )
    order = np.arange(len(train_x))
    np.random.shuffle(order)
    train_x = train_x[order]
    train_y = train_y[order]

    split = max(1, int(0.8 * len(train_x)))
    train_loader = DataLoader(
        CapDataset(train_x[:split], train_y[:split]),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        CapDataset(train_x[split:], train_y[split:]),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    score_loader = DataLoader(
        SequenceDataset(val_samples, channel_first=True),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    model = base_Model(config, device).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(config.beta1), float(config.beta2)),
        weight_decay=float(config.weight),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _cap_epoch(model, train_loader, device, optimizer)
        val_loss = _cap_epoch(model, val_loader, device, optimizer=None)
        row = {
            "epoch": epoch,
            "loss": train_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    score_rows = _score_cutaddpaste(args, model, score_loader, val_samples, device)
    return history, score_rows, {
        "official_components": [
            "models.CutAddPaste.network.model.base_Model",
            "dataloader.generate_negative.cut_add_paste_outlier",
        ],
        "input_dim": input_dim,
        "features_len": int(config.features_len),
        "pseudo_anomaly_rate": float(config.rate),
    }


def _run_timemixer(
    args: argparse.Namespace,
    train_samples: list[SequenceSample],
    val_samples: list[SequenceSample],
    device: torch.device,
) -> tuple[list[dict[str, float | int]], list[dict[str, Any]], dict[str, Any]]:
    from models.TimeMixer import Model

    input_dim = int(train_samples[0].values.shape[1])
    config = _timemixer_config(args, input_dim=input_dim)
    model = Model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    criterion = nn.MSELoss()
    point_criterion = nn.MSELoss(reduction="none")
    train_loader = DataLoader(
        SequenceDataset(train_samples),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        SequenceDataset(val_samples),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for batch, _ in train_loader:
            inputs = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, None, None, None)
            loss = criterion(outputs, inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * int(inputs.shape[0])
            count += int(inputs.shape[0])
        train_loss = total / max(count, 1)
        val_loss = _timemixer_val_loss(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "loss": train_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    score_rows = _score_timemixer(args, model, val_loader, val_samples, point_criterion, device)
    return history, score_rows, {
        "official_components": ["models.TimeMixer.Model"],
        "input_dim": input_dim,
        "down_sampling_layers": int(config.down_sampling_layers),
        "decomp_method": str(config.decomp_method),
    }


def _timemixer_config(args: argparse.Namespace, input_dim: int) -> SimpleNamespace:
    win_size = int(args.win_size or 64)
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=win_size,
        label_len=0,
        pred_len=win_size,
        down_sampling_window=2,
        down_sampling_layers=1 if win_size >= 8 else 0,
        down_sampling_method="avg",
        channel_independence=0,
        e_layers=int(args.e_layers),
        moving_avg=3,
        decomp_method="moving_avg",
        top_k=5,
        d_model=int(args.d_model),
        d_ff=int(args.d_model) * 4,
        dropout=float(args.dropout),
        enc_in=int(input_dim),
        c_out=int(input_dim),
        embed="timeF",
        freq="s",
        use_norm=1,
        use_future_temporal_feature=0,
    )


@torch.no_grad()
def _timemixer_val_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch, _ in loader:
        inputs = batch.to(device)
        outputs = model(inputs, None, None, None)
        loss = criterion(outputs, inputs)
        total += float(loss.item()) * int(inputs.shape[0])
        count += int(inputs.shape[0])
    return total / max(count, 1)


@torch.no_grad()
def _score_timemixer(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    samples: list[SequenceSample],
    point_criterion: nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch, indices in loader:
        inputs = batch.to(device)
        outputs = model(inputs, None, None, None)
        scores = torch.mean(point_criterion(inputs, outputs), dim=-1)
        sample_scores = _aggregate_time_scores(scores, float(args.top_fraction)).cpu().tolist()
        for sample_index, score in zip(indices.tolist(), sample_scores):
            rows.append(_score_row(args, samples[int(sample_index)], score))
    return rows


def _load_official_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cap_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(mode=optimizer is not None)
    total = 0.0
    count = 0
    for batch, labels in loader:
        inputs = batch.to(device)
        targets = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.item()) * int(inputs.shape[0])
        count += int(inputs.shape[0])
    return total / max(count, 1)


@torch.no_grad()
def _score_cutaddpaste(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    samples: list[SequenceSample],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch, indices in loader:
        logits = model(batch.to(device))
        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        for sample_index, score in zip(indices.tolist(), scores):
            rows.append(_score_row(args, samples[int(sample_index)], score))
    return rows


@torch.no_grad()
def _score_reconstruction_method(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    samples: list[SequenceSample],
    device: torch.device,
    score_fn: Any,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch, indices in loader:
        inputs = batch.to(device)
        outputs = model(inputs)[0]
        scores = score_fn(inputs, outputs)
        if scores.ndim == 3:
            scores = scores.mean(dim=-1)
        sample_scores = _aggregate_time_scores(scores, float(args.top_fraction)).cpu().tolist()
        for sample_index, score in zip(indices.tolist(), sample_scores):
            rows.append(_score_row(args, samples[int(sample_index)], score))
    return rows


def _score_row(args: argparse.Namespace, sample: SequenceSample, score: Any) -> dict[str, Any]:
    row = {
        "sample_id": sample.sample_id,
        **_window_id_field(sample),
        "sequence": sample.sequence,
        "track_id": sample.track_id,
        "source": f"official_{args.method}:{args.task}",
        "score": _finite(score),
        "component_scores": {"official_adapter_score": _finite(score)},
        "metadata": {
            "method": args.method,
            "task": args.task,
            "adapter": "fixed_length_multivariate_series",
            **sample.metadata,
        },
    }
    return row


def _aggregate_time_scores(scores: torch.Tensor, top_fraction: float) -> torch.Tensor:
    if scores.ndim == 1:
        return scores
    k = max(1, int(math.ceil(float(scores.shape[-1]) * max(min(top_fraction, 1.0), 0.0))))
    values, _ = torch.topk(scores, k=k, dim=-1)
    return values.mean(dim=-1)


def _load_samples(path: Path, task: str, win_size: int) -> list[SequenceSample]:
    samples: list[SequenceSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if task == "individual":
                sample = _individual_sample(row, win_size)
                if sample is not None:
                    samples.append(sample)
            else:
                samples.extend(_group_samples(row, win_size))
    return samples


def _individual_sample(row: dict[str, Any], win_size: int) -> SequenceSample | None:
    values: list[list[float]] = []
    previous_center: tuple[float, float] | None = None
    previous_frame: int | None = None
    for point in sorted(row.get("points", []), key=lambda item: int(item.get("frame_id", 0))):
        fused = point.get("fused") if isinstance(point.get("fused"), dict) else {}
        center = _center(fused)
        if center is None:
            continue
        frame_id = int(point.get("frame_id", len(values)))
        delta_frame = max(frame_id - previous_frame, 1) if previous_frame is not None else 1
        if previous_center is None:
            velocity = (0.0, 0.0)
        else:
            velocity = (
                (center[0] - previous_center[0]) / delta_frame,
                (center[1] - previous_center[1]) / delta_frame,
            )
        speed = math.hypot(*velocity)
        bbox_area = _bbox_area_from_point(point)
        modal = point.get("modal") if isinstance(point.get("modal"), dict) else {}
        source_modalities = fused.get("source_modalities", [])
        values.append(
            [
                center[0],
                center[1],
                velocity[0],
                velocity[1],
                speed,
                _float(fused.get("confidence")),
                _float(modal.get("offset_distance")),
                bbox_area,
                float(len(source_modalities)) if isinstance(source_modalities, list) else 0.0,
            ]
        )
        previous_center = center
        previous_frame = frame_id
    if not values:
        return None
    return SequenceSample(
        sample_id=str(row.get("sample_id", "")),
        sequence=str(row.get("sequence", "")),
        track_id=str(row.get("track_id", "")),
        values=_resample(np.asarray(values, dtype=np.float32), win_size),
        metadata={"num_original_points": len(values)},
    )


def _group_samples(window: dict[str, Any], win_size: int) -> list[SequenceSample]:
    sequence = str(window.get("sequence", ""))
    frame_start = int(window.get("frame_start", 0))
    frame_end = int(window.get("frame_end", frame_start + win_size - 1))
    frames = list(range(frame_start, frame_end + 1))
    if len(frames) != win_size:
        frames = frames[:win_size]
        while len(frames) < win_size:
            frames.append(frames[-1] + 1 if frames else len(frames))
    centroids = _group_centroids(window, frames)
    rows: list[SequenceSample] = []
    for obj in window.get("objects", []):
        track_id = str(obj.get("track_id", ""))
        sample_id = str(obj.get("sample_id", ""))
        if not sample_id:
            if not track_id:
                continue
            sample_id = f"{sequence}:{track_id}"
        if not track_id:
            track_id = sample_id
        values: list[list[float]] = []
        state_by_frame = {
            int(state.get("frame_id", -1)): state for state in obj.get("states", [])
        }
        last_center: tuple[float, float] | None = None
        last_frame: int | None = None
        visible_count = 0
        for frame_id in frames:
            state = state_by_frame.get(frame_id, {})
            center = _center_from_state(state) if isinstance(state, dict) else None
            visible = 1.0 if center is not None else 0.0
            if center is None:
                center = last_center if last_center is not None else centroids.get(frame_id, (0.0, 0.0))
            delta_frame = max(frame_id - last_frame, 1) if last_frame is not None else 1
            if last_center is None:
                velocity = (0.0, 0.0)
            else:
                velocity = (
                    (center[0] - last_center[0]) / delta_frame,
                    (center[1] - last_center[1]) / delta_frame,
                )
            centroid = centroids.get(frame_id, center)
            speed = math.hypot(*velocity)
            modal = state.get("modal") if isinstance(state.get("modal"), dict) else {}
            values.append(
                [
                    center[0],
                    center[1],
                    center[0] - centroid[0],
                    center[1] - centroid[1],
                    velocity[0],
                    velocity[1],
                    speed,
                    _bbox_area_from_state(state),
                    _float(modal.get("offset_distance")),
                    visible,
                ]
            )
            if visible:
                visible_count += 1
            last_center = center
            last_frame = frame_id
        rows.append(
            SequenceSample(
                sample_id=sample_id,
                sequence=sequence,
                track_id=track_id,
                values=np.asarray(values, dtype=np.float32),
                metadata={
                    "window_id": str(window.get("window_id", "")),
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "visible_count": visible_count,
                },
            )
        )
    return rows


def _group_centroids(window: dict[str, Any], frames: list[int]) -> dict[int, tuple[float, float]]:
    centers_by_frame: dict[int, list[tuple[float, float]]] = {frame_id: [] for frame_id in frames}
    for obj in window.get("objects", []):
        for state in obj.get("states", []):
            if not isinstance(state, dict):
                continue
            frame_id = int(state.get("frame_id", -1))
            center = _center_from_state(state)
            if frame_id in centers_by_frame and center is not None:
                centers_by_frame[frame_id].append(center)
    centroids: dict[int, tuple[float, float]] = {}
    last = (0.0, 0.0)
    for frame_id in frames:
        centers = centers_by_frame.get(frame_id, [])
        if centers:
            last = (
                float(np.mean([center[0] for center in centers])),
                float(np.mean([center[1] for center in centers])),
            )
        centroids[frame_id] = last
    return centroids


def _center_from_state(state: dict[str, Any]) -> tuple[float, float] | None:
    center = _center(state)
    if center is not None:
        return center
    for modality in ("fused", "rgb", "thermal"):
        value = state.get(modality)
        if isinstance(value, dict):
            center = _center(value)
            if center is not None:
                return center
    return None


def _center(value: dict[str, Any]) -> tuple[float, float] | None:
    center = value.get("center_xy")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    x = _float(center[0])
    y = _float(center[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _bbox_area_from_point(point: dict[str, Any]) -> float:
    fused_bbox = point.get("bbox_xywh")
    if isinstance(fused_bbox, (list, tuple)):
        return _bbox_area(fused_bbox)
    for modality in ("rgb", "thermal"):
        value = point.get(modality)
        if isinstance(value, dict):
            area = _bbox_area(value.get("bbox_xywh"))
            if area > 0:
                return area
    return 0.0


def _bbox_area_from_state(state: dict[str, Any]) -> float:
    for modality in ("fused", "rgb", "thermal"):
        value = state.get(modality)
        if isinstance(value, dict):
            area = _bbox_area(value.get("bbox_xywh"))
            if area > 0:
                return area
    return _bbox_area(state.get("bbox_xywh"))


def _bbox_area(bbox: Any) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return 0.0
    return max(_float(bbox[2]), 0.0) * max(_float(bbox[3]), 0.0)


def _resample(values: np.ndarray, win_size: int) -> np.ndarray:
    if values.shape[0] == win_size:
        return values.astype(np.float32)
    if values.shape[0] == 1:
        return np.repeat(values, win_size, axis=0).astype(np.float32)
    source = np.linspace(0.0, 1.0, values.shape[0])
    target = np.linspace(0.0, 1.0, win_size)
    columns = [np.interp(target, source, values[:, column]) for column in range(values.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def _standardize(train_samples: list[SequenceSample], val_samples: list[SequenceSample]) -> None:
    train_values = np.concatenate([sample.values for sample in train_samples], axis=0)
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    for sample in [*train_samples, *val_samples]:
        sample.values = ((sample.values - mean) / std).astype(np.float32)


def _patch_config(win_size: int) -> tuple[int, int]:
    patch_size = min(16, max(4, int(win_size)))
    if patch_size > win_size:
        patch_size = win_size
    patch_stride = max(1, patch_size // 2)
    return patch_size, patch_stride


def _cap_features_len(win_size: int) -> int:
    length = int(win_size)
    for kernel, padding in ((4, 2), (8, 4), (8, 4)):
        length = math.floor((length + 2 * padding - kernel) / 1 + 1)
        length = math.floor((length + 2 * 1 - 2) / 2 + 1)
    return int(length)


def _window_id_field(sample: SequenceSample) -> dict[str, str]:
    window_id = sample.metadata.get("window_id")
    if window_id in (None, ""):
        return {}
    return {"window_id": str(window_id)}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_score_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["sample_id", "window_id", "sequence", "track_id", "score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _finite(value: Any) -> float:
    result = _float(value)
    return result if math.isfinite(result) else 0.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gpu_name(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    try:
        return torch.cuda.get_device_name(device)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
