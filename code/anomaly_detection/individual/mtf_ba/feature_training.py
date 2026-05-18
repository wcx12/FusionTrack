from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


FEATURE_COLUMNS = {
    "route_rgb": ["latitude", "longitude"],
    "route_thermal": ["latitude", "longitude"],
    "speed_rgb": ["speed"],
    "speed_thermal": ["speed"],
    "shape_rgb": ["delta_x", "delta_y"],
    "shape_thermal": ["delta_x", "delta_y"],
}


FEATURE_NUM_LAYERS = {
    "route_rgb": 2,
    "route_thermal": 2,
    "speed_rgb": 1,
    "speed_thermal": 1,
    "shape_rgb": 2,
    "shape_thermal": 2,
}


# Feature sequences do not all have the same natural length scale.
#
# - route/speed sequences usually preserve most original trajectory timesteps,
#   so keeping the baseline-style minimum length of 10 is reasonable.
# - shape sequences are produced after de-duplication + normalization +
#   resampling, and are often much shorter. Reusing 10 here can easily filter
#   out every sample and make the training set empty.
#
# These defaults make the training pipeline robust while still allowing the user
# to override them from the CLI.
FEATURE_MIN_LENGTH_DEFAULTS = {
    "route_rgb": 10,
    "route_thermal": 10,
    "speed_rgb": 10,
    "speed_thermal": 10,
    "shape_rgb": 2,
    "shape_thermal": 2,
}


@dataclass
class TrainConfig:
    feature_name: str
    train_pkl: Path
    val_pkl: Path
    test_pkl: Path
    output_dir: Path
    hidden_size: int = 128
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 1e-5
    seed: int = 42
    normalize: bool = True
    min_length: int | None = None
    max_length: int = 1000
    num_workers: int = 0
    cuda_device: str = "cuda:0"


def set_seed(seed_value: int = 42) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_feature_pickle(path: str | Path) -> dict[str, pd.DataFrame]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def infer_feature_columns(feature_name: str) -> list[str]:
    if feature_name not in FEATURE_COLUMNS:
        raise ValueError(f"Unsupported feature name: {feature_name}")
    return FEATURE_COLUMNS[feature_name]


def infer_feature_min_length(feature_name: str, explicit_min_length: int | None) -> int:
    if explicit_min_length is not None:
        return explicit_min_length
    if feature_name not in FEATURE_MIN_LENGTH_DEFAULTS:
        raise ValueError(f"Unsupported feature name: {feature_name}")
    return FEATURE_MIN_LENGTH_DEFAULTS[feature_name]


def filter_sequences(
    data_dict: dict[str, pd.DataFrame],
    lower_threshold: int,
    upper_threshold: int,
) -> dict[str, pd.DataFrame]:
    return {
        key: value
        for key, value in data_dict.items()
        if lower_threshold <= len(value) <= upper_threshold
    }


def compute_normalization_stats(
    data_dict: dict[str, pd.DataFrame],
    columns: list[str],
) -> dict[str, list[float]]:
    """
    Compute train-only normalization statistics.

    We normalize per feature dimension using all timesteps from all training
    sequences. This keeps the training path data-driven instead of relying on
    dataset-specific constants from the original baseline.
    """
    arrays = []
    for trip in data_dict.values():
        values = trip[columns].to_numpy(dtype=np.float32)
        if values.size == 0:
            continue
        arrays.append(values)

    if not arrays:
        mean = np.zeros((len(columns),), dtype=np.float32)
        std = np.ones((len(columns),), dtype=np.float32)
    else:
        stacked = np.concatenate(arrays, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)

    return {
        "columns": list(columns),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


class SequenceFeatureDataset(Dataset):
    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        columns: list[str],
        normalize: bool,
        normalization_stats: dict[str, list[float]] | None,
    ) -> None:
        self.items: list[torch.Tensor] = []
        self.sample_ids: list[str] = []
        self.columns = columns
        self.normalize = normalize
        self.normalization_stats = normalization_stats

        mean = None
        std = None
        if self.normalize and normalization_stats is not None:
            mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
            std = np.asarray(normalization_stats["std"], dtype=np.float32)

        for sample_id, trip in data_dict.items():
            values = trip[columns].to_numpy(dtype=np.float32)
            if self.normalize and mean is not None and std is not None:
                values = (values - mean) / std
            self.sample_ids.append(sample_id)
            self.items.append(torch.tensor(values, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.items[idx]

    @staticmethod
    def collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad variable-length sequences and return a boolean validity mask.

        Why the mask is needed:

        Trajectories have different lengths, so we pad shorter sequences with
        zeros to build one rectangular tensor `[batch, max_len, feature_dim]`.
        If we were to compute MSE on the full padded tensor directly, the model
        would be penalized on artificial trailing zeros that do not correspond
        to any real trajectory timestep.

        The returned mask has shape `[batch, max_len]`:
        - True  -> this timestep is a real observed timestep
        - False -> this timestep is padding added only for batching

        Later the loss function expands this mask across the feature dimension
        and computes reconstruction error only on the True positions.
        """
        padded = pad_sequence(batch, batch_first=True, padding_value=0.0)
        lengths = torch.tensor([item.size(0) for item in batch], dtype=torch.long)
        max_len = padded.size(1)
        time_index = torch.arange(max_len, dtype=torch.long).unsqueeze(0)
        mask = time_index < lengths.unsqueeze(1)
        return padded, mask


def create_data_loader(
    data_dict: dict[str, pd.DataFrame],
    columns: list[str],
    batch_size: int,
    normalize: bool,
    normalization_stats: dict[str, list[float]] | None,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = SequenceFeatureDataset(
        data_dict=data_dict,
        columns=columns,
        normalize=normalize,
        normalization_stats=normalization_stats,
    )
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=SequenceFeatureDataset.collate_fn,
        num_workers=num_workers,
    )


class Encoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden_n, _) = self.rnn(x)
        return hidden_n[-1]


class Decoder(nn.Module):
    def __init__(self, input_dim: int, n_features: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        repeated = x.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.rnn(repeated)
        return self.output_layer(decoded)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, input_size, hidden_size, num_layers)

    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, x.size(1))
        if return_embeddings:
            return decoded, encoded
        return decoded


class AutoencoderTrainer:
    def __init__(
        self,
        model: LSTMAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        output_dir: Path,
        learning_rate: float,
        device: torch.device,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_val_loss = float("inf")
        self.loss_history = {"train": [], "val": []}
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

    @staticmethod
    def masked_mse_loss(
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE only on valid (non-padded) timesteps.

        Shapes:
        - recon  : [batch, max_len, feature_dim]
        - target : [batch, max_len, feature_dim]
        - mask   : [batch, max_len]

        We first expand the mask to `[batch, max_len, 1]`, so one timestep mask
        value applies to every feature channel at that timestep. Then we compute
        squared reconstruction error and average only over valid elements.

        This avoids a common issue in padded sequence training where the model
        starts learning to reconstruct padding zeros, which can bias the loss
        and make short trajectories look artificially easy.
        """
        expanded_mask = mask.unsqueeze(-1).to(dtype=recon.dtype)
        squared_error = (recon - target) ** 2
        masked_squared_error = squared_error * expanded_mask
        valid_count = expanded_mask.sum() * recon.size(-1)
        valid_count = torch.clamp(valid_count, min=1.0)
        return masked_squared_error.sum() / valid_count

    def run_epoch(self, loader: DataLoader, train: bool, desc: str) -> float:
        self.model.train(mode=train)
        total_loss = 0.0
        loop = tqdm(loader, desc=desc, leave=False)

        for batch, mask in loop:
            batch = batch.to(self.device)
            mask = mask.to(self.device)
            if train:
                self.optimizer.zero_grad()

            recon = self.model(batch)
            loss = self.masked_mse_loss(recon, batch, mask)

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def save_checkpoint(self, epoch: int) -> Path:
        """
        Save only the current best checkpoint.

        We intentionally overwrite the same `best_model.pth` file each time the
        validation loss improves. This keeps the model directory compact and
        avoids leaving behind dozens or hundreds of historical checkpoints when
        the user only cares about the best validation model.
        """
        path = self.output_dir / "best_model.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "loss_history": self.loss_history,
            },
            path,
        )
        return path

    def train(self, num_epochs: int) -> dict[str, Any]:
        last_checkpoint = None
        epochs_without_improvement = 0
        stopped_early = False
        stop_epoch = None

        for epoch in range(num_epochs):
            train_loss = self.run_epoch(
                self.train_loader,
                train=True,
                desc=f"Train epoch {epoch + 1}/{num_epochs}",
            )
            val_loss = self.run_epoch(
                self.val_loader,
                train=False,
                desc=f"Val epoch {epoch + 1}/{num_epochs}",
            )

            self.loss_history["train"].append(train_loss)
            self.loss_history["val"].append(val_loss)

            # Early stopping uses validation loss because validation data is not
            # used for gradient updates. We only reset patience when the new
            # validation loss beats the previous best by at least
            # `early_stopping_min_delta`. This avoids treating tiny numerical
            # fluctuations as meaningful progress.
            if (self.best_val_loss - val_loss) > self.early_stopping_min_delta:
                self.best_val_loss = val_loss
                last_checkpoint = self.save_checkpoint(epoch)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"- Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            if epochs_without_improvement >= self.early_stopping_patience:
                stopped_early = True
                stop_epoch = epoch + 1
                print(
                    "Early stopping triggered: "
                    f"no validation improvement greater than "
                    f"{self.early_stopping_min_delta} for "
                    f"{self.early_stopping_patience} consecutive epochs."
                )
                break

        test_loss = self.run_epoch(self.test_loader, train=False, desc="Test")
        summary = {
            "best_val_loss": self.best_val_loss,
            "test_loss": test_loss,
            "loss_history": self.loss_history,
            "best_checkpoint": str(last_checkpoint) if last_checkpoint else None,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
        }
        return summary


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def train_feature_detector(config: TrainConfig) -> dict[str, Any]:
    """
    Train one detector for one exported feature set.

    This is the training bridge between:
    - our VT-Tiny-MOT feature export pipeline
    - the baseline LSTM autoencoder idea

    The baseline code assumed a single `route/speed/shape` family and
    dataset-specific normalization constants. Here we generalize that setup to
    any exported feature file and compute train-only normalization statistics.
    """
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    columns = infer_feature_columns(config.feature_name)
    min_length = infer_feature_min_length(config.feature_name, config.min_length)
    input_size = len(columns)
    num_layers = FEATURE_NUM_LAYERS[config.feature_name]

    train_dict = filter_sequences(
        load_feature_pickle(config.train_pkl),
        lower_threshold=min_length,
        upper_threshold=config.max_length,
    )
    val_dict = filter_sequences(
        load_feature_pickle(config.val_pkl),
        lower_threshold=min_length,
        upper_threshold=config.max_length,
    )
    test_dict = filter_sequences(
        load_feature_pickle(config.test_pkl),
        lower_threshold=min_length,
        upper_threshold=config.max_length,
    )

    if len(train_dict) == 0:
        raise ValueError(
            f"No training sequences remain after filtering for feature "
            f"{config.feature_name!r}. "
            f"Resolved min_length={min_length}, max_length={config.max_length}. "
            f"Please lower --min-length or regenerate this feature."
        )
    if len(val_dict) == 0:
        raise ValueError(
            f"No validation sequences remain after filtering for feature "
            f"{config.feature_name!r}. "
            f"Resolved min_length={min_length}, max_length={config.max_length}. "
            f"Please lower --min-length or regenerate this feature."
        )
    if len(test_dict) == 0:
        raise ValueError(
            f"No test sequences remain after filtering for feature "
            f"{config.feature_name!r}. "
            f"Resolved min_length={min_length}, max_length={config.max_length}. "
            f"Please lower --min-length or regenerate this feature."
        )

    normalization_stats = (
        compute_normalization_stats(train_dict, columns) if config.normalize else None
    )
    if normalization_stats is not None:
        save_json(config.output_dir / "normalization_stats.json", normalization_stats)

    train_loader = create_data_loader(
        train_dict,
        columns=columns,
        batch_size=config.batch_size,
        normalize=config.normalize,
        normalization_stats=normalization_stats,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = create_data_loader(
        val_dict,
        columns=columns,
        batch_size=config.batch_size,
        normalize=config.normalize,
        normalization_stats=normalization_stats,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = create_data_loader(
        test_dict,
        columns=columns,
        batch_size=config.batch_size,
        normalize=config.normalize,
        normalization_stats=normalization_stats,
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = torch.device(
        config.cuda_device if torch.cuda.is_available() else "cpu"
    )
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=num_layers,
    )
    trainer = AutoencoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        device=device,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
    )

    summary = trainer.train(config.num_epochs)
    full_summary = {
        "feature_name": config.feature_name,
        "train_pkl": str(config.train_pkl),
        "val_pkl": str(config.val_pkl),
        "test_pkl": str(config.test_pkl),
        "output_dir": str(config.output_dir),
        "columns": columns,
        "input_size": input_size,
        "hidden_size": config.hidden_size,
        "num_layers": num_layers,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "seed": config.seed,
        "normalize": config.normalize,
        "resolved_min_length": min_length,
        "max_length": config.max_length,
        "device": str(device),
        "num_train_sequences": len(train_dict),
        "num_val_sequences": len(val_dict),
        "num_test_sequences": len(test_dict),
        **summary,
    }
    save_json(config.output_dir / "train_summary.json", full_summary)
    return full_summary
