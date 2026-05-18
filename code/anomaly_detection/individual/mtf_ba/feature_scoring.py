from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mtf_ba.feature_training import (
    FEATURE_MIN_LENGTH_DEFAULTS,
    FEATURE_NUM_LAYERS,
    LSTMAutoencoder,
    compute_normalization_stats,
    filter_sequences,
    infer_feature_columns,
    load_feature_pickle,
)


class InferenceSequenceDataset(Dataset):
    """
    Dataset used during the baseline-style encoding/scoring stage.

    Each item keeps:
    - `sample_id`: stable object identity used across the whole pipeline
    - `sequence`: normalized feature sequence tensor, shape [seq_len, feature_dim]

    We return `sample_id` explicitly because the baseline scoring stage ultimately
    needs per-trajectory dictionaries:
    - sample_id -> embedding
    - sample_id -> reconstruction loss
    - sample_id -> final score
    """

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        columns: list[str],
        normalize: bool,
        normalization_stats: dict[str, list[float]] | None,
    ) -> None:
        self.items: list[tuple[str, torch.Tensor]] = []
        mean = None
        std = None

        if normalize and normalization_stats is not None:
            mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
            std = np.asarray(normalization_stats["std"], dtype=np.float32)

        for sample_id, trip in data_dict.items():
            values = trip[columns].to_numpy(dtype=np.float32)
            if normalize and mean is not None and std is not None:
                values = (values - mean) / std
            self.items.append((sample_id, torch.tensor(values, dtype=torch.float32)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.items[idx]

    @staticmethod
    def collate_fn(
        batch: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Build one inference batch with the same padding mask semantics used in
        training.

        Returned values:
        - sample_ids: length-B list
        - padded:     [B, max_len, feature_dim]
        - mask:       [B, max_len], True on valid timesteps only
        """
        sample_ids = [sample_id for sample_id, _ in batch]
        sequences = [sequence for _, sequence in batch]
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        lengths = torch.tensor([item.size(0) for item in sequences], dtype=torch.long)
        max_len = padded.size(1)
        time_index = torch.arange(max_len, dtype=torch.long).unsqueeze(0)
        mask = time_index < lengths.unsqueeze(1)
        return sample_ids, padded, mask


def masked_mse_per_sample(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> list[float]:
    """
    Compute one reconstruction MSE score per trajectory.

    This mirrors the baseline idea of evaluating reconstruction quality for each
    trajectory individually, but respects our padded batching setup by ignoring
    padded timesteps.
    """
    squared_error = (recon - target) ** 2
    scores: list[float] = []

    for i in range(recon.size(0)):
        valid_steps = mask[i].unsqueeze(-1).to(dtype=recon.dtype)
        valid_error = squared_error[i] * valid_steps
        valid_count = valid_steps.sum() * recon.size(-1)
        valid_count = torch.clamp(valid_count, min=1.0)
        loss = valid_error.sum() / valid_count
        scores.append(float(loss.detach().cpu().item()))

    return scores


def compute_final_loss(
    embedding_dict: dict[str, np.ndarray],
    loss_dict: dict[str, float],
    k_neighbors: int,
) -> dict[str, float]:
    """
    Baseline-style neighborhood-adjusted anomaly score.

    This is the same high-level logic as `compute_final_loss(...)` in the
    baseline's `Ensemble/data_func.py`:

    1. use embeddings to find nearest neighbors
    2. compare each trajectory's reconstruction loss against its neighbors
    3. normalize that local deviation by the neighbors' own local deviations

    Intuition:
    - high reconstruction loss alone may be insufficient
    - a trajectory is more suspicious if its loss is unusually different from
      nearby trajectories in embedding space
    """
    keys = list(embedding_dict.keys())
    if not keys:
        return {}

    embeddings = np.array([embedding_dict[key] for key in keys], dtype=np.float32)
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(len(keys), -1)

    # NearestNeighbors requires n_neighbors <= num_samples.
    effective_k = min(k_neighbors + 1, len(keys))
    if effective_k <= 1:
        return {key: float(loss_dict[key]) for key in keys}

    nbrs = NearestNeighbors(n_neighbors=effective_k, algorithm="ball_tree").fit(
        embeddings
    )
    _, indices = nbrs.kneighbors(embeddings)

    avg_diffs: list[float] = []
    for i in range(len(keys)):
        item_loss = loss_dict[keys[i]]
        neighbor_indices = indices[i][1:]
        if len(neighbor_indices) == 0:
            avg_diffs.append(0.0)
            continue
        neighbor_losses = [loss_dict[keys[idx]] for idx in neighbor_indices]
        avg_diff = float(np.mean([abs(item_loss - loss) for loss in neighbor_losses]))
        avg_diffs.append(avg_diff)

    final_losses: dict[str, float] = {}
    for i in range(len(keys)):
        neighbor_indices = indices[i][1:]
        if len(neighbor_indices) == 0:
            final_losses[keys[i]] = float(avg_diffs[i])
            continue

        avg_diff_i = avg_diffs[i]
        avg_diff_j = float(np.mean([avg_diffs[j] for j in neighbor_indices]))
        if abs(avg_diff_j) < 1e-8:
            # If the neighborhood itself is perfectly flat, fall back to the
            # unnormalized local deviation rather than creating unstable ratios.
            final_loss = avg_diff_i
        else:
            final_loss = avg_diff_i / avg_diff_j
        final_losses[keys[i]] = float(final_loss)

    return final_losses


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_pickle(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def save_score_records_jsonl(
    path: str | Path,
    feature_name: str,
    score_dict: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample_id, score in score_dict.items():
            record = {
                "sample_id": sample_id,
                "feature_name": feature_name,
                "score": float(score),
            }
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def infer_feature_min_length(feature_name: str, explicit_min_length: int | None) -> int:
    """
    Keep scoring-time sequence filtering consistent with training-time filtering.

    We use the same feature-aware defaults as training:
    - route/speed: 10
    - shape: 2

    This is important because shape features are naturally shorter after
    de-duplication and resampling. If scoring used a hard-coded min_length=10,
    many valid shape trajectories would disappear at inference time even though
    they were kept during training.
    """
    if explicit_min_length is not None:
        return explicit_min_length
    if feature_name not in FEATURE_MIN_LENGTH_DEFAULTS:
        raise ValueError(f"Unsupported feature name: {feature_name}")
    return FEATURE_MIN_LENGTH_DEFAULTS[feature_name]


def select_checkpoint(train_summary: dict[str, Any], model_dir: Path) -> Path:
    best_checkpoint = train_summary.get("best_checkpoint")
    if best_checkpoint:
        return Path(best_checkpoint)

    best_model = model_dir / "best_model.pth"
    if best_model.exists():
        return best_model

    checkpoints = sorted(model_dir.glob("model_epoch_*.pth"))
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(f"No checkpoint found under {model_dir}")


def build_inference_loader(
    feature_dict: dict[str, pd.DataFrame],
    feature_name: str,
    normalize: bool,
    normalization_stats: dict[str, list[float]] | None,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    columns = infer_feature_columns(feature_name)
    dataset = InferenceSequenceDataset(
        data_dict=feature_dict,
        columns=columns,
        normalize=normalize,
        normalization_stats=normalization_stats,
    )
    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=InferenceSequenceDataset.collate_fn,
        num_workers=num_workers,
    )


def score_feature_detector(
    feature_name: str,
    feature_pkl: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    cuda_device: str = "cuda:0",
    k_neighbors: int = 6,
    min_length: int | None = None,
    max_length: int = 1000,
) -> dict[str, Any]:
    """
    Baseline-style encoding + scoring stage for one trained detector.

    Inputs:
    - one feature pickle (`*_train.pkl`, `*_val.pkl`, or `*_test.pkl`)
    - one trained detector directory containing:
      - `train_summary.json`
      - `normalization_stats.json`
      - model checkpoints

    Outputs:
    - `embeddings.pkl`
    - `reconstruction_loss.pkl`
    - `final_scores.pkl`
    - `score_records.jsonl`
    - `scoring_summary.json`
    """
    feature_pkl = Path(feature_pkl)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary = load_json(model_dir / "train_summary.json")
    normalization_stats = load_json(model_dir / "normalization_stats.json")
    checkpoint_path = select_checkpoint(train_summary, model_dir)

    columns = infer_feature_columns(feature_name)
    hidden_size = int(train_summary["hidden_size"])
    num_layers = int(train_summary["num_layers"])
    input_size = int(train_summary["input_size"])
    normalize = bool(train_summary["normalize"])
    resolved_min_length = infer_feature_min_length(feature_name, min_length)

    feature_dict = filter_sequences(
        load_feature_pickle(feature_pkl),
        lower_threshold=resolved_min_length,
        upper_threshold=max_length,
    )

    loader = build_inference_loader(
        feature_dict=feature_dict,
        feature_name=feature_name,
        normalize=normalize,
        normalization_stats=normalization_stats,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    embedding_dict: dict[str, np.ndarray] = {}
    loss_dict: dict[str, float] = {}

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Scoring {feature_name}", leave=False)
        for sample_ids, batch, mask in loop:
            batch = batch.to(device)
            mask = mask.to(device)

            recon, embeddings = model(batch, return_embeddings=True)
            per_sample_losses = masked_mse_per_sample(recon, batch, mask)
            embedding_array = embeddings.detach().cpu().numpy()

            for sample_id, sample_loss, sample_embedding in zip(
                sample_ids, per_sample_losses, embedding_array
            ):
                embedding_dict[sample_id] = sample_embedding
                loss_dict[sample_id] = float(sample_loss)

    final_score_dict = compute_final_loss(
        embedding_dict=embedding_dict,
        loss_dict=loss_dict,
        k_neighbors=k_neighbors,
    )

    save_pickle(output_dir / "embeddings.pkl", embedding_dict)
    save_pickle(output_dir / "reconstruction_loss.pkl", loss_dict)
    save_pickle(output_dir / "final_scores.pkl", final_score_dict)
    save_score_records_jsonl(
        output_dir / "score_records.jsonl",
        feature_name=feature_name,
        score_dict=final_score_dict,
    )

    summary = {
        "feature_name": feature_name,
        "feature_pkl": str(feature_pkl),
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "columns": columns,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "normalize": normalize,
        "batch_size": batch_size,
        "num_sequences_scored": len(final_score_dict),
        "k_neighbors": k_neighbors,
        "resolved_min_length": resolved_min_length,
        "max_length": max_length,
    }
    save_json(output_dir / "scoring_summary.json", summary)
    return summary
