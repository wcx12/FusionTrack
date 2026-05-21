from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable

from baselines.individual_features import extract_center_sequence


BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SOURCE = "individual_trajectory_lm:ngram"


def trajectory_to_tokens(
    trajectory: dict,
    grid_size: int = 16,
    speed_bins: tuple[float, ...] = (0.5, 1.5, 3.0),
    turn_bins: tuple[float, ...] = (0.5, 1.5),
    include_modality: bool = False,
) -> list[str]:
    sequence = extract_center_sequence(trajectory)
    if len(sequence) < 2:
        return []

    epsilon = 1.0 / float(max(int(grid_size), 1))
    tokens: list[str] = []
    step_vectors: list[tuple[float, float]] = []
    for (frame0, x0, y0), (frame1, x1, y1) in zip(sequence, sequence[1:]):
        delta_frames = max(int(frame1) - int(frame0), 1)
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        speed = math.hypot(dx, dy) / float(delta_frames)
        if step_vectors:
            angle = _turn_angle(step_vectors[-1], (dx, dy))
            tokens.append(f"turn:{_bin_index(angle, turn_bins)}")
        step_vectors.append((dx, dy))
        tokens.append(f"dir:{_direction_token(dx, dy, epsilon)}")
        tokens.append(f"spd:{_bin_index(speed, speed_bins)}")

    if include_modality:
        tokens.extend(_modal_offset_tokens(trajectory, grid_size))

    return tokens


def fit_ngram_language_model(
    train_trajectories: Iterable[dict],
    ngram_order: int = 2,
    alpha: float = 1.0,
    **token_params: Any,
) -> dict[str, Any]:
    ngram_order = max(1, int(ngram_order))
    alpha = float(alpha)
    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be a finite positive value")

    context_counts: Counter[tuple[str, ...]] = Counter()
    transition_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    vocab: set[str] = {EOS_TOKEN, UNK_TOKEN}
    sequence_count = 0
    for trajectory in train_trajectories:
        tokens = trajectory_to_tokens(trajectory, **token_params)
        vocab.update(tokens)
        sequence_count += 1
        for context, token in _ngram_events(tokens, ngram_order):
            context_counts[context] += 1
            transition_counts[context][token] += 1

    return {
        "ngram_order": ngram_order,
        "alpha": alpha,
        "token_params": dict(token_params),
        "vocab": sorted(vocab),
        "context_counts": dict(context_counts),
        "transition_counts": {
            context: dict(counts) for context, counts in transition_counts.items()
        },
        "sequence_count": sequence_count,
    }


def score_ngram_language_model(
    model: dict[str, Any],
    score_trajectories: Iterable[dict],
) -> list[dict[str, Any]]:
    ngram_order = int(model["ngram_order"])
    alpha = float(model["alpha"])
    token_params = dict(model.get("token_params", {}))
    vocab = set(model.get("vocab", [])) or {EOS_TOKEN, UNK_TOKEN}
    vocab_size = len(vocab)
    context_counts = model.get("context_counts", {})
    transition_counts = model.get("transition_counts", {})

    rows: list[dict[str, Any]] = []
    for trajectory in score_trajectories:
        tokens = trajectory_to_tokens(trajectory, **token_params)
        nll = 0.0
        event_count = 0
        for context, token in _ngram_events(tokens, ngram_order):
            mapped_context = tuple(
                item if item in vocab or item == BOS_TOKEN else UNK_TOKEN for item in context
            )
            mapped_token = token if token in vocab else UNK_TOKEN
            context_total = int(context_counts.get(mapped_context, 0))
            token_count = int(transition_counts.get(mapped_context, {}).get(mapped_token, 0))
            probability = (token_count + alpha) / (
                context_total + alpha * float(vocab_size)
            )
            nll -= math.log(probability)
            event_count += 1
        score = nll / float(event_count) if event_count else 0.0
        if not math.isfinite(score):
            score = 0.0
        rows.append(
            {
                "sample_id": _sample_id(trajectory),
                "sequence": str(trajectory.get("sequence", "")),
                "track_id": str(trajectory.get("track_id", "")),
                "source": SOURCE,
                "score": float(score),
                "component_scores": {
                    "negative_log_likelihood": float(score),
                    "num_tokens": int(len(tokens)),
                },
                "metadata": {
                    "ngram_order": ngram_order,
                    "alpha": alpha,
                    "grid_size": int(token_params.get("grid_size", 16)),
                    "seed": int(model.get("seed", 42)),
                    "vocab_size": int(vocab_size),
                },
            }
        )
    return rows


def run_ngram_language_model(
    train_trajectories: Iterable[dict],
    score_trajectories: Iterable[dict],
    ngram_order: int = 2,
    alpha: float = 1.0,
    grid_size: int = 16,
    seed: int = 42,
) -> list[dict[str, Any]]:
    model = fit_ngram_language_model(
        train_trajectories,
        ngram_order=ngram_order,
        alpha=alpha,
        grid_size=grid_size,
    )
    model["seed"] = int(seed)
    return score_ngram_language_model(model, score_trajectories)


def _ngram_events(tokens: list[str], ngram_order: int) -> list[tuple[tuple[str, ...], str]]:
    context_width = max(0, int(ngram_order) - 1)
    padded = [BOS_TOKEN] * context_width + list(tokens) + [EOS_TOKEN]
    events: list[tuple[tuple[str, ...], str]] = []
    for index in range(context_width, len(padded)):
        context = tuple(padded[index - context_width : index])
        events.append((context, padded[index]))
    return events


def _direction_token(dx: float, dy: float, epsilon: float) -> str:
    if math.hypot(dx, dy) <= epsilon:
        return "STAY"
    angle = math.atan2(dy, dx)
    directions = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
    index = int(round(angle / (math.pi / 4.0))) % len(directions)
    return directions[index]


def _bin_index(value: float, bins: tuple[float, ...]) -> int:
    for index, threshold in enumerate(bins):
        if value <= float(threshold):
            return index
    return len(bins)


def _turn_angle(
    previous: tuple[float, float],
    current: tuple[float, float],
) -> float:
    prev_norm = math.hypot(*previous)
    curr_norm = math.hypot(*current)
    if prev_norm == 0.0 or curr_norm == 0.0:
        return 0.0
    cosine = (
        previous[0] * current[0] + previous[1] * current[1]
    ) / (prev_norm * curr_norm)
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def _modal_offset_tokens(trajectory: dict, grid_size: int) -> list[str]:
    tokens: list[str] = []
    scale = float(max(int(grid_size), 1))
    for point in trajectory.get("points", []):
        fused = _center(point.get("fused"))
        if fused is None:
            continue
        offsets = []
        for modality in ("rgb", "thermal"):
            center = _center(point.get(modality))
            if center is not None:
                offsets.append(math.hypot(center[0] - fused[0], center[1] - fused[1]))
        if offsets:
            tokens.append(f"modal_offset:{_bin_index(sum(offsets) / len(offsets), (scale / 4, scale / 2, scale))}")
    return tokens


def _center(state: Any) -> tuple[float, float] | None:
    if not isinstance(state, dict):
        return None
    center = state.get("center_xy")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    try:
        x = float(center[0])
        y = float(center[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _sample_id(trajectory: dict) -> str:
    sample_id = trajectory.get("sample_id")
    if sample_id not in (None, ""):
        return str(sample_id)
    sequence = str(trajectory.get("sequence", ""))
    track_id = str(trajectory.get("track_id", ""))
    return f"{sequence}:{track_id}" if sequence or track_id else ""
