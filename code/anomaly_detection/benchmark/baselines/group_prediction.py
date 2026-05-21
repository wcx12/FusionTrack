from __future__ import annotations

import math
from typing import Any, Iterable

from protocol.schemas import build_sample_id


MODALITY_ORDER = ("fused", "rgb", "thermal")


def run_prediction_baseline(windows: Iterable[dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        rows.extend(_score_window(window))
    return rows


def _score_window(window: dict) -> list[dict[str, Any]]:
    sequence = str(window.get("sequence", ""))
    frame_start, frame_end = _window_frame_bounds(window)
    window_id = str(window.get("window_id", window.get("sample_id", "")))
    rows: list[dict[str, Any]] = []
    seen_track_ids: set[str] = set()
    for obj in sorted(window.get("objects", []), key=lambda item: str(item.get("track_id", ""))):
        track_id = obj.get("track_id")
        if track_id in (None, ""):
            continue
        track_id = str(track_id)
        if track_id in seen_track_ids:
            raise ValueError(f"Duplicate track_id '{track_id}' in window '{window_id}'")
        seen_track_ids.add(track_id)
        score = _prediction_residual_score(_center_sequence(obj))
        rows.append(
            {
                "sample_id": _sample_id(obj, sequence, track_id),
                "window_id": window_id,
                "sequence": sequence,
                "track_id": track_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "source": "group_prediction:linear",
                "score": score,
                "component_scores": {"prediction_residual": score},
                "metadata": {
                    "method": "linear",
                    "window_id": window_id,
                },
            }
        )
    return rows


def _prediction_residual_score(sequence: list[tuple[int, float, float]]) -> float:
    if len(sequence) < 3:
        return 0.0
    residuals: list[float] = []
    for previous_previous, previous, current in zip(sequence, sequence[1:], sequence[2:]):
        frame0, x0, y0 = previous_previous
        frame1, x1, y1 = previous
        frame2, x2, y2 = current
        delta01 = max(frame1 - frame0, 1)
        delta12 = max(frame2 - frame1, 1)
        vx = (x1 - x0) / float(delta01)
        vy = (y1 - y0) / float(delta01)
        predicted = (x1 + vx * delta12, y1 + vy * delta12)
        residuals.append(math.dist(predicted, (x2, y2)))
    score = max(residuals) if residuals else 0.0
    return float(score) if math.isfinite(score) else 0.0


def _center_sequence(obj: dict) -> list[tuple[int, float, float]]:
    sequence: list[tuple[int, float, float]] = []
    for state in obj.get("states", []):
        if "frame_id" not in state:
            continue
        center = _center_from_state(state)
        if center is None:
            continue
        sequence.append((int(state["frame_id"]), center[0], center[1]))
    return sorted(sequence, key=lambda item: item[0])


def _center_from_state(state: dict) -> tuple[float, float] | None:
    direct = _center(state)
    if direct is not None:
        return direct
    for modality in MODALITY_ORDER:
        modality_state = state.get(modality)
        if isinstance(modality_state, dict):
            center = _center(modality_state)
            if center is not None:
                return center
    return None


def _center(value: dict) -> tuple[float, float] | None:
    center = value.get("center_xy")
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


def _window_frame_bounds(window: dict) -> tuple[int, int]:
    frame_ids: list[int] = []
    for obj in window.get("objects", []):
        for state in obj.get("states", []):
            if "frame_id" in state:
                frame_ids.append(int(state["frame_id"]))
    default_start = min(frame_ids) if frame_ids else 0
    default_end = max(frame_ids) if frame_ids else default_start
    return (
        int(window.get("frame_start", default_start)),
        int(window.get("frame_end", default_end)),
    )


def _sample_id(obj: dict, sequence: str, track_id: str) -> str:
    sample_id = obj.get("sample_id")
    if sample_id not in (None, ""):
        return str(sample_id)
    return build_sample_id(sequence, track_id)
