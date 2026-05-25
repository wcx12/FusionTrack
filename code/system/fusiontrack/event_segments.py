from __future__ import annotations

import math
from typing import Any


def _finite_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return None
    return number


def _finite_components(components: Any) -> dict[str, float]:
    if not isinstance(components, dict):
        return {}
    normalized: dict[str, float] = {}
    for key, value in components.items():
        number = _finite_float(value, default=None)
        if number is not None:
            normalized[str(key)] = float(number)
    return normalized


def normalize_frame_event_scores(frame_event_scores: Any, source: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(frame_event_scores, list):
        return []
    normalized: list[dict[str, Any]] = []
    for row in frame_event_scores:
        if not isinstance(row, dict):
            continue
        frame = row.get("frame", row.get("frame_id"))
        try:
            frame_id = int(frame)
        except (TypeError, ValueError):
            continue
        score = _finite_float(row.get("score", 0.0), default=0.0)
        if score is None:
            continue
        item: dict[str, Any] = {
            "frame": frame_id,
            "score": float(score),
        }
        reason = row.get("dominant_reason", row.get("reason"))
        if reason:
            item["dominant_reason"] = str(reason)
        components = _finite_components(row.get("component_scores"))
        if components:
            item["component_scores"] = components
        item_source = source if source is not None else row.get("source")
        if item_source:
            item["source"] = str(item_source)
        normalized.append(item)
    return sorted(normalized, key=lambda item: (int(item["frame"]), str(item.get("source", ""))))


def event_segments_from_frame_scores(
    frame_event_scores: Any,
    threshold: float = 0.0,
    max_gap: int = 1,
    min_length: int = 1,
    source: str | None = None,
) -> list[dict[str, Any]]:
    rows = normalize_frame_event_scores(frame_event_scores, source=source)
    if not rows:
        return []
    max_gap = max(int(max_gap), 1)
    min_length = max(int(min_length), 1)
    threshold = float(threshold)
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def close_current() -> None:
        nonlocal current
        if current is not None and int(current["num_frames"]) >= min_length:
            if not current.get("component_scores"):
                current.pop("component_scores", None)
            segments.append(current)
        current = None

    for row in rows:
        frame = int(row["frame"])
        score = float(row.get("score", 0.0) or 0.0)
        if score <= threshold:
            close_current()
            continue
        if current is None or frame > int(current["frame_end"]) + max_gap:
            close_current()
            current = {
                "frame_start": frame,
                "frame_end": frame,
                "score": round(score, 6),
                "dominant_reason": str(row.get("dominant_reason", "event")),
                "num_frames": 1,
                "component_scores": dict(row.get("component_scores", {})),
            }
            if row.get("source"):
                current["source"] = str(row["source"])
            continue
        current["frame_end"] = frame
        current["num_frames"] = int(current["num_frames"]) + 1
        if score > float(current["score"]):
            current["score"] = round(score, 6)
            current["dominant_reason"] = str(row.get("dominant_reason", current.get("dominant_reason", "event")))
        current_components = current.setdefault("component_scores", {})
        for key, value in (row.get("component_scores", {}) or {}).items():
            current_components[key] = max(float(current_components.get(key, 0.0)), float(value))
    close_current()
    return segments
