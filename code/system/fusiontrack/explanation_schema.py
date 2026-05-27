from __future__ import annotations

import math
from typing import Any

from fusiontrack.event_segments import event_segments_from_frame_scores


def build_explanation_schema(
    row: dict[str, Any],
    *,
    threshold: float = 0.0,
    max_gap: int = 1,
    min_length: int = 1,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build a stable backend explanation payload for score rows."""

    components = _rank_components(row.get("component_scores"), top_k=top_k)
    event_segments = _event_segments(row, threshold=threshold, max_gap=max_gap, min_length=min_length)
    peak_event = _peak_event(event_segments)
    if peak_event is not None:
        top_reason = str(peak_event.get("dominant_reason") or "event")
        evidence_source = _event_source(row)
    elif components:
        top_reason = components[0]["name"]
        evidence_source = "component_scores"
    else:
        top_reason = "score"
        evidence_source = "score"

    return {
        "schema_version": 1,
        "top_reason": top_reason,
        "evidence_source": evidence_source,
        "policy": {
            "event_threshold": float(threshold),
            "max_gap": int(max_gap),
            "min_length": int(min_length),
        },
        "score": _finite_float(row.get("score"), default=0.0),
        "event_score": _finite_float(row.get("event_score"), default=0.0),
        "peak_event": peak_event,
        "score_components": components,
    }


def _event_segments(
    row: dict[str, Any],
    *,
    threshold: float,
    max_gap: int,
    min_length: int,
) -> list[dict[str, Any]]:
    raw_segments = row.get("event_segments")
    if isinstance(raw_segments, list) and raw_segments:
        segments = [dict(segment) for segment in raw_segments if isinstance(segment, dict)]
    else:
        segments = event_segments_from_frame_scores(
            row.get("frame_event_scores", []),
            threshold=threshold,
            max_gap=max_gap,
            min_length=min_length,
        )
    return [
        segment
        for segment in segments
        if _finite_float(segment.get("score"), default=float("-inf")) >= threshold
    ]


def _event_source(row: dict[str, Any]) -> str:
    raw_segments = row.get("event_segments")
    if isinstance(raw_segments, list) and raw_segments:
        return "event_segments"
    raw_frames = row.get("frame_event_scores")
    if isinstance(raw_frames, list) and raw_frames:
        return "frame_event_scores"
    return "event"


def _peak_event(event_segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not event_segments:
        return None
    peak = max(
        event_segments,
        key=lambda segment: (
            _finite_float(segment.get("score"), default=float("-inf")),
            -int(_finite_float(segment.get("frame_start"), default=0)),
        ),
    )
    return {
        "frame_start": _int_or_none(peak.get("frame_start")),
        "frame_end": _int_or_none(peak.get("frame_end")),
        "score": round(_finite_float(peak.get("score"), default=0.0), 6),
        "dominant_reason": str(peak.get("dominant_reason") or "event"),
        "num_frames": _int_or_none(peak.get("num_frames")),
        "source": None if peak.get("source") is None else str(peak.get("source")),
    }


def _rank_components(component_scores: Any, *, top_k: int) -> list[dict[str, Any]]:
    if not isinstance(component_scores, dict):
        return []
    rows = []
    for name, value in component_scores.items():
        score = _finite_float(value, default=None)
        if score is None:
            continue
        rows.append(
            {
                "name": str(name),
                "value": round(score, 6),
                "family": _component_family(str(name)),
            }
        )
    primary = [item for item in rows if not item["name"].startswith("S_")]
    primary.sort(key=lambda item: (-abs(float(item["value"])), item["name"]))
    aggregate = [item for item in rows if item["name"].startswith("S_")]
    aggregate.sort(key=lambda item: (0 if item["name"] == "S_event" else 1, -abs(float(item["value"])), item["name"]))
    ordered = primary[:1] + aggregate[:1] + primary[1:] + aggregate[1:]
    return ordered[: max(int(top_k), 0)]


def _component_family(name: str) -> str:
    if name.startswith("individual_") or name in {"route_score", "speed_score", "shape_score"}:
        return "individual"
    if name.startswith("group_") or name.startswith("graph_"):
        return "group"
    if name.startswith("registration_"):
        return "registration"
    if name.startswith("S_"):
        return "fusion"
    return "other"


def _finite_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _int_or_none(value: Any) -> int | None:
    number = _finite_float(value, default=None)
    return None if number is None else int(number)
