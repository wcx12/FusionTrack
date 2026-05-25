from __future__ import annotations

import math
from typing import Any, Iterable

from mtf_ba.schemas import build_sample_id


MODALITIES = ("rgb", "thermal")
MODALITY_STATE_FIELDS = (
    "file",
    "bbox_xywh",
    "center_xy",
    "confidence",
    "visibility",
    "velocity_px_per_frame",
    "speed_px_per_frame",
    "velocity_px_per_second",
    "speed_px_per_second",
)
MODAL_RELATION_FIELDS = (
    "offset_dx_thermal_minus_rgb",
    "offset_dy_thermal_minus_rgb",
    "offset_distance",
    "bbox_iou",
)


def to_float(value: Any, field_name: str = "value") -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field_name}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Invalid finite numeric value for {field_name}: {value!r}")
    return parsed


def to_int(value: Any, field_name: str = "value") -> int | None:
    parsed = to_float(value, field_name=field_name)
    if parsed is None:
        return None
    return int(parsed)


def parse_modality_state(row: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    """Parse one modality from a flat observations CSV row.

    A modality is considered available only when both center coordinates are
    present. Other fields are still normalized to the same keys when available.
    """
    cx = to_float(row.get(f"{prefix}_cx"), f"{prefix}_cx")
    cy = to_float(row.get(f"{prefix}_cy"), f"{prefix}_cy")
    if cx is None or cy is None:
        return None

    return {
        "file": row.get(f"{prefix}_file") or None,
        "bbox_xywh": [
            to_float(row.get(f"{prefix}_x"), f"{prefix}_x"),
            to_float(row.get(f"{prefix}_y"), f"{prefix}_y"),
            to_float(row.get(f"{prefix}_w"), f"{prefix}_w"),
            to_float(row.get(f"{prefix}_h"), f"{prefix}_h"),
        ],
        "center_xy": [cx, cy],
        "confidence": to_float(row.get(f"{prefix}_confidence"), f"{prefix}_confidence"),
        "visibility": to_float(row.get(f"{prefix}_visibility"), f"{prefix}_visibility"),
        "velocity_px_per_frame": [
            to_float(row.get(f"{prefix}_vx_px_per_frame"), f"{prefix}_vx_px_per_frame"),
            to_float(row.get(f"{prefix}_vy_px_per_frame"), f"{prefix}_vy_px_per_frame"),
        ],
        "speed_px_per_frame": to_float(
            row.get(f"{prefix}_speed_px_per_frame"),
            f"{prefix}_speed_px_per_frame",
        ),
        "velocity_px_per_second": [
            to_float(row.get(f"{prefix}_vx_px_per_second"), f"{prefix}_vx_px_per_second"),
            to_float(row.get(f"{prefix}_vy_px_per_second"), f"{prefix}_vy_px_per_second"),
        ],
        "speed_px_per_second": to_float(
            row.get(f"{prefix}_speed_px_per_second"),
            f"{prefix}_speed_px_per_second",
        ),
    }


def parse_modal_relation(
    row: dict[str, Any],
    *,
    keep_empty: bool = False,
) -> dict[str, float | None] | None:
    relation = {
        "offset_dx_thermal_minus_rgb": to_float(
            row.get("modal_offset_dx_thermal_minus_rgb"),
            "modal_offset_dx_thermal_minus_rgb",
        ),
        "offset_dy_thermal_minus_rgb": to_float(
            row.get("modal_offset_dy_thermal_minus_rgb"),
            "modal_offset_dy_thermal_minus_rgb",
        ),
        "offset_distance": to_float(row.get("modal_offset_distance"), "modal_offset_distance"),
        "bbox_iou": to_float(row.get("modal_bbox_iou"), "modal_bbox_iou"),
    }
    if all(value is None for value in relation.values()):
        inferred = _infer_modal_relation(row)
        if inferred is not None:
            relation = inferred
    if all(value is None for value in relation.values()) and not keep_empty:
        return None
    return relation


def standardize_observation_row(row: dict[str, Any]) -> dict[str, Any]:
    sequence = str(row.get("sequence") or "")
    track_id = str(row.get("track_id") or "")
    standardized_modalities = {
        modality: _standardize_modality(row, modality) for modality in MODALITIES
    }
    modal_relation = parse_modal_relation(row)
    available_modalities = [
        modality
        for modality, state in standardized_modalities.items()
        if bool(state["available"])
    ]
    missing_modalities = [
        modality
        for modality, state in standardized_modalities.items()
        if not bool(state["available"])
    ]

    return {
        "schema_version": 1,
        "dataset": row.get("dataset") or None,
        "sequence": sequence,
        "track_id": track_id,
        "sample_id": build_sample_id(sequence, track_id),
        "frame_id": to_int(row.get("frame_id"), "frame_id"),
        "category_id": to_int(row.get("category_id"), "category_id"),
        "category_name": row.get("category_name") or None,
        "fps": to_float(row.get("fps"), "fps"),
        "modalities": standardized_modalities,
        "modal_relation": {
            "available": modal_relation is not None,
            **_empty_modal_relation(),
            **(modal_relation or {}),
        },
        "quality": {
            "num_available_modalities": len(available_modalities),
            "available_modalities": available_modalities,
            "missing_modalities": missing_modalities,
            "has_cross_modal_relation": modal_relation is not None,
        },
    }


def standardize_observation_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    standardized = [standardize_observation_row(row) for row in rows]
    return sorted(
        standardized,
        key=lambda item: (
            str(item["sequence"]),
            _track_sort_key(str(item["track_id"])),
            -1 if item["frame_id"] is None else int(item["frame_id"]),
        ),
    )


def point_from_observation_row(
    row: dict[str, Any],
    *,
    keep_empty_modal_relation: bool = True,
) -> dict[str, Any]:
    return {
        "frame_id": to_int(row.get("frame_id"), "frame_id"),
        "rgb": parse_modality_state(row, "rgb"),
        "thermal": parse_modality_state(row, "thermal"),
        "modal": parse_modal_relation(row, keep_empty=keep_empty_modal_relation),
    }


def _standardize_modality(row: dict[str, Any], modality: str) -> dict[str, Any]:
    state = parse_modality_state(row, modality)
    if state is None:
        return {"available": False, **_empty_modality_state()}
    return {"available": True, **state}


def _empty_modality_state() -> dict[str, Any]:
    return {
        "file": None,
        "bbox_xywh": None,
        "center_xy": None,
        "confidence": None,
        "visibility": None,
        "velocity_px_per_frame": None,
        "speed_px_per_frame": None,
        "velocity_px_per_second": None,
        "speed_px_per_second": None,
    }


def _empty_modal_relation() -> dict[str, float | None]:
    return {field: None for field in MODAL_RELATION_FIELDS}


def _infer_modal_relation(row: dict[str, Any]) -> dict[str, float | None] | None:
    rgb = parse_modality_state(row, "rgb")
    thermal = parse_modality_state(row, "thermal")
    if rgb is None or thermal is None:
        return None
    rgb_center = rgb["center_xy"]
    thermal_center = thermal["center_xy"]
    dx = float(thermal_center[0]) - float(rgb_center[0])
    dy = float(thermal_center[1]) - float(rgb_center[1])
    return {
        "offset_dx_thermal_minus_rgb": dx,
        "offset_dy_thermal_minus_rgb": dy,
        "offset_distance": math.hypot(dx, dy),
        "bbox_iou": _bbox_iou(rgb.get("bbox_xywh"), thermal.get("bbox_xywh")),
    }


def _bbox_iou(left: list[float | None] | None, right: list[float | None] | None) -> float | None:
    if left is None or right is None or any(value is None for value in [*left, *right]):
        return None
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    lx2, ly2 = lx + lw, ly + lh
    rx2, ry2 = rx + rw, ry + rh
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - inter
    return inter / union if union > 0.0 else None


def _track_sort_key(track_id: str) -> tuple[int, int | str]:
    return (0, int(track_id)) if str(track_id).isdigit() else (1, str(track_id))
