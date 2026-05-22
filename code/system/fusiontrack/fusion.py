from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModalityState:
    center_xy: tuple[float, float]
    bbox_xywh: tuple[float | None, float | None, float | None, float | None]
    speed_px_per_frame: float | None = None
    file: str | None = None


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _parse_modality(row: dict[str, str], prefix: str) -> ModalityState | None:
    cx = _to_float(row.get(f"{prefix}_cx"))
    cy = _to_float(row.get(f"{prefix}_cy"))
    if cx is None or cy is None:
        return None
    return ModalityState(
        center_xy=(cx, cy),
        bbox_xywh=(
            _to_float(row.get(f"{prefix}_x")),
            _to_float(row.get(f"{prefix}_y")),
            _to_float(row.get(f"{prefix}_w")),
            _to_float(row.get(f"{prefix}_h")),
        ),
        speed_px_per_frame=_to_float(row.get(f"{prefix}_speed_px_per_frame")),
        file=row.get(f"{prefix}_file") or None,
    )


def _state_to_dict(state: ModalityState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "file": state.file,
        "bbox_xywh": list(state.bbox_xywh),
        "center_xy": [state.center_xy[0], state.center_xy[1]],
        "speed_px_per_frame": state.speed_px_per_frame,
    }


def _modal_relation(row: dict[str, str]) -> dict[str, float | None]:
    return {
        "offset_dx_thermal_minus_rgb": _to_float(row.get("modal_offset_dx_thermal_minus_rgb")),
        "offset_dy_thermal_minus_rgb": _to_float(row.get("modal_offset_dy_thermal_minus_rgb")),
        "offset_distance": _to_float(row.get("modal_offset_distance")),
        "bbox_iou": _to_float(row.get("modal_bbox_iou")),
    }


def fuse_centers(
    rgb: ModalityState | None,
    thermal: ModalityState | None,
) -> tuple[tuple[float, float], dict[str, float]]:
    if rgb is None and thermal is None:
        raise ValueError("At least one modality state is required.")

    if rgb is not None and thermal is not None:
        rx, ry = rgb.center_xy
        tx, ty = thermal.center_xy
        offset = math.hypot(tx - rx, ty - ry)
        confidence = 1.0 / (1.0 + offset / 25.0)
        return (
            ((rx + tx) / 2.0, (ry + ty) / 2.0),
            {
                "confidence": confidence,
                "num_modalities": 2.0,
                "modal_offset_distance": offset,
            },
        )

    state = rgb if rgb is not None else thermal
    assert state is not None
    return (
        state.center_xy,
        {
            "confidence": 0.55,
            "num_modalities": 1.0,
            "modal_offset_distance": 0.0,
        },
    )


def fuse_observations_csv(
    csv_path: str | Path,
    output_jsonl: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    output_jsonl = Path(output_jsonl)
    output_csv = Path(output_csv)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    flat_rows: list[dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence = row["sequence"]
            track_id = row["track_id"]
            key = (sequence, track_id)
            rgb = _parse_modality(row, "rgb")
            thermal = _parse_modality(row, "thermal")
            if rgb is None and thermal is None:
                continue

            center, components = fuse_centers(rgb, thermal)
            source_modalities = []
            if rgb is not None:
                source_modalities.append("rgb")
            if thermal is not None:
                source_modalities.append("thermal")

            frame_id = _to_int(row.get("frame_id"))
            point = {
                "frame_id": frame_id,
                "rgb": _state_to_dict(rgb),
                "thermal": _state_to_dict(thermal),
                "modal": _modal_relation(row),
                "fused": {
                    "center_xy": [center[0], center[1]],
                    "confidence": components["confidence"],
                    "source_modalities": source_modalities,
                    "component_scores": dict(components),
                },
            }

            if key not in grouped:
                grouped[key] = {
                    "sample_id": f"{sequence}:{track_id}",
                    "sequence": sequence,
                    "track_id": track_id,
                    "category_id": _to_int(row.get("category_id")),
                    "category_name": row.get("category_name") or None,
                    "fps": _to_float(row.get("fps")),
                    "points": [],
                }
            grouped[key]["points"].append(point)
            flat_rows.append(
                {
                    "sequence": sequence,
                    "track_id": track_id,
                    "category_id": grouped[key]["category_id"],
                    "category_name": grouped[key]["category_name"],
                    "frame_id": frame_id,
                    "fused_cx": center[0],
                    "fused_cy": center[1],
                    "confidence": components["confidence"],
                    "num_modalities": components["num_modalities"],
                    "modal_offset_distance": components["modal_offset_distance"],
                    "source_modalities": "|".join(source_modalities),
                }
            )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for key in sorted(grouped):
            payload = grouped[key]
            payload["points"] = sorted(
                payload["points"],
                key=lambda item: -1 if item["frame_id"] is None else item["frame_id"],
            )
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")

    fieldnames = [
        "sequence",
        "track_id",
        "category_id",
        "category_name",
        "frame_id",
        "fused_cx",
        "fused_cy",
        "confidence",
        "num_modalities",
        "modal_offset_distance",
        "source_modalities",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    return {
        "input_csv": str(csv_path),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "num_fused_trajectories": len(grouped),
        "num_fused_states": len(flat_rows),
    }
