from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from mtf_ba.schemas import ObjectIdentity

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):  # type: ignore[misc]
        return iterable


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _parse_modality(row: dict[str, str], prefix: str) -> dict[str, Any] | None:
    """
    Parse one modality state from a flat CSV row.

    `prefix` is typically `"rgb"` or `"thermal"`.

    Returned fields are per-frame observations for that modality only:
    - bounding box
    - box center
    - velocity / speed

    If the modality is not visible in this frame, return None instead of a
    partially filled structure so downstream code can explicitly reason about
    modality presence/absence.
    """
    cx = _to_float(row.get(f"{prefix}_cx"))
    cy = _to_float(row.get(f"{prefix}_cy"))
    if cx is None or cy is None:
        return None

    return {
        "file": row.get(f"{prefix}_file") or None,
        "bbox_xywh": [
            _to_float(row.get(f"{prefix}_x")),
            _to_float(row.get(f"{prefix}_y")),
            _to_float(row.get(f"{prefix}_w")),
            _to_float(row.get(f"{prefix}_h")),
        ],
        "center_xy": [cx, cy],
        "velocity_px_per_frame": [
            _to_float(row.get(f"{prefix}_vx_px_per_frame")),
            _to_float(row.get(f"{prefix}_vy_px_per_frame")),
        ],
        "speed_px_per_frame": _to_float(row.get(f"{prefix}_speed_px_per_frame")),
        "velocity_px_per_second": [
            _to_float(row.get(f"{prefix}_vx_px_per_second")),
            _to_float(row.get(f"{prefix}_vy_px_per_second")),
        ],
        "speed_px_per_second": _to_float(row.get(f"{prefix}_speed_px_per_second")),
    }


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def load_object_trajectories(
    csv_path: str | Path,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Convert observations CSV into object-centric trajectories keyed by sample_id.

    Output item shape:
      {
        "sample_id": "sequence:track_id",
        "sequence": str,
        "track_id": str,
        "category_id": int | None,
        "category_name": str | None,
        "fps": float | None,
        "num_points": int,
        "visible_rgb_frames": int,
        "visible_thermal_frames": int,
        "points": [
          {
            "frame_id": int,
            "rgb": {...} | None,
            "thermal": {...} | None,
            "modal": {...}
          }
        ]
      }

    Semantics of the three point-level branches:

    - `rgb`:
      RGB-modality observation of this object at the current frame.
      Contains position, size, and motion measured in the RGB stream only.

    - `thermal`:
      Thermal-modality observation of this object at the current frame.
      Contains position, size, and motion measured in the thermal stream only.

    - `modal`:
      Cross-modal relation features for the same object at the same frame.
      This is not a separate modality; it describes how different RGB and
      thermal are from each other, for example center offset and box overlap.
    """
    csv_path = Path(csv_path)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    total_rows = _count_csv_rows(csv_path) if show_progress else None

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row_iter = tqdm(
            reader,
            total=total_rows,
            desc="Reading observations",
            unit="row",
            disable=not show_progress,
        )
        for row in row_iter:
            sequence = row["sequence"]
            track_id = row["track_id"]
            key = (sequence, track_id)

            point = {
                "frame_id": _to_int(row.get("frame_id")),
                "rgb": _parse_modality(row, "rgb"),
                "thermal": _parse_modality(row, "thermal"),
                "modal": {
                    # Thermal-center minus RGB-center displacement. These two
                    # values preserve direction, unlike offset_distance.
                    "offset_dx_thermal_minus_rgb": _to_float(
                        row.get("modal_offset_dx_thermal_minus_rgb")
                    ),
                    "offset_dy_thermal_minus_rgb": _to_float(
                        row.get("modal_offset_dy_thermal_minus_rgb")
                    ),
                    # Euclidean distance between RGB and thermal centers for
                    # the same object at this frame.
                    "offset_distance": _to_float(row.get("modal_offset_distance")),
                    # Spatial overlap between RGB and thermal boxes. Higher is
                    # more consistent across modalities.
                    "bbox_iou": _to_float(row.get("modal_bbox_iou")),
                },
            }
            grouped[key].append(point)

            if key not in metadata:
                identity = ObjectIdentity(
                    sequence=sequence,
                    track_id=track_id,
                    category_id=_to_int(row.get("category_id")),
                    category_name=row.get("category_name") or None,
                )
                metadata[key] = {
                    "sample_id": identity.sample_id,
                    "sequence": sequence,
                    "track_id": track_id,
                    "category_id": identity.category_id,
                    "category_name": identity.category_name,
                    "fps": _to_float(row.get("fps")),
                }

    trajectories: list[dict[str, Any]] = []
    key_iter = tqdm(
        sorted(grouped),
        desc="Building trajectories",
        unit="trajectory",
        disable=not show_progress,
    )
    for key in key_iter:
        points = sorted(
            grouped[key],
            key=lambda item: (-1 if item["frame_id"] is None else item["frame_id"]),
        )
        # Visibility counts are useful later when deciding whether a trajectory
        # is reliable enough to build RGB-only, thermal-only, or cross-modal
        # features from it.
        visible_rgb_frames = sum(1 for point in points if point["rgb"] is not None)
        visible_thermal_frames = sum(
            1 for point in points if point["thermal"] is not None
        )

        trajectory = {
            **metadata[key],
            "num_points": len(points),
            "visible_rgb_frames": visible_rgb_frames,
            "visible_thermal_frames": visible_thermal_frames,
            "points": points,
        }
        trajectories.append(trajectory)

    return trajectories
