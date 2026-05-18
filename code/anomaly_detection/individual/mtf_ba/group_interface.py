from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

from mtf_ba.schemas import ScoreRecord, build_sample_id


MODALITIES = ("rgb", "thermal")
GROUP_FEATURE_NAMES = (
    "cx",
    "cy",
    "x",
    "y",
    "w",
    "h",
    "vx_px_per_frame",
    "vy_px_per_frame",
    "speed_px_per_frame",
)


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _track_sort_key(track_id: str) -> tuple[int, int | str]:
    return (0, int(track_id)) if str(track_id).isdigit() else (1, str(track_id))


def _parse_modality(row: dict[str, str], prefix: str) -> dict[str, Any] | None:
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


def _parse_modal_relation(row: dict[str, str]) -> dict[str, float | None] | None:
    modal = {
        "offset_dx_thermal_minus_rgb": _to_float(
            row.get("modal_offset_dx_thermal_minus_rgb")
        ),
        "offset_dy_thermal_minus_rgb": _to_float(
            row.get("modal_offset_dy_thermal_minus_rgb")
        ),
        "offset_distance": _to_float(row.get("modal_offset_distance")),
        "bbox_iou": _to_float(row.get("modal_bbox_iou")),
    }
    if all(value is None for value in modal.values()):
        return None
    return modal


def build_group_window_id(sequence: str, frame_start: int, frame_end: int) -> str:
    """Build the stable ID for one scene/window sample."""
    return f"{sequence}:{frame_start}-{frame_end}"


@dataclass(frozen=True)
class GroupWindowConfig:
    """Configuration for constructing future group-anomaly input windows."""

    sample_mode: Literal["sequence", "window"] = "window"
    window_size: int = 16
    stride: int = 8
    min_visible_frames: int = 2
    require_both_modalities: bool = False

    def __post_init__(self) -> None:
        if self.sample_mode not in {"sequence", "window"}:
            raise ValueError("sample_mode must be 'sequence' or 'window'.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if self.min_visible_frames < 0:
            raise ValueError("min_visible_frames must be non-negative.")


@dataclass
class GroupWindow:
    """
    Serializable scene/window sample for future group anomaly detectors.

    `objects[i]["states"][t]` is aligned to `frames[t]`. Each object also keeps
    the shared object-level `sample_id`, so group scores can later be fused with
    individual scores without another ID-mapping step.
    """

    window_id: str
    sequence: str
    frame_start: int
    frame_end: int
    frames: list[int]
    objects: list[dict[str, Any]]
    modalities: list[str] = field(default_factory=lambda: list(MODALITIES))
    feature_names: list[str] = field(default_factory=lambda: list(GROUP_FEATURE_NAMES))
    sample_mode: str = "window"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "sequence": self.sequence,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "frames": list(self.frames),
            "num_frames": len(self.frames),
            "num_objects": len(self.objects),
            "objects": self.objects,
            "modalities": list(self.modalities),
            "feature_names": list(self.feature_names),
            "sample_mode": self.sample_mode,
            "metadata": dict(self.metadata),
        }


@dataclass
class GroupScoreRecord:
    """
    Object-aligned score emitted by a future group anomaly detector.

    One detector may emit multiple records for the same `sample_id` if that
    object appears in multiple windows. Use `aggregate_group_scores_by_sample`
    to reduce window-level records to the standard object-level ScoreRecord.
    """

    sequence: str
    track_id: str
    window_id: str
    frame_start: int
    frame_end: int
    score: float
    category_id: int | None = None
    category_name: str | None = None
    component_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "group"

    @property
    def sample_id(self) -> str:
        return build_sample_id(self.sequence, self.track_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sequence": self.sequence,
            "track_id": self.track_id,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "source": self.source,
            "score": float(self.score),
            "window_id": self.window_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "component_scores": dict(self.component_scores),
            "metadata": dict(self.metadata),
        }


class GroupAnomalyDetector(Protocol):
    """Protocol future group anomaly implementations should satisfy."""

    def score_windows(
        self,
        windows: Iterable[GroupWindow],
    ) -> Iterable[GroupScoreRecord]:
        """Return object-aligned group anomaly scores for scene/window samples."""


def read_observations_by_sequence(
    csv_path: str | Path,
) -> dict[str, dict[int, dict[str, dict[str, str]]]]:
    """
    Read observations CSV into sequence -> frame_id -> track_id -> row.

    The CSV is generated by `extract_vt_tiny_mot_trajectories.py`.
    """
    csv_path = Path(csv_path)
    sequences: dict[str, dict[int, dict[str, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequence = row["sequence"]
            frame_id = _to_int(row.get("frame_id"))
            track_id = row["track_id"]
            if frame_id is None:
                continue
            sequences[sequence][frame_id][track_id] = row

    return {
        sequence: {frame_id: dict(track_rows) for frame_id, track_rows in frames.items()}
        for sequence, frames in sequences.items()
    }


def _build_group_window(
    sequence: str,
    frame_map: dict[int, dict[str, dict[str, str]]],
    frames: list[int],
    config: GroupWindowConfig,
) -> GroupWindow:
    track_ids = sorted(
        {
            track_id
            for frame_id in frames
            for track_id in frame_map.get(frame_id, {}).keys()
        },
        key=_track_sort_key,
    )

    objects: list[dict[str, Any]] = []
    for track_id in track_ids:
        category_id: int | None = None
        category_name: str | None = None
        visible_rgb_frames = 0
        visible_thermal_frames = 0
        visible_any_frames = 0
        states: list[dict[str, Any]] = []

        for frame_id in frames:
            row = frame_map.get(frame_id, {}).get(track_id)
            if row is None:
                states.append(
                    {
                        "frame_id": int(frame_id),
                        "rgb": None,
                        "thermal": None,
                        "modal": None,
                    }
                )
                continue

            row_category_id = _to_int(row.get("category_id"))
            if row_category_id is not None:
                category_id = row_category_id
            if row.get("category_name"):
                category_name = row["category_name"]

            rgb = _parse_modality(row, "rgb")
            thermal = _parse_modality(row, "thermal")
            modal = _parse_modal_relation(row)

            if rgb is not None:
                visible_rgb_frames += 1
            if thermal is not None:
                visible_thermal_frames += 1
            if rgb is not None or thermal is not None:
                visible_any_frames += 1

            states.append(
                {
                    "frame_id": int(frame_id),
                    "rgb": rgb,
                    "thermal": thermal,
                    "modal": modal,
                }
            )

        if visible_any_frames < config.min_visible_frames:
            continue
        if config.require_both_modalities and (
            visible_rgb_frames == 0 or visible_thermal_frames == 0
        ):
            continue

        objects.append(
            {
                "sample_id": build_sample_id(sequence, track_id),
                "sequence": sequence,
                "track_id": track_id,
                "category_id": category_id,
                "category_name": category_name,
                "visible_rgb_frames": visible_rgb_frames,
                "visible_thermal_frames": visible_thermal_frames,
                "visible_any_frames": visible_any_frames,
                "states": states,
            }
        )

    frame_start = int(frames[0])
    frame_end = int(frames[-1])
    return GroupWindow(
        window_id=build_group_window_id(sequence, frame_start, frame_end),
        sequence=sequence,
        frame_start=frame_start,
        frame_end=frame_end,
        frames=[int(frame_id) for frame_id in frames],
        objects=objects,
        sample_mode=config.sample_mode,
        metadata={
            "window_size": config.window_size,
            "stride": config.stride,
            "min_visible_frames": config.min_visible_frames,
            "require_both_modalities": config.require_both_modalities,
        },
    )


def iter_group_windows(
    csv_path: str | Path,
    config: GroupWindowConfig | None = None,
    sequence: str | None = None,
) -> Iterable[GroupWindow]:
    """Yield group-anomaly input windows from an observations CSV."""
    config = config or GroupWindowConfig()
    sequences = read_observations_by_sequence(csv_path)
    if sequence is not None:
        sequences = {
            name: frame_map
            for name, frame_map in sequences.items()
            if name == sequence
        }

    for sequence_name, frame_map in sorted(sequences.items()):
        sequence_frames = sorted(frame_map)
        if not sequence_frames:
            continue

        if config.sample_mode == "sequence":
            yield _build_group_window(
                sequence=sequence_name,
                frame_map=frame_map,
                frames=sequence_frames,
                config=config,
            )
            continue

        if len(sequence_frames) < config.window_size:
            continue
        max_start = len(sequence_frames) - config.window_size
        for start_pos in range(0, max_start + 1, config.stride):
            frames = sequence_frames[start_pos : start_pos + config.window_size]
            yield _build_group_window(
                sequence=sequence_name,
                frame_map=frame_map,
                frames=frames,
                config=config,
            )


def build_group_windows(
    csv_path: str | Path,
    config: GroupWindowConfig | None = None,
    sequence: str | None = None,
) -> list[GroupWindow]:
    """Materialize all group windows as a list."""
    return list(iter_group_windows(csv_path=csv_path, config=config, sequence=sequence))


def write_group_windows_jsonl(
    path: str | Path,
    windows: Iterable[GroupWindow],
) -> dict[str, Any]:
    """Write group windows to JSONL and return basic export stats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_windows = 0
    total_objects = 0
    with path.open("w", encoding="utf-8") as f:
        for window in windows:
            payload = window.to_dict()
            total_objects += int(payload["num_objects"])
            num_windows += 1
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")

    return {
        "output_jsonl": str(path),
        "num_windows": num_windows,
        "total_objects": total_objects,
        "avg_objects_per_window": (
            total_objects / num_windows if num_windows > 0 else 0.0
        ),
    }


def iter_group_window_jsonl(jsonl_path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield serialized group windows from JSONL."""
    jsonl_path = Path(jsonl_path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _record_to_dict(record: GroupScoreRecord | ScoreRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    raise TypeError(f"Unsupported score record type: {type(record)!r}")


def write_group_score_records_jsonl(
    path: str | Path,
    records: Iterable[GroupScoreRecord | ScoreRecord | dict[str, Any]],
) -> int:
    """Write future group detector scores to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_record_to_dict(record), ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def load_group_score_records_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load group score JSONL records as dictionaries."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate_group_scores_by_sample(
    records: Iterable[GroupScoreRecord | dict[str, Any]],
    method: Literal["max", "mean"] = "max",
) -> list[ScoreRecord]:
    """
    Reduce window-level group scores to one standard ScoreRecord per sample_id.

    This is the bridge from future group anomaly output into the existing
    object-level fusion contract.
    """
    if method not in {"max", "mean"}:
        raise ValueError("method must be 'max' or 'mean'.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        payload = record if isinstance(record, dict) else record.to_dict()
        grouped[payload["sample_id"]].append(payload)

    aggregated: list[ScoreRecord] = []
    for sample_id, items in sorted(grouped.items()):
        scores = [float(item["score"]) for item in items]
        score = max(scores) if method == "max" else sum(scores) / len(scores)
        first = items[0]
        aggregated.append(
            ScoreRecord(
                sequence=str(first["sequence"]),
                track_id=str(first["track_id"]),
                source="group",
                score=float(score),
                category_id=first.get("category_id"),
                category_name=first.get("category_name"),
                component_scores={
                    "group_window_max": float(max(scores)),
                    "group_window_mean": float(sum(scores) / len(scores)),
                },
                metadata={
                    "aggregation": method,
                    "num_group_windows": len(items),
                    "window_ids": [item.get("window_id") for item in items],
                    "sample_id": sample_id,
                },
            )
        )
    return aggregated
