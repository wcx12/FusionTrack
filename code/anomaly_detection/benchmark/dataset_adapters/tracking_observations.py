from __future__ import annotations

import configparser
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MODALITIES = ("rgb", "thermal")

BASE_COLUMNS = (
    "dataset",
    "sequence",
    "track_id",
    "category_id",
    "category_name",
    "fps",
    "frame_id",
)
MODALITY_VALUE_COLUMNS = (
    "file",
    "x",
    "y",
    "w",
    "h",
    "cx",
    "cy",
    "confidence",
    "visibility",
    "vx_px_per_frame",
    "vy_px_per_frame",
    "speed_px_per_frame",
    "vx_px_per_second",
    "vy_px_per_second",
    "speed_px_per_second",
)
MODAL_COLUMNS = (
    "modal_offset_dx_thermal_minus_rgb",
    "modal_offset_dy_thermal_minus_rgb",
    "modal_offset_distance",
    "modal_bbox_iou",
)
OBSERVATION_COLUMNS = (
    *BASE_COLUMNS,
    *(f"{modality}_{column}" for modality in MODALITIES for column in MODALITY_VALUE_COLUMNS),
    *MODAL_COLUMNS,
)


CATEGORY_NAMES_BY_PROFILE = {
    "motchallenge": {1: "pedestrian"},
    "m3ot": {1: "vehicle"},
    "dancetrack": {1: "person"},
    "sportsmot": {1: "player"},
}


@dataclass(frozen=True)
class MotSequence:
    sequence: str
    gt_path: Path
    sequence_dir: Path
    fps: float | None = None
    image_dir: str | None = None
    image_ext: str | None = None


@dataclass(frozen=True)
class MotObservation:
    dataset: str
    sequence: str
    track_id: str
    frame_id: int
    bbox_xywh: tuple[float, float, float, float]
    category_id: int | None
    category_name: str | None
    confidence: float | None
    visibility: float | None
    fps: float | None
    file: str | None = None

    @property
    def center_xy(self) -> tuple[float, float]:
        x, y, w, h = self.bbox_xywh
        return x + w / 2.0, y + h / 2.0


def convert_mot_roots_to_observations(
    output_csv: str | Path,
    dataset: str,
    rgb_root: str | Path | None = None,
    thermal_root: str | Path | None = None,
    profile: str = "motchallenge",
    split: str | None = None,
    fps: float | None = None,
    frame_digits: int | None = None,
    keep_category_ids: Sequence[int] | None = None,
    use_default_category_filter: bool = True,
    include_ignored: bool = False,
    sequences: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Convert one or two MOT-style roots into FusionTrack observations CSV.

    `rgb_root` covers MOT17/MOT20, DanceTrack, and SportsMOT. Passing both
    `rgb_root` and `thermal_root` covers M3OT-style paired RGB/IR exports.
    """
    if rgb_root is None and thermal_root is None:
        raise ValueError("At least one of rgb_root or thermal_root is required.")

    points_by_modality: dict[str, list[MotObservation]] = {}
    sequence_filter = set(sequences or [])

    if rgb_root is not None:
        points_by_modality["rgb"] = load_mot_root(
            rgb_root,
            dataset=dataset,
            profile=profile,
            split=split,
            fps=fps,
            frame_digits=frame_digits,
            keep_category_ids=keep_category_ids,
            use_default_category_filter=use_default_category_filter,
            include_ignored=include_ignored,
            sequences=sequence_filter,
        )
    if thermal_root is not None:
        points_by_modality["thermal"] = load_mot_root(
            thermal_root,
            dataset=dataset,
            profile=profile,
            split=split,
            fps=fps,
            frame_digits=frame_digits,
            keep_category_ids=keep_category_ids,
            use_default_category_filter=use_default_category_filter,
            include_ignored=include_ignored,
            sequences=sequence_filter,
        )

    rows = build_observation_rows(points_by_modality)
    write_observations_csv(output_csv, rows)
    return conversion_summary(output_csv, rows, points_by_modality, dataset, profile, split)


def load_mot_root(
    root: str | Path,
    dataset: str,
    profile: str = "motchallenge",
    split: str | None = None,
    fps: float | None = None,
    frame_digits: int | None = None,
    keep_category_ids: Sequence[int] | None = None,
    use_default_category_filter: bool = True,
    include_ignored: bool = False,
    sequences: set[str] | None = None,
) -> list[MotObservation]:
    root_path = Path(root)
    mot_sequences = discover_mot_sequences(root_path, split=split)
    if sequences:
        mot_sequences = [sequence for sequence in mot_sequences if sequence.sequence in sequences]
    observations: list[MotObservation] = []
    for sequence in mot_sequences:
        observations.extend(
            read_mot_sequence(
                sequence,
                dataset=dataset,
                profile=profile,
                fps_override=fps,
                frame_digits=frame_digits if frame_digits is not None else _default_frame_digits(profile),
                keep_category_ids=(
                    keep_category_ids
                    if keep_category_ids is not None
                    else _default_keep_category_ids(profile, use_default_category_filter)
                ),
                include_ignored=include_ignored,
            )
        )
    return sorted(observations, key=lambda item: (item.sequence, _track_sort_key(item.track_id), item.frame_id))


def discover_mot_sequences(root: str | Path, split: str | None = None) -> list[MotSequence]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"MOT root does not exist: {root_path}")

    candidates: list[Path] = []
    direct = root_path / "gt" / "gt.txt"
    if direct.exists():
        candidates.append(direct)
    candidates.extend(
        path
        for path in root_path.rglob("gt.txt")
        if path.parent.name.lower() == "gt" and path not in candidates
    )

    sequences: list[MotSequence] = []
    for gt_path in sorted(candidates):
        sequence_dir = gt_path.parent.parent
        if split and split not in sequence_dir.parts:
            continue
        seqinfo = read_seqinfo(sequence_dir / "seqinfo.ini")
        sequence_name = str(seqinfo.get("name") or sequence_dir.name)
        sequences.append(
            MotSequence(
                sequence=sequence_name,
                gt_path=gt_path,
                sequence_dir=sequence_dir,
                fps=_optional_float(seqinfo.get("fps") or seqinfo.get("framerate")),
                image_dir=_optional_str(seqinfo.get("imDir") or seqinfo.get("imdir")),
                image_ext=_optional_str(seqinfo.get("imExt") or seqinfo.get("imext")),
            )
        )
    if not sequences:
        split_note = f" for split {split!r}" if split else ""
        raise FileNotFoundError(f"No MOT gt/gt.txt files found under {root_path}{split_note}.")
    return sequences


def read_seqinfo(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if not parser.has_section("Sequence"):
        return {}
    return {key: value for key, value in parser.items("Sequence")}


def read_mot_sequence(
    sequence: MotSequence,
    dataset: str,
    profile: str = "motchallenge",
    fps_override: float | None = None,
    frame_digits: int = 6,
    keep_category_ids: Sequence[int] | None = None,
    include_ignored: bool = False,
) -> list[MotObservation]:
    observations: list[MotObservation] = []
    with sequence.gt_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(line for line in f if line.strip() and not line.lstrip().startswith("#"))
        for row in reader:
            if len(row) < 7:
                raise ValueError(f"Expected at least 7 MOT columns in {sequence.gt_path}: {row}")
            frame_id = _required_int(row[0], "frame_id")
            track_id = str(_required_int(row[1], "track_id"))
            bbox = (
                _required_float(row[2], "bbox_left"),
                _required_float(row[3], "bbox_top"),
                _required_float(row[4], "bbox_width"),
                _required_float(row[5], "bbox_height"),
            )
            confidence = _optional_float(row[6])
            if not include_ignored and confidence is not None and confidence <= 0.0:
                continue

            category_id = _category_id(row, profile)
            if (
                keep_category_ids is not None
                and category_id is not None
                and category_id not in set(keep_category_ids)
            ):
                continue
            visibility = _visibility(row, profile)
            observations.append(
                MotObservation(
                    dataset=dataset,
                    sequence=sequence.sequence,
                    track_id=track_id,
                    frame_id=frame_id,
                    bbox_xywh=bbox,
                    category_id=category_id,
                    category_name=_category_name(profile, category_id),
                    confidence=confidence,
                    visibility=visibility,
                    fps=fps_override if fps_override is not None else sequence.fps,
                    file=_image_file(sequence, frame_id, frame_digits=frame_digits),
                )
            )
    return observations


def build_observation_rows(
    points_by_modality: dict[str, list[MotObservation]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, MotObservation]] = {}
    for modality, observations in points_by_modality.items():
        if modality not in MODALITIES:
            raise ValueError(f"Unsupported modality: {modality}")
        for observation in observations:
            key = (observation.sequence, observation.track_id, observation.frame_id)
            grouped.setdefault(key, {})[modality] = observation

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0], _track_sort_key(item[1]), item[2])):
        modalities = grouped[key]
        representative = modalities.get("rgb") or modalities.get("thermal")
        if representative is None:
            continue
        row = _empty_row(representative)
        for modality, observation in modalities.items():
            _fill_modality(row, modality, observation)
        _fill_modal_relation(row)
        rows.append(row)

    _add_temporal_features(rows)
    return rows


def write_observations_csv(output_csv: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(OBSERVATION_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in OBSERVATION_COLUMNS})


def write_summary_json(summary_path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def conversion_summary(
    output_csv: str | Path,
    rows: list[dict[str, Any]],
    points_by_modality: dict[str, list[MotObservation]],
    dataset: str,
    profile: str,
    split: str | None,
) -> dict[str, Any]:
    sequence_names = sorted({row["sequence"] for row in rows})
    track_keys = {(row["sequence"], row["track_id"]) for row in rows}
    return {
        "dataset": dataset,
        "profile": profile,
        "split": split,
        "output_csv": str(Path(output_csv)),
        "num_rows": len(rows),
        "num_sequences": len(sequence_names),
        "sequences": sequence_names,
        "num_tracks": len(track_keys),
        "modalities": {
            modality: {
                "num_observations": len(observations),
                "num_sequences": len({item.sequence for item in observations}),
                "num_tracks": len({(item.sequence, item.track_id) for item in observations}),
            }
            for modality, observations in sorted(points_by_modality.items())
        },
    }


def _empty_row(observation: MotObservation) -> dict[str, Any]:
    row = {column: None for column in OBSERVATION_COLUMNS}
    row.update(
        {
            "dataset": observation.dataset,
            "sequence": observation.sequence,
            "track_id": observation.track_id,
            "category_id": observation.category_id,
            "category_name": observation.category_name,
            "fps": observation.fps,
            "frame_id": observation.frame_id,
        }
    )
    return row


def _fill_modality(row: dict[str, Any], modality: str, observation: MotObservation) -> None:
    x, y, w, h = observation.bbox_xywh
    cx, cy = observation.center_xy
    values = {
        "file": observation.file,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cx": cx,
        "cy": cy,
        "confidence": observation.confidence,
        "visibility": observation.visibility,
    }
    for key, value in values.items():
        row[f"{modality}_{key}"] = value


def _fill_modal_relation(row: dict[str, Any]) -> None:
    rgb_center = _center(row, "rgb")
    thermal_center = _center(row, "thermal")
    if rgb_center is None or thermal_center is None:
        return
    dx = thermal_center[0] - rgb_center[0]
    dy = thermal_center[1] - rgb_center[1]
    row["modal_offset_dx_thermal_minus_rgb"] = dx
    row["modal_offset_dy_thermal_minus_rgb"] = dy
    row["modal_offset_distance"] = math.hypot(dx, dy)
    row["modal_bbox_iou"] = bbox_iou(_bbox(row, "rgb"), _bbox(row, "thermal"))


def _add_temporal_features(rows: list[dict[str, Any]]) -> None:
    rows_by_track: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_track.setdefault((str(row["sequence"]), str(row["track_id"])), []).append(row)

    for track_rows in rows_by_track.values():
        track_rows.sort(key=lambda item: int(item["frame_id"]))
        for modality in MODALITIES:
            previous: dict[str, Any] | None = None
            for current in track_rows:
                if _center(current, modality) is None:
                    continue
                if previous is None:
                    previous = current
                    continue
                delta_frame = max(int(current["frame_id"]) - int(previous["frame_id"]), 1)
                current_center = _center(current, modality)
                previous_center = _center(previous, modality)
                if current_center is None or previous_center is None:
                    previous = current
                    continue
                dx = current_center[0] - previous_center[0]
                dy = current_center[1] - previous_center[1]
                vx = dx / float(delta_frame)
                vy = dy / float(delta_frame)
                speed = math.hypot(dx, dy) / float(delta_frame)
                current[f"{modality}_vx_px_per_frame"] = vx
                current[f"{modality}_vy_px_per_frame"] = vy
                current[f"{modality}_speed_px_per_frame"] = speed

                fps = _optional_float(current.get("fps")) or _optional_float(previous.get("fps"))
                if fps:
                    current[f"{modality}_vx_px_per_second"] = vx * fps
                    current[f"{modality}_vy_px_per_second"] = vy * fps
                    current[f"{modality}_speed_px_per_second"] = speed * fps
                previous = current


def _category_id(row: Sequence[str], profile: str) -> int | None:
    if profile in {"motchallenge", "m3ot"} and len(row) >= 8:
        return _optional_int(row[7])
    if profile in {"dancetrack", "sportsmot"}:
        return 1
    return _optional_int(row[7]) if len(row) >= 8 else None


def _visibility(row: Sequence[str], profile: str) -> float | None:
    if profile in {"motchallenge", "m3ot"} and len(row) >= 9:
        return _optional_float(row[8])
    if profile in {"dancetrack", "sportsmot"} and len(row) >= 9:
        return _optional_float(row[8])
    return None


def _category_name(profile: str, category_id: int | None) -> str | None:
    if category_id is None:
        return None
    return CATEGORY_NAMES_BY_PROFILE.get(profile, {}).get(category_id, str(category_id))


def _image_file(sequence: MotSequence, frame_id: int, frame_digits: int) -> str | None:
    if not sequence.image_dir:
        return None
    image_ext = sequence.image_ext or ".jpg"
    return f"{sequence.sequence}/{sequence.image_dir}/{frame_id:0{frame_digits}d}{image_ext}"


def _default_frame_digits(profile: str) -> int:
    return 8 if profile == "dancetrack" else 6


def _default_keep_category_ids(profile: str, enabled: bool) -> tuple[int, ...] | None:
    if enabled and profile in {"motchallenge", "m3ot", "dancetrack", "sportsmot"}:
        return (1,)
    return None


def _center(row: dict[str, Any], modality: str) -> tuple[float, float] | None:
    cx = row.get(f"{modality}_cx")
    cy = row.get(f"{modality}_cy")
    if cx is None or cy is None:
        return None
    return float(cx), float(cy)


def _bbox(row: dict[str, Any], modality: str) -> tuple[float, float, float, float] | None:
    values = [row.get(f"{modality}_{column}") for column in ("x", "y", "w", "h")]
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def bbox_iou(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> float | None:
    if left is None or right is None:
        return None
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    lx2, ly2 = lx + lw, ly + lh
    rx2, ry2 = rx + rw, ry + rh
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - inter
    return inter / union if union > 0.0 else None


def _track_sort_key(track_id: str) -> tuple[int, int | str]:
    return (0, int(track_id)) if str(track_id).isdigit() else (1, str(track_id))


def _required_int(value: str, field: str) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        raise ValueError(f"Missing integer value for {field}")
    return parsed


def _required_float(value: str, field: str) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        raise ValueError(f"Missing float value for {field}")
    return parsed


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(float(str(value)))


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def summary_asdict(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: [asdict(item) for item in value] if isinstance(value, list) else value
        for key, value in summary.items()
    }
