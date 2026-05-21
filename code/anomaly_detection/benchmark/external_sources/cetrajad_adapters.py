from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from baselines.individual_features import extract_center_sequence
from evaluation.io import load_jsonl, write_jsonl


CETRAJAD_SOURCE = "official_cetrajad:comp_ensemble"
CETRAJAD_REPOSITORY = "https://github.com/ShuruiCao/comp-ensemble-ad"
ID_COLUMNS = ("sample_id", "trajectory_id", "track_id", "id")
SCORE_COLUMNS = ("score", "anomaly_score", "final_score", "cetrajad_score")


def write_cetrajad_official_input_bundle(
    trajectories: Iterable[dict[str, Any]],
    output_dir: Path,
    dict_name: str = "fusiontrack",
    sidecar_name: str = "cetrajad_sidecar.json",
    manifest_name: str = "cetrajad_manifest.json",
) -> dict[str, Any]:
    """Prepare a conservative input bundle for an external CETrajAD checkout.

    The official CETrajAD repository exposes scripts that load pickle files such
    as ``detour_evaluation_gps.pkl`` and pass them to ``TripProcessor.process``.
    This adapter therefore writes ``<dict_name>_evaluation_gps.pkl`` as a dict
    of pandas DataFrames with ``latitude`` and ``longitude`` columns, plus a
    sidecar and manifest. Use the manifest paths as the data path arguments when
    running the official scripts in a separate CETrajAD checkout.
    """

    if not dict_name:
        raise ValueError("dict_name must be non-empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = output_dir / f"{dict_name}_evaluation_gps.pkl"
    sidecar_path = output_dir / sidecar_name
    manifest_path = output_dir / manifest_name
    point_csv_path = output_dir / f"{dict_name}_points.csv"
    records_jsonl_path = output_dir / f"{dict_name}_records.jsonl"

    trip_dict: dict[int, pd.DataFrame] = {}
    sidecar_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []

    for trajectory_id, trajectory in enumerate(trajectories):
        frame_rows = _trajectory_to_frame_rows(trajectory)
        if not frame_rows:
            raise ValueError(f"trajectory {trajectory_id} has no usable center points")
        if len(frame_rows) == 1:
            frame_rows.append(dict(frame_rows[0]))

        trip_dict[trajectory_id] = pd.DataFrame(
            frame_rows,
            columns=["timestamp", "frame_id", "latitude", "longitude"],
        )
        sample_id = _sample_id(trajectory)
        sidecar = {
            "trajectory_id": trajectory_id,
            "sample_id": sample_id,
            "sequence": str(trajectory.get("sequence", "")),
            "track_id": str(trajectory.get("track_id", "")),
            "num_points": len(frame_rows),
        }
        sidecar_rows.append(sidecar)
        record_rows.append(
            {
                "trajectory_id": trajectory_id,
                "sample_id": sample_id,
                "sequence": sidecar["sequence"],
                "track_id": sidecar["track_id"],
                "points": frame_rows,
            }
        )
        for frame_row in frame_rows:
            point_rows.append({**sidecar, **frame_row})

    if not trip_dict:
        raise ValueError("Cannot write CETrajAD inputs with no trajectories")

    pd.to_pickle(trip_dict, pickle_path)
    sidecar_path.write_text(
        json.dumps(sidecar_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_jsonl(records_jsonl_path, record_rows)
    _write_point_csv(point_csv_path, point_rows)

    manifest = {
        "official_source": "CETrajAD",
        "official_repository": CETRAJAD_REPOSITORY,
        "source": CETRAJAD_SOURCE,
        "dict_name": dict_name,
        "num_trajectories": len(sidecar_rows),
        "input_pickle": str(pickle_path),
        "sidecar_json": str(sidecar_path),
        "point_csv": str(point_csv_path),
        "records_jsonl": str(records_jsonl_path),
        "usage_note": (
            "Prepared for an external CETrajAD checkout; pass this directory or "
            "the input pickle as the official script data path according to the "
            "checked-out CETrajAD script parameters."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def convert_cetrajad_scores_to_jsonl(
    cetrajad_scores: Path,
    sidecar_json: Path | None,
    output_jsonl: Path,
    source: str = CETRAJAD_SOURCE,
    score_column: str | None = None,
    id_column: str | None = None,
) -> list[dict[str, Any]]:
    """Convert external CETrajAD score CSV/JSON/JSONL to benchmark score JSONL."""

    sidecar_rows = _load_sidecar(sidecar_json)
    sidecar_maps = _sidecar_maps(sidecar_rows)
    rows: list[dict[str, Any]] = []

    for row_number, raw_row in enumerate(_load_score_rows(cetrajad_scores), start=1):
        current_id_column = _resolve_column(
            raw_row,
            ID_COLUMNS,
            id_column,
            cetrajad_scores,
            row_number,
            "ID",
        )
        current_score_column = _resolve_column(
            raw_row,
            SCORE_COLUMNS,
            score_column,
            cetrajad_scores,
            row_number,
            "score",
        )
        if current_id_column is None or current_score_column is None:
            raise ValueError(
                f"{cetrajad_scores}:{row_number} must contain one of {ID_COLUMNS} "
                f"and one of {SCORE_COLUMNS}"
            )

        identifier = raw_row[current_id_column]
        sidecar = _match_sidecar(
            sidecar_maps,
            current_id_column,
            identifier,
            sidecar_required=sidecar_json is not None and current_id_column != "sample_id",
            scores_path=cetrajad_scores,
            row_number=row_number,
        )
        sample_id = str(sidecar.get("sample_id", raw_row.get("sample_id", "")))
        if sample_id in ("", "None"):
            raise ValueError(f"{cetrajad_scores}:{row_number} resolved empty sample_id")
        sequence = str(sidecar.get("sequence", raw_row.get("sequence", "")))
        track_id = str(sidecar.get("track_id", raw_row.get("track_id", "")))
        score = _finite_score(raw_row[current_score_column])
        trajectory_id = sidecar.get("trajectory_id", raw_row.get("trajectory_id"))

        rows.append(
            {
                "sample_id": sample_id,
                "sequence": sequence,
                "track_id": track_id,
                "source": source,
                "score": score,
                "component_scores": {"cetrajad_official_score": score},
                "metadata": {
                    "official_source": "CETrajAD",
                    "score_file": str(cetrajad_scores),
                    "id_column": current_id_column,
                    "score_column": current_score_column,
                    "trajectory_id": trajectory_id,
                },
            }
        )

    if not rows:
        raise ValueError(f"{cetrajad_scores} contains no score rows")

    write_jsonl(output_jsonl, rows)
    return rows


def _trajectory_to_frame_rows(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id, x_coord, y_coord in extract_center_sequence(trajectory):
        rows.append(
            {
                "timestamp": int(frame_id),
                "frame_id": int(frame_id),
                "latitude": float(y_coord),
                "longitude": float(x_coord),
            }
        )
    return rows


def _write_point_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "trajectory_id",
        "sample_id",
        "sequence",
        "track_id",
        "num_points",
        "timestamp",
        "frame_id",
        "latitude",
        "longitude",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_score_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return _ensure_dict_rows(path, data)
        if isinstance(data, dict):
            for key in ("scores", "rows", "results"):
                value = data.get(key)
                if isinstance(value, list):
                    return _ensure_dict_rows(path, value)
            if any(column in data for column in (*ID_COLUMNS, *SCORE_COLUMNS)):
                return [data]
            return [
                {"sample_id": str(identifier), "score": score}
                for identifier, score in data.items()
                if not isinstance(score, (dict, list))
            ]
    raise ValueError(f"Unsupported score file extension for {path}; expected .csv, .json, or .jsonl")


def _ensure_dict_rows(path: Path, rows: list[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index} is not a JSON object")
        converted.append(row)
    return converted


def _load_sidecar(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [row for row in rows if isinstance(row, dict)]


def _sidecar_maps(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    maps: dict[str, dict[str, dict[str, Any]]] = {column: {} for column in ID_COLUMNS}
    duplicate_values: dict[str, set[str]] = {column: set() for column in ID_COLUMNS}
    for row in rows:
        for column in ID_COLUMNS:
            value = row.get(column)
            if value not in (None, ""):
                key = str(value)
                if key in maps[column]:
                    duplicate_values[column].add(key)
                else:
                    maps[column][key] = row
    for column, values in duplicate_values.items():
        for value in values:
            maps[column].pop(value, None)
    return maps


def _match_sidecar(
    sidecar_maps: dict[str, dict[str, dict[str, Any]]],
    id_column: str,
    identifier: Any,
    sidecar_required: bool,
    scores_path: Path,
    row_number: int,
) -> dict[str, Any]:
    if identifier in (None, ""):
        if sidecar_required:
            raise ValueError(f"{scores_path}:{row_number} has empty {id_column}")
        return {}
    matched = sidecar_maps.get(id_column, {}).get(str(identifier))
    if matched is not None:
        return matched
    if id_column == "id":
        for column in ("sample_id", "trajectory_id"):
            matched = sidecar_maps.get(column, {}).get(str(identifier))
            if matched is not None:
                return matched
    if sidecar_required:
        raise ValueError(f"{scores_path}:{row_number} has unknown {id_column} {identifier}")
    return {}


def _first_present(row: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in row and row[candidate] not in (None, ""):
            return candidate
    return None


def _resolve_column(
    row: dict[str, Any],
    candidates: tuple[str, ...],
    explicit_column: str | None,
    scores_path: Path,
    row_number: int,
    column_kind: str,
) -> str | None:
    if explicit_column is not None:
        if explicit_column not in row or row[explicit_column] in (None, ""):
            raise ValueError(
                f"{scores_path}:{row_number} missing explicit {column_kind} column "
                f"{explicit_column}"
            )
        return explicit_column
    present = [candidate for candidate in candidates if candidate in row and row[candidate] not in (None, "")]
    if len(present) > 1 and column_kind != "ID":
        raise ValueError(
            f"{scores_path}:{row_number} has multiple candidate {column_kind} columns "
            f"{present}; specify one explicitly"
        )
    return present[0] if present else None


def _finite_score(value: Any) -> float:
    score = float(value)
    return score if math.isfinite(score) else 0.0


def _sample_id(trajectory: dict[str, Any]) -> str:
    sample_id = trajectory.get("sample_id")
    if sample_id not in (None, ""):
        return str(sample_id)
    sequence = str(trajectory.get("sequence", ""))
    track_id = str(trajectory.get("track_id", ""))
    return f"{sequence}:{track_id}" if sequence or track_id else ""
