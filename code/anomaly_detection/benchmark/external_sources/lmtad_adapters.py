from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from baselines.individual_features import extract_center_sequence
from evaluation.io import load_jsonl, write_jsonl


LMTAD_SOURCE = "official_lmtad:lm"
LMTAD_OFFICIAL_REPOSITORY = "https://github.com/jonathankabala/LMTAD"
LMTAD_OFFICIAL_COMMIT_INSPECTED = "80bb89a8ea108db8f13cb9959826424e9c45f41c"
SPECIAL_TOKENS = ("PAD", "EOT", "SOT")
SCORE_COLUMNS = ("score", "anomaly_score", "nll", "log_perplexity")
ID_COLUMNS = ("sample_id", "trajectory_id", "track_id")


def write_lmtad_official_inputs(
    trajectories: Iterable[dict[str, Any]],
    output_dir: Path,
    grid_size: float = 25.0,
    sequence_filename: str = "lmtad_sequences.jsonl",
    manifest_filename: str = "manifest.json",
    vocab_filename: str = "vocab.json",
) -> dict[str, Any]:
    """Write intermediate LM-TAD inputs for use from an external LMTAD checkout.

    The official LMTAD code inspected here is bound to its Porto and Pattern-of-Life
    dataset loaders. This adapter writes a neutral sequence JSONL, vocab, and
    manifest. To run official LM-TAD on FusionTrack trajectories, add a custom
    Dataset in the external LMTAD checkout that reads these files and then run
    the official train/eval scripts against that loader.
    """

    grid_size = float(grid_size)
    if not math.isfinite(grid_size) or grid_size <= 0.0:
        raise ValueError("grid_size must be a positive finite number")

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = output_dir / sequence_filename
    vocab_path = output_dir / vocab_filename
    manifest_path = output_dir / manifest_filename

    vocab: dict[str, int] = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    sequence_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for trajectory_id, trajectory in enumerate(trajectories):
        centers_with_frames = extract_center_sequence(trajectory)
        frames = [int(frame_id) for frame_id, _, _ in centers_with_frames]
        centers = [[float(x), float(y)] for _, x, y in centers_with_frames]
        tokens = [_token_for_center(x, y, grid_size) for _, x, y in centers_with_frames]
        if not tokens:
            tokens = ["cell_empty"]

        token_ids = [_token_id(token, vocab) for token in tokens]
        sample_id = _sample_id(trajectory)
        sequence = str(trajectory.get("sequence", ""))
        track_id = str(trajectory.get("track_id", ""))

        sequence_rows.append(
            {
                "trajectory_id": trajectory_id,
                "sample_id": sample_id,
                "sequence": sequence,
                "track_id": track_id,
                "frames": frames,
                "centers": centers,
                "tokens": tokens,
                "token_ids": token_ids,
                "num_points": len(centers_with_frames),
            }
        )
        manifest_rows.append(
            {
                "trajectory_id": trajectory_id,
                "sample_id": sample_id,
                "sequence": sequence,
                "track_id": track_id,
                "num_tokens": len(tokens),
            }
        )

    if not sequence_rows:
        raise ValueError("Cannot write LM-TAD inputs with no trajectories")

    write_jsonl(sequence_path, sequence_rows)
    vocab_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema": "fusiontrack.lmtad_official_inputs.v1",
        "official_repository": LMTAD_OFFICIAL_REPOSITORY,
        "official_commit_inspected": LMTAD_OFFICIAL_COMMIT_INSPECTED,
        "official_entrypoints": [
            "code/train_LMTAD.py",
            "code/eval_lm.py",
            "code/eval_porto.py",
        ],
        "official_loader_constraint": (
            "The inspected official code uses PortoDataset and POLDataset loaders "
            "with Porto/POL-specific preprocessing outputs."
        ),
        "external_checkout_integration": (
            "Use these files from an external LMTAD checkout by adding a custom "
            "dataset loader that reads lmtad_sequences.jsonl and vocab.json, "
            "then run the official LM-TAD train/eval entrypoints. Main-table "
            "paper results must come from that official checkout run."
        ),
        "files": {
            "sequence_jsonl": sequence_filename,
            "vocab_json": vocab_filename,
            "manifest_json": manifest_filename,
        },
        "tokenization": {
            "grid_size": grid_size,
            "token_format": "cell_{floor(x/grid_size)}_{floor(y/grid_size)}",
            "special_tokens": {token: vocab[token] for token in SPECIAL_TOKENS},
        },
        "trajectories": manifest_rows,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def convert_lmtad_scores_to_jsonl(
    scores_path: Path,
    output_jsonl: Path,
    manifest_json: Path | None = None,
    source: str = LMTAD_SOURCE,
    score_column: str | None = None,
    id_column: str | None = None,
) -> list[dict[str, Any]]:
    rows = _load_score_rows(scores_path)
    if not rows:
        raise ValueError(f"{scores_path} contains no score rows")

    manifest_by_trajectory = _load_manifest_by_trajectory(manifest_json)
    converted: list[dict[str, Any]] = []
    for row_number, raw_row in enumerate(rows, start=1):
        current_id_column = _resolve_column(
            raw_row,
            ID_COLUMNS,
            id_column,
            scores_path,
            row_number,
            "ID",
        )
        if current_id_column is None:
            raise ValueError(
                f"{scores_path}:{row_number} must contain one of {', '.join(ID_COLUMNS)}"
            )
        current_score_column = _resolve_column(
            raw_row,
            SCORE_COLUMNS,
            score_column,
            scores_path,
            row_number,
            "score",
        )
        if current_score_column is None:
            raise ValueError(
                f"{scores_path}:{row_number} must contain one of {', '.join(SCORE_COLUMNS)}"
            )

        score = _finite_score(raw_row[current_score_column])
        identifier = raw_row.get(current_id_column)
        sidecar = _sidecar_for_identifier(
            current_id_column,
            identifier,
            manifest_by_trajectory,
            scores_path,
            row_number,
        )

        sample_id = _score_sample_id(raw_row, current_id_column, identifier, sidecar)
        if sample_id in ("", "None"):
            raise ValueError(f"{scores_path}:{row_number} resolved empty sample_id")
        sequence = str(raw_row.get("sequence", sidecar.get("sequence", "")))
        track_id = str(raw_row.get("track_id", sidecar.get("track_id", "")))
        metadata = {
            "official_source": "LM-TAD",
            "score_column": current_score_column,
            "id_column": current_id_column,
        }
        if "trajectory_id" in raw_row or "trajectory_id" in sidecar:
            metadata["trajectory_id"] = _coerce_int(
                raw_row.get("trajectory_id", sidecar.get("trajectory_id"))
            )

        converted.append(
            {
                "sample_id": sample_id,
                "sequence": sequence,
                "track_id": track_id,
                "source": source,
                "score": score,
                "component_scores": {f"official_lmtad_{current_score_column}": score},
                "metadata": metadata,
            }
        )

    write_jsonl(output_jsonl, converted)
    return converted


def _load_score_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            for key in ("scores", "results", "rows", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    payload = value
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"{path} must contain a JSON object or list of objects")
        return list(payload)
    raise ValueError(f"Unsupported score file extension for {path}; expected .csv, .tsv, .json, or .jsonl")


def _load_manifest_by_trajectory(manifest_json: Path | None) -> dict[str, dict[str, Any]]:
    if manifest_json is None:
        return {}
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    trajectories = manifest.get("trajectories", [])
    if not isinstance(trajectories, list):
        raise ValueError(f"{manifest_json} must contain a trajectories list")
    mapping: dict[str, dict[str, Any]] = {}
    for row in trajectories:
        if isinstance(row, dict) and "trajectory_id" in row:
            mapping[str(row["trajectory_id"])] = row
    return mapping


def _sidecar_for_identifier(
    id_column: str,
    identifier: Any,
    manifest_by_trajectory: dict[str, dict[str, Any]],
    scores_path: Path,
    row_number: int,
) -> dict[str, Any]:
    if id_column != "trajectory_id":
        return {}
    sidecar = manifest_by_trajectory.get(str(identifier))
    if manifest_by_trajectory and sidecar is None:
        raise ValueError(
            f"{scores_path}:{row_number} has unknown trajectory_id {identifier}"
        )
    return sidecar or {}


def _sample_id(trajectory: dict[str, Any]) -> str:
    sample_id = trajectory.get("sample_id")
    if sample_id not in (None, ""):
        return str(sample_id)
    sequence = str(trajectory.get("sequence", ""))
    track_id = str(trajectory.get("track_id", ""))
    return f"{sequence}:{track_id}" if sequence or track_id else ""


def _score_sample_id(
    row: dict[str, Any],
    id_column: str,
    identifier: Any,
    sidecar: dict[str, Any],
) -> str:
    if row.get("sample_id") not in (None, ""):
        return str(row["sample_id"])
    if sidecar.get("sample_id") not in (None, ""):
        return str(sidecar["sample_id"])
    if id_column == "track_id" and identifier not in (None, ""):
        return str(identifier)
    if row.get("sequence") not in (None, "") or row.get("track_id") not in (None, ""):
        return f"{row.get('sequence', '')}:{row.get('track_id', '')}"
    return str(identifier)


def _token_for_center(x: float, y: float, grid_size: float) -> str:
    grid_x = math.floor(float(x) / grid_size)
    grid_y = math.floor(float(y) / grid_size)
    return f"cell_{grid_x}_{grid_y}"


def _token_id(token: str, vocab: dict[str, int]) -> int:
    if token not in vocab:
        vocab[token] = len(vocab)
    return vocab[token]


def _first_present(row: dict[str, Any], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in row and row[candidate] not in (None, ""):
            return candidate
    return None


def _resolve_column(
    row: dict[str, Any],
    candidates: Iterable[str],
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


def _coerce_int(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)
