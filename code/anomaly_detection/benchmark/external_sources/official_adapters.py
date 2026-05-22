from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from baselines.individual_features import extract_center_sequence
from evaluation.io import write_jsonl


PIDPM_SOURCE = "official_pidpm:diffusion"


def write_pidpm_trajectory_csv(
    trajectories: Iterable[dict],
    output_csv: Path,
    sidecar_json: Path,
    max_points: int = 32,
) -> list[dict[str, Any]]:
    """Write trajectories as Pi-DPM flattened numeric CSV plus sample-id sidecar."""

    if int(max_points) < 1:
        raise ValueError("max_points must be at least 1")

    rows: list[list[float]] = []
    sidecar_rows: list[dict[str, Any]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        flattened = _flatten_centers(trajectory, max_points=int(max_points))
        rows.append(flattened)
        sidecar_rows.append(
            {
                "trajectory_id": trajectory_index,
                "sample_id": _sample_id(trajectory),
                "sequence": str(trajectory.get("sequence", "")),
                "track_id": str(trajectory.get("track_id", "")),
                "max_points": int(max_points),
            }
        )

    if not rows:
        raise ValueError("Cannot write Pi-DPM CSV with no trajectories")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_csv, np.asarray(rows, dtype=float), delimiter=",")
    sidecar_json.parent.mkdir(parents=True, exist_ok=True)
    sidecar_json.write_text(
        json.dumps(sidecar_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sidecar_rows


def convert_pidpm_scores_to_jsonl(
    pidpm_scores_csv: Path,
    sidecar_json: Path,
    output_jsonl: Path,
    source: str = PIDPM_SOURCE,
) -> list[dict[str, Any]]:
    sidecar_rows = json.loads(sidecar_json.read_text(encoding="utf-8"))
    if not isinstance(sidecar_rows, list):
        raise ValueError(f"{sidecar_json} must contain a JSON list")
    sidecar_by_id = {
        int(row["trajectory_id"]): row
        for row in sidecar_rows
        if isinstance(row, dict) and "trajectory_id" in row
    }

    rows: list[dict[str, Any]] = []
    with pidpm_scores_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, raw_row in enumerate(reader, start=2):
            if "trajectory_id" not in raw_row or "anomaly_score" not in raw_row:
                raise ValueError(
                    f"{pidpm_scores_csv}:{row_number} must contain trajectory_id and anomaly_score"
                )
            trajectory_id = int(raw_row["trajectory_id"])
            sidecar = sidecar_by_id.get(trajectory_id)
            if sidecar is None:
                raise ValueError(
                    f"{pidpm_scores_csv}:{row_number} has unknown trajectory_id {trajectory_id}"
                )
            score = float(raw_row["anomaly_score"])
            rows.append(
                {
                    "sample_id": str(sidecar["sample_id"]),
                    "sequence": str(sidecar.get("sequence", "")),
                    "track_id": str(sidecar.get("track_id", "")),
                    "source": source,
                    "score": score if np.isfinite(score) else 0.0,
                    "component_scores": {"diffusion_reconstruction_mse": score},
                    "metadata": {
                        "official_source": "Pi-DPM",
                        "trajectory_id": trajectory_id,
                        "max_points": int(sidecar.get("max_points", 0)),
                    },
                }
            )

    write_jsonl(output_jsonl, rows)
    return rows


def _flatten_centers(trajectory: dict, max_points: int) -> list[float]:
    sequence = extract_center_sequence(trajectory)
    centers = [(float(x), float(y)) for _, x, y in sequence[:max_points]]
    if not centers:
        centers = [(0.0, 0.0)]
    while len(centers) < max_points:
        centers.append(centers[-1])
    return [coordinate for center in centers for coordinate in center]


def _sample_id(trajectory: dict) -> str:
    sample_id = trajectory.get("sample_id")
    if sample_id not in (None, ""):
        return str(sample_id)
    sequence = str(trajectory.get("sequence", ""))
    track_id = str(trajectory.get("track_id", ""))
    return f"{sequence}:{track_id}" if sequence or track_id else ""
