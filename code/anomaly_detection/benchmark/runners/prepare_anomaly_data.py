from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.io import load_jsonl, write_jsonl
from protocol.schemas import build_sample_id
from protocol.inject_group import DEFAULT_GROUP_ANOMALIES, inject_group_anomalies
from protocol.inject_individual import (
    DEFAULT_INDIVIDUAL_ANOMALIES,
    inject_individual_anomalies,
)


LEVELS = ("individual", "group")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic injected anomaly benchmark data and labels."
    )
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--labels-jsonl", required=True, type=Path)
    parser.add_argument("--anomaly-fraction", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--anomaly-types",
        nargs="+",
        default=None,
        help="Optional anomaly type subset. Defaults depend on --level.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional JSON manifest with injection metadata.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_jsonl(args.input_jsonl)
    anomaly_types = _selected_types(args.level, args.anomaly_types)

    if args.level == "individual":
        injected, labels = inject_individual_anomalies(
            rows,
            anomaly_fraction=args.anomaly_fraction,
            seed=args.seed,
            anomaly_types=anomaly_types,
        )
    else:
        injected, labels = inject_group_anomalies(
            rows,
            anomaly_fraction=args.anomaly_fraction,
            seed=args.seed,
            anomaly_types=anomaly_types,
        )

    positive_label_rows = [label.to_dict() for label in labels]
    if args.level == "individual":
        label_rows = _complete_individual_labels(injected, positive_label_rows, args.seed)
    else:
        label_rows = _complete_group_labels(injected, positive_label_rows, args.seed)
    write_jsonl(args.output_jsonl, injected)
    write_jsonl(args.labels_jsonl, label_rows)

    manifest = {
        "level": args.level,
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "labels_jsonl": str(args.labels_jsonl),
        "anomaly_fraction": float(args.anomaly_fraction),
        "seed": int(args.seed),
        "anomaly_types": list(anomaly_types),
        "num_input_rows": len(rows),
        "num_output_rows": len(injected),
        "num_labels": len(label_rows),
    }
    if args.manifest_json is not None:
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_json.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _selected_types(level: str, values: list[str] | None) -> tuple[str, ...]:
    if values is not None:
        return tuple(values)
    if level == "individual":
        return DEFAULT_INDIVIDUAL_ANOMALIES
    return DEFAULT_GROUP_ANOMALIES


def _complete_individual_labels(
    trajectories: list[dict],
    positive_rows: list[dict],
    seed: int,
) -> list[dict]:
    rows_by_sample_id = {str(row["sample_id"]): dict(row) for row in positive_rows}
    for trajectory in trajectories:
        sequence = str(trajectory.get("sequence", ""))
        track_id = str(trajectory.get("track_id", ""))
        sample_id = str(trajectory.get("sample_id") or build_sample_id(sequence, track_id))
        rows_by_sample_id.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "sequence": sequence,
                "track_id": track_id,
                "frame_start": _point_frame_bounds(trajectory.get("points", []))[0],
                "frame_end": _point_frame_bounds(trajectory.get("points", []))[1],
                "label": 0,
                "anomaly_type": "normal",
                "injection_seed": int(seed),
                "metadata": {"source": "individual_injection"},
            },
        )
    return list(rows_by_sample_id.values())


def _complete_group_labels(
    windows: list[dict],
    positive_rows: list[dict],
    seed: int,
) -> list[dict]:
    rows_by_key: dict[tuple[str, str], dict] = {}
    for row in positive_rows:
        normalized = dict(row)
        window_id = _label_window_id(normalized)
        normalized["window_id"] = window_id
        normalized.setdefault("metadata", {})["window_id"] = window_id
        rows_by_key[(str(normalized["sample_id"]), window_id)] = normalized

    for window in windows:
        window_id = str(window.get("window_id", window.get("sample_id", "")))
        sequence = str(window.get("sequence", ""))
        frame_start, frame_end = _window_frame_bounds(window)
        for obj in window.get("objects", []):
            track_id = obj.get("track_id")
            if track_id in (None, ""):
                continue
            track_id = str(track_id)
            sample_id = str(obj.get("sample_id") or build_sample_id(sequence, track_id))
            rows_by_key.setdefault(
                (sample_id, window_id),
                {
                    "sample_id": sample_id,
                    "window_id": window_id,
                    "sequence": sequence,
                    "track_id": track_id,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "label": 0,
                    "anomaly_type": "normal",
                    "injection_seed": int(seed),
                    "metadata": {
                        "source": "group_injection",
                        "window_id": window_id,
                    },
                },
            )
    return list(rows_by_key.values())


def _label_window_id(row: dict) -> str:
    window_id = row.get("window_id")
    if window_id in (None, ""):
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            window_id = metadata.get("window_id")
    return str(window_id or "")


def _point_frame_bounds(points: list[dict]) -> tuple[int, int]:
    frame_ids = [int(point["frame_id"]) for point in points if "frame_id" in point]
    if not frame_ids:
        return 0, 0
    return min(frame_ids), max(frame_ids)


def _window_frame_bounds(window: dict) -> tuple[int, int]:
    frame_ids = [
        int(state["frame_id"])
        for obj in window.get("objects", [])
        for state in obj.get("states", [])
        if "frame_id" in state
    ]
    default_start = min(frame_ids) if frame_ids else 0
    default_end = max(frame_ids) if frame_ids else default_start
    return (
        int(window.get("frame_start", default_start)),
        int(window.get("frame_end", default_end)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
