from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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
MANIFEST_SCHEMA_VERSION = 2


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
    parser.add_argument(
        "--dataset-manifest-json",
        type=Path,
        default=None,
        help="Optional dataset manifest JSON to bind this synthetic protocol run to a dataset fingerprint.",
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

    manifest = _build_manifest(
        args=args,
        anomaly_types=anomaly_types,
        num_input_rows=len(rows),
        num_output_rows=len(injected),
        label_rows=label_rows,
    )
    if args.manifest_json is not None:
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_json.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _build_manifest(
    args: argparse.Namespace,
    anomaly_types: Sequence[str],
    num_input_rows: int,
    num_output_rows: int,
    label_rows: list[dict],
) -> dict:
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "synthetic_anomaly_protocol",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "level": args.level,
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "labels_jsonl": str(args.labels_jsonl),
        "anomaly_fraction": float(args.anomaly_fraction),
        "seed": int(args.seed),
        "anomaly_types": list(anomaly_types),
        "num_input_rows": int(num_input_rows),
        "num_output_rows": int(num_output_rows),
        "num_labels": len(label_rows),
        "protocol": {
            "level": args.level,
            "key_fields": _key_fields(args.level),
            "anomaly_fraction": float(args.anomaly_fraction),
            "seed": int(args.seed),
            "anomaly_types": list(anomaly_types),
            "allowed_anomaly_types": list(_selected_types(args.level, None)),
            "label_completion": "all_samples_with_normal_negatives",
        },
        "artifacts": {
            "input_jsonl": _artifact_summary(args.input_jsonl, num_input_rows),
            "output_jsonl": _artifact_summary(args.output_jsonl, num_output_rows),
            "labels_jsonl": _artifact_summary(args.labels_jsonl, len(label_rows)),
        },
        "label_distribution": _label_distribution(label_rows),
        "dataset_manifest": _dataset_manifest_summary(args.dataset_manifest_json),
        "replay": {
            "working_directory": str(BENCHMARK_ROOT),
            "argv": _replay_argv(args, anomaly_types),
        },
    }
    return manifest


def _selected_types(level: str, values: list[str] | None) -> tuple[str, ...]:
    if values is not None:
        return tuple(values)
    if level == "individual":
        return DEFAULT_INDIVIDUAL_ANOMALIES
    return DEFAULT_GROUP_ANOMALIES


def _key_fields(level: str) -> list[str]:
    return ["sample_id", "window_id"] if level == "group" else ["sample_id"]


def _artifact_summary(path: Path, num_rows: int) -> dict:
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "num_rows": int(num_rows),
    }


def _dataset_manifest_summary(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        payload = {}
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "schema_version": payload.get("schema_version"),
        "dataset_name": payload.get("dataset_name"),
        "status": payload.get("status"),
        "dataset_fingerprint": payload.get("dataset_fingerprint"),
    }


def _label_distribution(label_rows: list[dict]) -> dict:
    by_type = Counter(str(row.get("anomaly_type", "unknown")) for row in label_rows)
    num_positive = sum(1 for row in label_rows if int(row.get("label", 0) or 0) == 1)
    return {
        "num_labels": len(label_rows),
        "num_positive": num_positive,
        "num_negative": len(label_rows) - num_positive,
        "by_anomaly_type": dict(sorted(by_type.items())),
    }


def _replay_argv(args: argparse.Namespace, anomaly_types: Sequence[str]) -> list[str]:
    argv = [
        "python",
        "runners/prepare_anomaly_data.py",
        "--level",
        str(args.level),
        "--input-jsonl",
        str(args.input_jsonl),
        "--output-jsonl",
        str(args.output_jsonl),
        "--labels-jsonl",
        str(args.labels_jsonl),
        "--anomaly-fraction",
        str(float(args.anomaly_fraction)),
        "--seed",
        str(int(args.seed)),
    ]
    if anomaly_types:
        argv.extend(["--anomaly-types", *[str(item) for item in anomaly_types]])
    if args.manifest_json is not None:
        argv.extend(["--manifest-json", str(args.manifest_json)])
    if args.dataset_manifest_json is not None:
        argv.extend(["--dataset-manifest-json", str(args.dataset_manifest_json)])
    return argv


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
