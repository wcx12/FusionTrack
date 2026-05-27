#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SYSTEM_ROOT = REPO_ROOT / "code" / "system"
MTF_BA_ROOT = REPO_ROOT / "code" / "anomaly_detection" / "individual"
for path in (SYSTEM_ROOT, MTF_BA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fusiontrack.final_dashboard import build_final_dashboard
from fusiontrack.final_results import load_final_results_dashboard


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_fixture(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    final_root = root / "final"
    score_root = root / "scores"
    labels_root = root / "labels"
    individual_labels = labels_root / "individual_labels_val.jsonl"
    group_labels = labels_root / "group_labels_val.jsonl"
    fused_jsonl = root / "fused.jsonl"

    _write_jsonl(
        individual_labels,
        [
            {
                "sample_id": "S1:1",
                "sequence": "S1",
                "track_id": "1",
                "frame_start": 10,
                "frame_end": 20,
                "label": 1,
                "anomaly_type": "speed_spike",
                "injection_seed": 42,
            },
            {
                "sample_id": "S1:2",
                "sequence": "S1",
                "track_id": "2",
                "frame_start": 12,
                "frame_end": 18,
                "label": 1,
                "anomaly_type": "route_shift",
                "injection_seed": 42,
            },
        ],
    )
    _write_jsonl(
        group_labels,
        [
            {
                "sample_id": "G1:1",
                "sequence": "G1",
                "track_id": "1",
                "frame_start": 1,
                "frame_end": 8,
                "label": 1,
                "anomaly_type": "population_change",
                "injection_seed": 42,
            },
            {
                "sample_id": "G1:2",
                "sequence": "G1",
                "track_id": "2",
                "frame_start": 1,
                "frame_end": 8,
                "label": 0,
                "anomaly_type": "normal",
                "injection_seed": 42,
            },
        ],
    )
    _write_jsonl(
        score_root / "individual" / "scores" / "fusiontrack_individual_nn.jsonl",
        [
            {"sample_id": "S1:1", "sequence": "S1", "track_id": "1", "score": 0.95},
            {"sample_id": "S1:2", "sequence": "S1", "track_id": "2", "score": 0.12},
        ],
    )
    _write_jsonl(
        score_root / "group" / "scores" / "group_prediction_linear.jsonl",
        [
            {
                "sample_id": "G1:1",
                "sequence": "G1",
                "track_id": "1",
                "score": 0.82,
                "event_score": 0.82,
                "frame_event_scores": [
                    {"frame": 2, "score": 0.62, "dominant_reason": "dispersion"},
                    {"frame": 6, "score": 0.82, "dominant_reason": "population_change"},
                ],
            },
            {"sample_id": "G1:2", "sequence": "G1", "track_id": "2", "score": 0.15},
        ],
    )
    _write_csv(
        final_root / "final_individual_all_methods_summary.csv",
        [
            {
                "method": "fusiontrack_individual_nn",
                "source": "scores/individual/scores/fusiontrack_individual_nn.jsonl",
                "split": "val",
                "seed": 42,
                "auroc": 0.70,
                "auprc": 0.40,
                "f1": 0.50,
                "precision_at_k": 0.50,
                "recall_at_k": 0.50,
                "num_score_rows": 2,
                "num_missing_score_keys": 0,
            }
        ],
    )
    _write_csv(
        final_root / "final_group_all_methods_summary.csv",
        [
            {
                "method": "group_prediction_linear",
                "source": "scores/group/scores/group_prediction_linear.jsonl",
                "split": "val",
                "seed": 42,
                "auroc": 0.62,
                "auprc": 0.20,
                "f1": 0.30,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "num_score_rows": 2,
                "num_missing_score_keys": 0,
            }
        ],
    )
    _write_jsonl(
        fused_jsonl,
        [
            {
                "sample_id": "S1:1",
                "sequence": "S1",
                "track_id": "1",
                "category_name": "plane",
                "points": [
                    {"frame_id": 10, "fused": {"center_xy": [10, 20], "confidence": 0.9}},
                    {"frame_id": 20, "fused": {"center_xy": [24, 38], "confidence": 0.8}},
                ],
            },
            {
                "sample_id": "S1:2",
                "sequence": "S1",
                "track_id": "2",
                "category_name": "plane",
                "points": [
                    {"frame_id": 12, "fused": {"center_xy": [42, 58], "confidence": 0.9}},
                    {"frame_id": 18, "fused": {"center_xy": [50, 63], "confidence": 0.8}},
                ],
            },
            {
                "sample_id": "G1:1",
                "sequence": "G1",
                "track_id": "1",
                "category_name": "plane",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [15, 25], "confidence": 0.9}},
                    {"frame_id": 8, "fused": {"center_xy": [30, 42], "confidence": 0.8}},
                ],
            },
            {
                "sample_id": "G1:2",
                "sequence": "G1",
                "track_id": "2",
                "category_name": "plane",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [35, 45], "confidence": 0.9}},
                    {"frame_id": 8, "fused": {"center_xy": [45, 55], "confidence": 0.8}},
                ],
            },
        ],
    )
    return final_root, score_root, individual_labels, group_labels, fused_jsonl


def build_sample_dashboard(output_dir: Path) -> dict[str, object]:
    fixture_root = output_dir / "_fixture"
    final_root, score_root, individual_labels, group_labels, fused_jsonl = _build_fixture(fixture_root)
    dashboard = load_final_results_dashboard(
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
        top_k=2,
        case_limit=3,
    )
    dashboard_dir = output_dir / "sample_dashboard"
    summary = build_final_dashboard(
        dashboard=dashboard,
        output_dir=dashboard_dir,
        fused_jsonl=fused_jsonl,
        data_root=output_dir / "missing_data_root",
        top_sequences=2,
    )
    summary_path = output_dir / "sample_dashboard_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dashboard_dir": str(dashboard_dir), "summary_path": str(summary_path), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny FusionTrack dashboard artifact for CI.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "sample_dashboard_ci")
    args = parser.parse_args()
    result = build_sample_dashboard(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
