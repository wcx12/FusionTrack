from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from baselines.individual_features import build_handcrafted_feature_table
from evaluation.io import load_jsonl, write_jsonl
from evaluation.reporting import evaluate_score_file, summarize_metric_files
from fusiontrack.group_temporal_profile import (
    _residual_gated_rank_fusion,
    _residual_side_gates,
)
from fusiontrack.individual_scoring import _feature_stratified_rank01


INDIVIDUAL_WEIGHT_VARIANTS = (
    ("base_weights", (0.40, 0.35, 0.25)),
    ("lof_heavy", (0.25, 0.55, 0.20)),
    ("nearest_lof", (0.45, 0.45, 0.10)),
    ("nearest_heavy", (0.60, 0.30, 0.10)),
    ("balanced", (0.33, 0.34, 0.33)),
)
INDIVIDUAL_CALIBRATION_VARIANTS = (
    ("raw", (), 4, 1.0),
    ("speed_gw05", ("mean_speed",), 4, 0.5),
    ("motion_gw03", ("mean_speed", "duration_frames", "num_points"), 4, 0.3),
    ("motion_gw05", ("mean_speed", "duration_frames", "num_points"), 4, 0.5),
    ("motion_gw07", ("mean_speed", "duration_frames", "num_points"), 4, 0.7),
    ("speed_duration_gw05", ("mean_speed", "duration_frames"), 4, 0.5),
)

GROUP_WEIGHT_VARIANTS = (
    ("base_weights", (0.60, 0.20, 0.20)),
    ("pred70_side15", (0.70, 0.15, 0.15)),
    ("pred80_side10", (0.80, 0.10, 0.10)),
    ("pred50_side25", (0.50, 0.25, 0.25)),
    ("graph_heavy", (0.60, 0.30, 0.10)),
    ("temporal_heavy", (0.60, 0.10, 0.30)),
)
GROUP_GATE_VARIANTS = (
    ("ungated", False, 1.0, 0.0),
    ("gate_p1_f00", True, 1.0, 0.0),
    ("gate_p1_f05", True, 1.0, 0.05),
    ("gate_p1_f10", True, 1.0, 0.10),
    ("gate_p2_f00", True, 2.0, 0.0),
    ("gate_p2_f05", True, 2.0, 0.05),
    ("gate_p2_f10", True, 2.0, 0.10),
    ("gate_p3_f00", True, 3.0, 0.0),
    ("gate_p3_f05", True, 3.0, 0.05),
    ("gate_p3_f10", True, 3.0, 0.10),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep FusionTrack score-combination variants from cached component scores."
    )
    parser.add_argument("--protocol-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    individual_metrics = _run_individual_grid(
        protocol_root=args.protocol_root,
        result_root=args.result_root,
        output_dir=output_dir / "individual",
    )
    group_metrics = _run_group_grid(
        protocol_root=args.protocol_root,
        result_root=args.result_root,
        output_dir=output_dir / "group",
    )
    best = {
        "individual": _best_by_metric(individual_metrics),
        "group": _best_by_metric(group_metrics),
    }
    best_path = output_dir / "best.json"
    best_path.write_text(
        json.dumps(best, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "best": best}, ensure_ascii=False))
    return 0


def _run_individual_grid(
    protocol_root: Path,
    result_root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    score_dir = output_dir / "scores"
    metric_dir = output_dir / "metrics"
    score_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_jsonl(
        result_root / "individual" / "scores" / "fusiontrack_individual_ensemble.jsonl"
    )
    rows_by_sample = {str(row["sample_id"]): row for row in base_rows}
    feature_df = build_handcrafted_feature_table(
        load_jsonl(protocol_root / "fused_trajectories_val.jsonl")
    )
    sample_ids = feature_df["sample_id"].astype(str).tolist()
    aligned_rows = [rows_by_sample[sample_id] for sample_id in sample_ids]
    nearest = _component_array(aligned_rows, "nearest_feature_rank")
    lof = _component_array(aligned_rows, "lof_novelty_rank")
    iforest = _component_array(aligned_rows, "isolation_forest_rank")

    metrics: list[dict[str, Any]] = []
    metric_files: list[Path] = []
    for weight_name, weights in INDIVIDUAL_WEIGHT_VARIANTS:
        clean_weights = _normalize_weights(weights, fallback=(0.4, 0.35, 0.25))
        base_scores = (
            clean_weights[0] * nearest
            + clean_weights[1] * lof
            + clean_weights[2] * iforest
        )
        for calibration_name, columns, bins, global_weight in INDIVIDUAL_CALIBRATION_VARIANTS:
            method = f"fusiontrack_individual_grid_{weight_name}_{calibration_name}"
            if columns:
                scores = _feature_stratified_rank01(
                    base_scores,
                    feature_df,
                    columns=columns,
                    bins=bins,
                    global_weight=global_weight,
                )
            else:
                scores = [float(score) if np.isfinite(score) else 0.0 for score in base_scores]

            rows = []
            for row, score in zip(aligned_rows, scores):
                component_scores = dict(row.get("component_scores", {}))
                component_scores["uncalibrated_ensemble_rank"] = float(
                    base_scores[len(rows)]
                )
                rows.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "sequence": str(row.get("sequence", "")),
                        "track_id": str(row.get("track_id", "")),
                        "source": "fusiontrack_individual:grid",
                        "score": float(score) if np.isfinite(score) else 0.0,
                        "component_scores": component_scores,
                        "metadata": {
                            "method": method,
                            "weights": _individual_weight_metadata(clean_weights),
                            "calibration": {
                                "enabled": bool(columns),
                                "columns": list(columns),
                                "bins": int(bins),
                                "global_weight": float(global_weight),
                            },
                        },
                    }
                )
            score_path = score_dir / f"{method}.jsonl"
            metric_path = metric_dir / f"{method}.json"
            write_jsonl(score_path, rows)
            metric = evaluate_score_file(
                score_path=score_path,
                label_path=protocol_root / "individual_labels_val.jsonl",
                key_fields=("sample_id",),
                k=100,
                require_unique_keys=True,
                require_score_key_match=True,
            )
            metric.update({"method": method, "task": "fusiontrack_individual_grid"})
            metric_path.write_text(
                json.dumps(metric, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            metric_files.append(metric_path)
            metrics.append(metric)

    summarize_metric_files(metric_files, output_csv=output_dir / "summary.csv")
    return metrics


def _run_group_grid(
    protocol_root: Path,
    result_root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    score_dir = output_dir / "scores"
    metric_dir = output_dir / "metrics"
    score_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_jsonl(
        result_root / "group" / "scores" / "fusiontrack_group_hybrid.jsonl"
    )
    prediction = _component_array(base_rows, "prediction_residual_rank")
    graph = _component_array(base_rows, "graph_rank")
    temporal = _component_array(base_rows, "temporal_profile_rank")

    metrics: list[dict[str, Any]] = []
    metric_files: list[Path] = []
    for weight_name, weights in GROUP_WEIGHT_VARIANTS:
        clean_weights = _normalize_weights(weights, fallback=(0.6, 0.2, 0.2))
        for gate_name, enabled, power, floor in GROUP_GATE_VARIANTS:
            method = f"fusiontrack_group_grid_{weight_name}_{gate_name}"
            scores = _residual_gated_rank_fusion(
                prediction_rank=prediction,
                graph_rank=graph,
                temporal_rank=temporal,
                weights=clean_weights,
                enabled=enabled,
                gate_power=power,
                gate_floor=floor,
            )
            gates = _residual_side_gates(
                prediction,
                enabled=enabled,
                gate_power=power,
                gate_floor=floor,
            )
            rows = []
            for row, score, gate in zip(base_rows, scores, gates):
                component_scores = dict(row.get("component_scores", {}))
                component_scores["residual_side_gate"] = float(gate)
                rows.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "window_id": str(row["window_id"]),
                        "sequence": str(row.get("sequence", "")),
                        "track_id": str(row.get("track_id", "")),
                        "frame_start": row.get("frame_start"),
                        "frame_end": row.get("frame_end"),
                        "source": "fusiontrack_group:grid",
                        "score": float(score) if np.isfinite(score) else 0.0,
                        "component_scores": component_scores,
                        "metadata": {
                            "method": method,
                            "weights": _group_weight_metadata(clean_weights),
                            "residual_gate": {
                                "enabled": bool(enabled),
                                "power": float(power),
                                "floor": float(floor),
                            },
                        },
                    }
                )
            score_path = score_dir / f"{method}.jsonl"
            metric_path = metric_dir / f"{method}.json"
            write_jsonl(score_path, rows)
            metric = evaluate_score_file(
                score_path=score_path,
                label_path=protocol_root / "group_labels_val.jsonl",
                key_fields=("sample_id", "window_id"),
                k=100,
                require_unique_keys=True,
                require_score_key_match=True,
            )
            metric.update({"method": method, "task": "fusiontrack_group_grid"})
            metric_path.write_text(
                json.dumps(metric, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            metric_files.append(metric_path)
            metrics.append(metric)

    summarize_metric_files(metric_files, output_csv=output_dir / "summary.csv")
    return metrics


def _component_array(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    values = [
        float(row.get("component_scores", {}).get(key, 0.0) or 0.0)
        for row in rows
    ]
    return np.asarray(values, dtype=float)


def _normalize_weights(
    values: Sequence[float],
    fallback: Sequence[float],
) -> tuple[float, float, float]:
    weights = np.asarray(list(values), dtype=float)
    if len(weights) != 3 or not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.asarray(list(fallback), dtype=float)
    weights = weights / float(weights.sum())
    return float(weights[0]), float(weights[1]), float(weights[2])


def _individual_weight_metadata(weights: Sequence[float]) -> dict[str, float]:
    return {
        "nearest_feature_rank": float(weights[0]),
        "lof_novelty_rank": float(weights[1]),
        "isolation_forest_rank": float(weights[2]),
    }


def _group_weight_metadata(weights: Sequence[float]) -> dict[str, float]:
    return {
        "prediction_residual_rank": float(weights[0]),
        "graph_rank": float(weights[1]),
        "temporal_profile_rank": float(weights[2]),
    }


def _best_by_metric(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: _best_row(metrics, metric)
        for metric in ("auroc", "auprc", "f1", "precision_at_k", "recall_at_k")
    }


def _best_row(metrics: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    row = max(metrics, key=lambda item: float(item.get(metric, float("-inf"))))
    return {
        "method": row["method"],
        metric: float(row[metric]),
        "auroc": float(row.get("auroc", 0.0)),
        "auprc": float(row.get("auprc", 0.0)),
        "f1": float(row.get("f1", 0.0)),
        "precision_at_k": float(row.get("precision_at_k", 0.0)),
        "recall_at_k": float(row.get("recall_at_k", 0.0)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
