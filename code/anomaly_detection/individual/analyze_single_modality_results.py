#!/usr/bin/env python3
"""
Analyze ensemble outputs for the single-object anomaly detection pipeline.

This script is intentionally analysis-oriented rather than model-oriented.
It helps answer questions such as:

- Which trajectories are ranked highest by the ensemble?
- Which detector scores pushed a sample to the top?
- Are high-score samples concentrated in a few sequences?
- Are high-score samples concentrated in a few categories?

Expected inputs:
- outputs/vt_tiny_mot_ensemble/aligned_scores_<split>.csv
- outputs/vt_tiny_mot_ensemble/mean_scores_<split>.csv
- outputs/vt_tiny_mot_ensemble/max_scores_<split>.csv
- outputs/vt_tiny_mot_individual/individual_trajectories_<split>.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mtf_ba.trajectory_jsonl import iter_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ensemble anomaly scores and summarize top-ranked trajectories."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Which split to analyze.",
    )
    parser.add_argument(
        "--ensemble-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_ensemble",
        help="Directory containing ensemble outputs.",
    )
    parser.add_argument(
        "--trajectory-jsonl",
        type=Path,
        default=None,
        help=(
            "Path to individual_trajectories_<split>.jsonl. "
            "If omitted, a split-aware default path is used."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many top-ranked trajectories to include in detailed analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_analysis",
        help="Directory for analysis CSV/JSON outputs.",
    )
    return parser.parse_args()


def default_trajectory_jsonl(split: str) -> Path:
    if split == "test":
        return (
            Path("outputs")
            / "vt_tiny_mot_individual"
            / "individual_trajectories_test.jsonl"
        )
    if split == "train":
        return (
            Path("outputs")
            / "vt_tiny_mot_individual_split"
            / "individual_trajectories_train.jsonl"
        )
    return (
        Path("outputs")
        / "vt_tiny_mot_individual_split"
        / "individual_trajectories_val.jsonl"
    )


def build_metadata_map(jsonl_path: str | Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for item in iter_trajectory_jsonl(jsonl_path):
        metadata[item["sample_id"]] = {
            "sample_id": item["sample_id"],
            "sequence": item["sequence"],
            "track_id": item["track_id"],
            "category_id": item["category_id"],
            "category_name": item["category_name"],
            "fps": item["fps"],
            "num_points": item["num_points"],
            "visible_rgb_frames": item["visible_rgb_frames"],
            "visible_thermal_frames": item["visible_thermal_frames"],
        }
    return metadata


def load_ensemble_tables(ensemble_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned_scores = pd.read_csv(ensemble_dir / f"aligned_scores_{split}.csv")
    mean_scores = pd.read_csv(ensemble_dir / f"mean_scores_{split}.csv")
    max_scores = pd.read_csv(ensemble_dir / f"max_scores_{split}.csv")
    return aligned_scores, mean_scores, max_scores


def merge_analysis_frame(
    aligned_scores: pd.DataFrame,
    mean_scores: pd.DataFrame,
    max_scores: pd.DataFrame,
    metadata_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    df = aligned_scores.merge(mean_scores, on="sample_id", how="left")
    df = df.merge(max_scores, on="sample_id", how="left")

    # Some pandas versions do not accept `dict_values` views here, so we
    # materialize it as a list for cross-version compatibility.
    metadata_df = pd.DataFrame.from_records(list(metadata_map.values()))
    df = df.merge(metadata_df, on="sample_id", how="left")
    return df


def add_detector_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    detector_columns = [
        column
        for column in df.columns
        if column
        in [
            "route_rgb",
            "speed_rgb",
            "shape_rgb",
            "route_thermal",
            "speed_thermal",
            "shape_thermal",
        ]
    ]

    def top_detector(row: pd.Series) -> str:
        values = row[detector_columns].to_dict()
        return max(values, key=values.get)

    def top_detector_score(row: pd.Series) -> float:
        values = row[detector_columns].to_dict()
        return float(max(values.values()))

    def detector_breakdown(row: pd.Series) -> str:
        values = row[detector_columns].sort_values(ascending=False)
        return "; ".join(f"{key}={value:.6f}" for key, value in values.items())

    df = df.copy()
    df["top_detector"] = df.apply(top_detector, axis=1)
    df["top_detector_score"] = df.apply(top_detector_score, axis=1)
    df["detector_breakdown"] = df.apply(detector_breakdown, axis=1)
    return df


def summarize_group(df: pd.DataFrame, group_column: str, top_k: int) -> pd.DataFrame:
    top_df = df.sort_values("mean_score", ascending=False).head(top_k)
    summary = (
        top_df.groupby(group_column)
        .agg(
            top_k_count=("sample_id", "count"),
            mean_of_mean_score=("mean_score", "mean"),
            mean_of_max_score=("max_score", "mean"),
        )
        .sort_values(["top_k_count", "mean_of_mean_score"], ascending=[False, False])
        .reset_index()
    )
    return summary


def build_top_k_records(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    columns = [
        "sample_id",
        "sequence",
        "track_id",
        "category_name",
        "num_points",
        "visible_rgb_frames",
        "visible_thermal_frames",
        "mean_score",
        "max_score",
        "top_detector",
        "top_detector_score",
        "route_rgb",
        "speed_rgb",
        "shape_rgb",
        "route_thermal",
        "speed_thermal",
        "shape_thermal",
        "detector_breakdown",
    ]
    return df.sort_values("mean_score", ascending=False).head(top_k)[columns]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    ensemble_dir = args.ensemble_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_jsonl = (
        args.trajectory_jsonl.resolve()
        if args.trajectory_jsonl is not None
        else default_trajectory_jsonl(args.split).resolve()
    )

    aligned_scores, mean_scores, max_scores = load_ensemble_tables(
        ensemble_dir=ensemble_dir,
        split=args.split,
    )
    metadata_map = build_metadata_map(trajectory_jsonl)
    analysis_df = merge_analysis_frame(
        aligned_scores=aligned_scores,
        mean_scores=mean_scores,
        max_scores=max_scores,
        metadata_map=metadata_map,
    )
    analysis_df = add_detector_summary_columns(analysis_df)

    top_k_df = build_top_k_records(analysis_df, top_k=args.top_k)
    top_sequence_df = summarize_group(analysis_df, group_column="sequence", top_k=args.top_k)
    top_category_df = summarize_group(
        analysis_df, group_column="category_name", top_k=args.top_k
    )

    analysis_df.to_csv(output_dir / f"analysis_full_{args.split}.csv", index=False)
    top_k_df.to_csv(output_dir / f"top_{args.top_k}_{args.split}.csv", index=False)
    top_sequence_df.to_csv(
        output_dir / f"top_{args.top_k}_sequences_{args.split}.csv",
        index=False,
    )
    top_category_df.to_csv(
        output_dir / f"top_{args.top_k}_categories_{args.split}.csv",
        index=False,
    )

    summary = {
        "split": args.split,
        "trajectory_jsonl": str(trajectory_jsonl),
        "ensemble_dir": str(ensemble_dir),
        "output_dir": str(output_dir),
        "num_samples_analyzed": int(len(analysis_df)),
        "top_k": args.top_k,
        "top_mean_sample_ids": top_k_df["sample_id"].head(10).tolist(),
        "top_sequences": top_sequence_df.head(10).to_dict(orient="records"),
        "top_categories": top_category_df.head(10).to_dict(orient="records"),
    }
    save_json(output_dir / f"analysis_summary_{args.split}.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
