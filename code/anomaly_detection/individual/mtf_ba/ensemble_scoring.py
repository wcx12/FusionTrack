from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from kneed import KneeLocator


FEATURES = [
    "route_rgb",
    "speed_rgb",
    "shape_rgb",
    "route_thermal",
    "speed_thermal",
    "shape_thermal",
]


def load_final_scores(path: str | Path) -> dict[str, float]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_pickle(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def save_jsonl_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def jaccard_similarity(score_a: np.ndarray, score_b: np.ndarray, top_k: int = 500) -> float:
    """
    Baseline-style top-k Jaccard similarity between two score lists.

    We compare the overlap of the top-k ranked samples instead of comparing raw
    score magnitudes directly. This mirrors the baseline's emphasis on ranking
    stability across complementary detectors.
    """
    rank_a = stats.rankdata(score_a)
    rank_b = stats.rankdata(score_b)
    top_rank_a = set(np.argsort(-rank_a)[:top_k])
    top_rank_b = set(np.argsort(-rank_b)[:top_k])
    union = top_rank_a.union(top_rank_b)
    if not union:
        return 1.0
    return len(top_rank_a.intersection(top_rank_b)) / len(union)


def align_score_dicts(
    score_dicts: dict[str, dict[str, float]],
    fill_value: float | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Align all detector score dictionaries by sample_id.

    Default behavior:
    - take the union of sample IDs across detectors
    - if a detector is missing a sample, fill with that detector's median score

    Median fill keeps every sample aligned for ensemble use while being less
    aggressive than zero-fill.
    """
    all_sample_ids = sorted({sample_id for scores in score_dicts.values() for sample_id in scores})
    aligned: dict[str, list[float]] = {}

    for feature_name, score_map in score_dicts.items():
        values = list(score_map.values())
        detector_fill = float(np.median(values)) if values else 0.0
        if fill_value is not None:
            detector_fill = fill_value
        aligned[feature_name] = [
            float(score_map.get(sample_id, detector_fill)) for sample_id in all_sample_ids
        ]

    score_df = pd.DataFrame(aligned, index=all_sample_ids)
    score_df.index.name = "sample_id"
    return all_sample_ids, score_df


def compute_mean_score(score_df: pd.DataFrame) -> pd.Series:
    inverse_rank_lists = np.stack(
        [1.0 / stats.rankdata(-score_df[col].to_numpy(dtype=float)) for col in score_df.columns],
        axis=-1,
    )
    mean_score = np.mean(inverse_rank_lists, axis=-1)
    return pd.Series(mean_score, index=score_df.index, name="mean_score")


def compute_max_score(score_df: pd.DataFrame) -> pd.Series:
    inverse_rank_lists = np.stack(
        [1.0 / stats.rankdata(-score_df[col].to_numpy(dtype=float)) for col in score_df.columns],
        axis=-1,
    )
    max_score = np.max(inverse_rank_lists, axis=-1)
    return pd.Series(max_score, index=score_df.index, name="max_score")


def complementary_elimination(
    score_df: pd.DataFrame,
    top_k: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adaptation of the baseline complementary elimination procedure.

    At each iteration:
    1. compute the current ensemble score using MAX over inverse-rank detector scores
    2. measure how much the ensemble ranking changes if each detector is removed
    3. remove the detector whose removal causes the least change

    Outputs:
    - metrics_df: one row per elimination step
    - tau_df: detector-specific Jaccard scores collected before each removal
    """
    excluded_keys: list[str] = []
    original_columns = list(score_df.columns)
    metrics_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []

    for iteration in range(len(original_columns) + 1):
        current_columns = [col for col in original_columns if col not in excluded_keys]
        if len(current_columns) <= 1:
            break

        current_array = score_df[current_columns].to_numpy(dtype=float)
        max_s_rank = np.max(
            np.stack(
                [1.0 / stats.rankdata(-current_array[:, idx]) for idx in range(current_array.shape[1])],
                axis=-1,
            ),
            axis=-1,
        )

        detector_similarities: dict[str, float] = {}
        for column in current_columns:
            remaining_columns = [col for col in current_columns if col != column]
            if len(remaining_columns) == 0:
                detector_similarities[column] = 0.0
                continue

            reduced_array = score_df[remaining_columns].to_numpy(dtype=float)
            max_reduced_rank = np.max(
                np.stack(
                    [
                        1.0 / stats.rankdata(-reduced_array[:, idx])
                        for idx in range(reduced_array.shape[1])
                    ],
                    axis=-1,
                ),
                axis=-1,
            )
            detector_similarities[column] = jaccard_similarity(
                max_s_rank,
                max_reduced_rank,
                top_k=top_k,
            )

        tau_rows.append({"iteration": iteration, **detector_similarities})

        removed_column = min(detector_similarities, key=detector_similarities.get)
        excluded_keys.append(removed_column)
        subset_columns = [col for col in original_columns if col not in excluded_keys]
        subset_array = score_df[subset_columns].to_numpy(dtype=float)

        max_subset_rank = np.max(
            np.stack(
                [1.0 / stats.rankdata(-subset_array[:, idx]) for idx in range(subset_array.shape[1])],
                axis=-1,
            ),
            axis=-1,
        )
        jac_max = jaccard_similarity(max_s_rank, max_subset_rank, top_k=top_k)

        metrics_rows.append(
            {
                "iteration": iteration,
                "removed": removed_column,
                "remaining_detectors": subset_columns,
                "num_remaining_detectors": len(subset_columns),
                "jaccard_max_before_after": jac_max,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    tau_df = pd.DataFrame(tau_rows)
    return metrics_df, tau_df


def choose_stopping_point(metrics_df: pd.DataFrame) -> int | None:
    if metrics_df.empty or len(metrics_df) < 2:
        return None

    xdata = list(range(1, len(metrics_df) + 1))
    ydata = metrics_df["jaccard_max_before_after"].to_numpy(dtype=float)
    if len(ydata) >= 3:
        ydata = savgol_filter(ydata, window_length=3, polyorder=1)

    kneedle = KneeLocator(
        xdata,
        ydata,
        S=1,
        curve="concave",
        direction="decreasing",
    )
    if kneedle.knee is None:
        return None
    return int(kneedle.knee)


def build_score_records(
    sample_ids: list[str],
    score_series: pd.Series,
    source: str,
    component_scores: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        record = {
            "sample_id": sample_id,
            "source": source,
            "score": float(score_series.loc[sample_id]),
        }
        if component_scores is not None:
            record["component_scores"] = {
                key: float(value) for key, value in component_scores.loc[sample_id].to_dict().items()
            }
        records.append(record)
    return records


def run_ensemble(
    score_root_dir: str | Path,
    split: str,
    output_dir: str | Path,
    features: list[str] | None = None,
    top_k: int = 500,
) -> dict[str, Any]:
    """
    Run baseline-style ensemble on detector final scores.

    Expected input layout:
      score_root_dir/<feature_name>/<split>/final_scores.pkl
    """
    score_root_dir = Path(score_root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = features or FEATURES

    score_dicts: dict[str, dict[str, float]] = {}
    input_paths: dict[str, str] = {}
    for feature_name in features:
        score_path = score_root_dir / feature_name / split / "final_scores.pkl"
        score_dicts[feature_name] = load_final_scores(score_path)
        input_paths[feature_name] = str(score_path)

    sample_ids, score_df = align_score_dicts(score_dicts)
    mean_score = compute_mean_score(score_df)
    max_score = compute_max_score(score_df)
    metrics_df, tau_df = complementary_elimination(score_df, top_k=top_k)
    stopping_point = choose_stopping_point(metrics_df)

    mean_records = build_score_records(
        sample_ids=sample_ids,
        score_series=mean_score,
        source="mean_ensemble",
        component_scores=score_df,
    )
    max_records = build_score_records(
        sample_ids=sample_ids,
        score_series=max_score,
        source="max_ensemble",
        component_scores=score_df,
    )

    score_df.to_csv(output_dir / f"aligned_scores_{split}.csv")
    mean_score.to_csv(output_dir / f"mean_scores_{split}.csv", header=True)
    max_score.to_csv(output_dir / f"max_scores_{split}.csv", header=True)
    metrics_df.to_csv(output_dir / f"complementary_metrics_{split}.csv", index=False)
    tau_df.to_csv(output_dir / f"complementary_tau_{split}.csv", index=False)

    save_pickle(output_dir / f"aligned_scores_{split}.pkl", score_df)
    save_pickle(output_dir / f"mean_scores_{split}.pkl", mean_score.to_dict())
    save_pickle(output_dir / f"max_scores_{split}.pkl", max_score.to_dict())
    save_jsonl_records(output_dir / f"mean_score_records_{split}.jsonl", mean_records)
    save_jsonl_records(output_dir / f"max_score_records_{split}.jsonl", max_records)

    summary = {
        "split": split,
        "input_paths": input_paths,
        "num_samples": len(sample_ids),
        "features": features,
        "top_k": top_k,
        "stopping_point": stopping_point,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / f"ensemble_summary_{split}.json", summary)
    return summary
