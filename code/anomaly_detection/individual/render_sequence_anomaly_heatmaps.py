#!/usr/bin/env python3
"""
Render per-sequence anomaly heatmaps for the VT-Tiny-MOT test split.

Design choices:
- use one sequence at a time instead of mixing sequences together
- use the ensemble score as trajectory-level anomaly intensity
- project RGB trajectory centers back into image space
- distribute each trajectory's score across its visible RGB points
- aggregate into a fixed spatial grid, then upsample to an image heatmap
- overlay the heatmap on the RGB background frame

This yields an application-friendly visualization:
- where in the scene anomalous motion concentrates
- which regions deserve attention
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from mtf_ba.trajectory_jsonl import iter_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render sequence-level anomaly heatmaps from ensemble scores."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train", "val"],
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--score-csv",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_ensemble" / "mean_scores_test.csv",
        help="CSV containing ensemble trajectory scores. Defaults to mean test scores.",
    )
    parser.add_argument(
        "--trajectory-jsonl",
        type=Path,
        default=Path("outputs")
        / "vt_tiny_mot_individual"
        / "individual_trajectories_test.jsonl",
        help="Object-centric trajectory JSONL used to recover RGB trajectory points.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("..") / "datasets" / "VT-Tiny-MOT",
        help="VT-Tiny-MOT root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "vt_tiny_mot_heatmaps",
        help="Directory where heatmap images and grid summaries will be written.",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=32,
        help="Number of horizontal spatial bins.",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=18,
        help="Number of vertical spatial bins.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Heatmap overlay alpha.",
    )
    parser.add_argument(
        "--top-sequences",
        type=int,
        default=0,
        help=(
            "If > 0, render only the sequences with the largest total anomaly mass. "
            "Use 0 to render every sequence present in the scored trajectories."
        ),
    )
    return parser.parse_args()


def resolve_default_frame_root(data_root: Path, split: str) -> Path:
    if split == "test":
        return data_root / "test2017"
    return data_root / "train2017"


def load_scores(score_csv: Path) -> dict[str, float]:
    df = pd.read_csv(score_csv)
    score_column = [col for col in df.columns if col != "sample_id"][-1]
    return dict(zip(df["sample_id"], df[score_column]))


def group_scored_trajectories_by_sequence(
    trajectory_jsonl: Path,
    score_map: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in iter_trajectory_jsonl(trajectory_jsonl):
        sample_id = trajectory["sample_id"]
        if sample_id not in score_map:
            continue
        grouped[trajectory["sequence"]].append(
            {
                "sample_id": sample_id,
                "score": float(score_map[sample_id]),
                "trajectory": trajectory,
            }
        )
    return dict(grouped)


def extract_rgb_points(trajectory: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for point in trajectory["points"]:
        rgb = point.get("rgb")
        if rgb is None:
            continue
        center = rgb.get("center_xy")
        if center is None:
            continue
        points.append((float(center[0]), float(center[1])))
    return points


def _find_first_image_in_dir(directory: Path) -> Path | None:
    if not directory.exists():
        return None

    images = sorted(directory.glob("*.jpg"))
    if not images:
        images = sorted(directory.glob("*.png"))
    if not images:
        return None
    return images[0]


def _find_background_from_trajectory_files(
    data_root: Path,
    scored_items: list[dict[str, Any]],
) -> Path | None:
    """
    Recover the true RGB frame path from the trajectory JSONL itself.

    Why this fallback is needed:
    - the original implementation assumed backgrounds always live under
      ``data_root/<split>2017/<sequence>/00/*.jpg``.
    - that assumption is fragile across machines because some environments keep
      the exact frame files but place them under a slightly different split
      root, mount point, or copied dataset layout.
    - each trajectory point already stores ``rgb.file`` such as
      ``DJI_0028_3/00/00000.jpg``.

    So here we scan the scored trajectories, grab the first available RGB file
    reference, and try a few robust candidate roots:
    - ``data_root / rgb.file``
    - ``data_root / train2017 / rgb.file``
    - ``data_root / test2017 / rgb.file``

    This makes the renderer resilient even when the caller passes a frame root
    that does not match the actual dataset structure on the target machine.
    """
    for item in scored_items:
        for point in item["trajectory"]["points"]:
            rgb = point.get("rgb")
            if not rgb:
                continue
            rgb_file = rgb.get("file")
            if not rgb_file:
                continue

            relative_path = Path(str(rgb_file))
            candidates = [
                data_root / relative_path,
                data_root / "train2017" / relative_path,
                data_root / "test2017" / relative_path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
    return None


def find_background_image(
    frame_root: Path,
    data_root: Path,
    sequence: str,
    scored_items: list[dict[str, Any]],
) -> Path | None:
    """
    Resolve a background image for one sequence.

    Resolution strategy:
    1. Try the original explicit sequence directory under the chosen split root.
    2. If that fails, recover the path from ``rgb.file`` stored in trajectories.

    This two-step strategy keeps the common case fast while also handling
    cross-machine directory differences more gracefully.
    """
    direct_match = _find_first_image_in_dir(frame_root / sequence / "00")
    if direct_match is not None:
        return direct_match

    return _find_background_from_trajectory_files(
        data_root=data_root,
        scored_items=scored_items,
    )


def accumulate_heat_grid(
    scored_items: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Build a spatial anomaly grid for one sequence.

    Weighting rule:
    - one trajectory has one anomaly score
    - to avoid long trajectories dominating the map purely because they contain
      more points, we distribute the trajectory score equally across its RGB
      points: weight_per_point = score / num_rgb_points
    """
    heat = np.zeros((grid_height, grid_width), dtype=np.float32)
    point_records: list[dict[str, Any]] = []

    for item in scored_items:
        sample_id = item["sample_id"]
        score = float(item["score"])
        rgb_points = extract_rgb_points(item["trajectory"])
        if not rgb_points:
            continue

        point_weight = score / max(len(rgb_points), 1)
        for x, y in rgb_points:
            x = min(max(x, 0.0), image_width - 1.0)
            y = min(max(y, 0.0), image_height - 1.0)
            grid_x = min(int((x / image_width) * grid_width), grid_width - 1)
            grid_y = min(int((y / image_height) * grid_height), grid_height - 1)
            heat[grid_y, grid_x] += point_weight
            point_records.append(
                {
                    "sample_id": sample_id,
                    "x": x,
                    "y": y,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "trajectory_score": score,
                    "point_weight": point_weight,
                }
            )

    return heat, point_records


def make_overlay_figure(
    background_rgb: np.ndarray,
    heat_grid: np.ndarray,
    alpha: float,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(background_rgb)

    if float(np.max(heat_grid)) > 0.0:
        ax.imshow(
            heat_grid,
            cmap="jet",
            alpha=alpha,
            interpolation="bilinear",
            extent=[0, background_rgb.shape[1], background_rgb.shape[0], 0],
        )

    ax.set_title(title)
    ax.set_axis_off()
    return fig


def save_heat_grid_csv(
    path: Path,
    heat_grid: np.ndarray,
) -> None:
    rows = []
    for grid_y in range(heat_grid.shape[0]):
        for grid_x in range(heat_grid.shape[1]):
            rows.append(
                {
                    "grid_y": grid_y,
                    "grid_x": grid_x,
                    "heat_value": float(heat_grid[grid_y, grid_x]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def sequence_mass_summary(grouped: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for sequence, items in grouped.items():
        total_score = float(sum(item["score"] for item in items))
        rows.append(
            {
                "sequence": sequence,
                "num_scored_trajectories": len(items),
                "total_anomaly_mass": total_score,
                "mean_trajectory_score": total_score / max(len(items), 1),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["total_anomaly_mass", "num_scored_trajectories"],
        ascending=[False, False],
    )


def main() -> None:
    args = parse_args()
    score_map = load_scores(args.score_csv.resolve())
    grouped = group_scored_trajectories_by_sequence(
        trajectory_jsonl=args.trajectory_jsonl.resolve(),
        score_map=score_map,
    )

    frame_root = resolve_default_frame_root(args.data_root.resolve(), args.split)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mass_df = sequence_mass_summary(grouped)
    mass_df.to_csv(output_dir / f"sequence_mass_summary_{args.split}.csv", index=False)

    sequences = list(mass_df["sequence"])
    if args.top_sequences > 0:
        sequences = sequences[: args.top_sequences]

    rendered_sequences = []
    skipped_sequences = []

    for sequence in sequences:
        background_path = find_background_image(
            frame_root=frame_root,
            data_root=args.data_root.resolve(),
            sequence=sequence,
            scored_items=grouped[sequence],
        )
        if background_path is None:
            skipped_sequences.append({"sequence": sequence, "reason": "missing_background"})
            continue

        background = np.asarray(Image.open(background_path).convert("RGB"))
        heat_grid, point_records = accumulate_heat_grid(
            scored_items=grouped[sequence],
            image_width=background.shape[1],
            image_height=background.shape[0],
            grid_width=args.grid_width,
            grid_height=args.grid_height,
        )

        figure = make_overlay_figure(
            background_rgb=background,
            heat_grid=heat_grid,
            alpha=args.alpha,
            title=f"{sequence} anomaly heatmap ({args.split})",
        )

        sequence_dir = output_dir / args.split / sequence
        sequence_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(sequence_dir / "anomaly_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

        save_heat_grid_csv(sequence_dir / "heat_grid.csv", heat_grid)
        pd.DataFrame(point_records).to_csv(sequence_dir / "point_contributions.csv", index=False)

        rendered_sequences.append(
            {
                "sequence": sequence,
                "background_path": str(background_path),
                "num_scored_trajectories": len(grouped[sequence]),
                "num_point_contributions": len(point_records),
                "max_heat_value": float(np.max(heat_grid)) if heat_grid.size else 0.0,
                "output_dir": str(sequence_dir),
            }
        )

    summary = {
        "split": args.split,
        "score_csv": str(args.score_csv.resolve()),
        "trajectory_jsonl": str(args.trajectory_jsonl.resolve()),
        "frame_root": str(frame_root),
        "output_dir": str(output_dir),
        "grid_width": args.grid_width,
        "grid_height": args.grid_height,
        "alpha": args.alpha,
        "num_sequences_available": len(grouped),
        "num_sequences_rendered": len(rendered_sequences),
        "num_sequences_skipped": len(skipped_sequences),
        "rendered_sequences": rendered_sequences[:20],
        "skipped_sequences": skipped_sequences[:20],
    }
    with (output_dir / f"heatmap_summary_{args.split}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
