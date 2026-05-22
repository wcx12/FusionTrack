from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
ANOMALY_ROOT = BENCHMARK_ROOT.parent
INDIVIDUAL_ROOT = ANOMALY_ROOT / "individual"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.io import load_jsonl, write_jsonl
from fusiontrack.fused_trajectories import build_fused_trajectories
from runners.prepare_vt_tiny_mot_protocol import (
    DEFAULT_EXPERIMENTS,
    _run_benchmark_script,
    _run_individual_script,
    _write_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a VT-Tiny-MOT holdout anomaly protocol: train on one clean "
            "source split and inject/evaluate anomalies on a separate source split."
        )
    )
    parser.add_argument("--data-root", type=Path, default=ANOMALY_ROOT / "datasets" / "VT-Tiny-MOT")
    parser.add_argument("--output-root", type=Path, default=BENCHMARK_ROOT / "outputs" / "protocol_holdout")
    parser.add_argument("--work-root", type=Path, default=BENCHMARK_ROOT / "outputs" / "vt_tiny_mot_holdout_prepared")
    parser.add_argument(
        "--train-source-split",
        choices=("train", "test", "val"),
        default="train",
        help="Clean VT-Tiny-MOT split used to fit unsupervised detectors.",
    )
    parser.add_argument(
        "--eval-source-split",
        choices=("train", "test", "val"),
        default="test",
        help="Held-out VT-Tiny-MOT split where synthetic anomalies are injected.",
    )
    parser.add_argument(
        "--split-name",
        default=None,
        help="Evaluation split label and output suffix. Defaults to --eval-source-split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--individual-anomaly-fraction", type=float, default=0.1)
    parser.add_argument("--group-anomaly-fraction", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--smoke-max-train", type=int, default=0)
    parser.add_argument("--smoke-max-eval", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.train_source_split == args.eval_source_split:
        raise ValueError(
            "Holdout protocol requires different train and eval source splits "
            f"(got {args.train_source_split!r})."
        )

    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"VT-Tiny-MOT data root not found: {data_root}")

    split_name = str(args.split_name or args.eval_source_split)
    output_root = args.output_root.resolve()
    work_root = args.work_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    train_exports = _extract_split(
        data_root=data_root,
        split=str(args.train_source_split),
        work_root=work_root,
        window_size=int(args.window_size),
        stride=int(args.stride),
    )
    eval_exports = _extract_split(
        data_root=data_root,
        split=str(args.eval_source_split),
        work_root=work_root,
        window_size=int(args.window_size),
        stride=int(args.stride),
    )

    train_trajectories = load_jsonl(train_exports["individual_jsonl"])
    eval_trajectories = load_jsonl(eval_exports["individual_jsonl"])
    train_windows = load_jsonl(train_exports["group_windows_jsonl"])
    eval_windows = load_jsonl(eval_exports["group_windows_jsonl"])
    if args.smoke_max_train > 0:
        train_trajectories = train_trajectories[: args.smoke_max_train]
        train_windows = train_windows[: args.smoke_max_train]
    if args.smoke_max_eval > 0:
        eval_trajectories = eval_trajectories[: args.smoke_max_eval]
        eval_windows = eval_windows[: args.smoke_max_eval]

    fused_train = build_fused_trajectories(train_trajectories)
    fused_eval_clean = build_fused_trajectories(eval_trajectories)
    write_jsonl(output_root / "fused_trajectories_train.jsonl", fused_train)
    write_jsonl(output_root / f"fused_trajectories_{split_name}_clean.jsonl", fused_eval_clean)
    write_jsonl(output_root / "group_windows_train.jsonl", train_windows)
    write_jsonl(output_root / f"group_windows_{split_name}_clean.jsonl", eval_windows)

    _run_benchmark_script(
        "prepare_anomaly_data.py",
        "--level",
        "individual",
        "--input-jsonl",
        str(output_root / f"fused_trajectories_{split_name}_clean.jsonl"),
        "--output-jsonl",
        str(output_root / f"fused_trajectories_{split_name}.jsonl"),
        "--labels-jsonl",
        str(output_root / f"individual_labels_{split_name}.jsonl"),
        "--anomaly-fraction",
        str(args.individual_anomaly_fraction),
        "--seed",
        str(args.seed),
        "--manifest-json",
        str(output_root / f"individual_injection_manifest_{split_name}.json"),
    )
    _run_benchmark_script(
        "prepare_anomaly_data.py",
        "--level",
        "group",
        "--input-jsonl",
        str(output_root / f"group_windows_{split_name}_clean.jsonl"),
        "--output-jsonl",
        str(output_root / f"group_windows_{split_name}.jsonl"),
        "--labels-jsonl",
        str(output_root / f"group_labels_{split_name}.jsonl"),
        "--anomaly-fraction",
        str(args.group_anomaly_fraction),
        "--seed",
        str(args.seed),
        "--manifest-json",
        str(output_root / f"group_injection_manifest_{split_name}.json"),
    )

    individual_config = _holdout_matrix_config(
        label_file=f"individual_labels_{split_name}.jsonl",
        experiments=_experiments_for_split(
            DEFAULT_EXPERIMENTS["individual"],
            split_name=split_name,
            seed=int(args.seed),
        ),
        split_name=split_name,
        seed=int(args.seed),
        require_unique_keys=True,
        require_score_key_match=True,
    )
    group_config = _holdout_matrix_config(
        label_file=f"group_labels_{split_name}.jsonl",
        experiments=_experiments_for_split(
            DEFAULT_EXPERIMENTS["group"],
            split_name=split_name,
            seed=int(args.seed),
        ),
        split_name=split_name,
        seed=int(args.seed),
        key_fields=["sample_id", "window_id"],
        require_unique_keys=True,
        require_score_key_match=True,
    )
    _write_json(output_root / f"individual_{split_name}_matrix.json", individual_config)
    _write_json(output_root / f"group_{split_name}_matrix.json", group_config)

    manifest = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "work_root": str(work_root),
        "train_source_split": str(args.train_source_split),
        "eval_source_split": str(args.eval_source_split),
        "score_split": split_name,
        "seed": int(args.seed),
        "individual_anomaly_fraction": float(args.individual_anomaly_fraction),
        "group_anomaly_fraction": float(args.group_anomaly_fraction),
        "group_window_size": int(args.window_size),
        "group_stride": int(args.stride),
        "num_fused_train": len(fused_train),
        "num_fused_eval_clean": len(fused_eval_clean),
        "num_group_train": len(train_windows),
        "num_group_eval_clean": len(eval_windows),
        "individual_matrix": str(output_root / f"individual_{split_name}_matrix.json"),
        "group_matrix": str(output_root / f"group_{split_name}_matrix.json"),
    }
    _write_json(output_root / "protocol_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _extract_split(
    data_root: Path,
    split: str,
    work_root: Path,
    window_size: int,
    stride: int,
) -> dict[str, Path]:
    trajectory_dir = work_root / f"vt_tiny_mot_trajectories_{split}"
    individual_dir = work_root / f"vt_tiny_mot_individual_{split}"
    group_dir = work_root / f"vt_tiny_mot_group_{split}"

    _run_individual_script(
        "extract_vt_tiny_mot_trajectories.py",
        "--data-root",
        str(data_root),
        "--split",
        split,
        "--output-dir",
        str(trajectory_dir),
    )
    _run_individual_script(
        "export_vt_tiny_mot_individual_trajectories.py",
        "--split",
        split,
        "--csv-path",
        str(trajectory_dir / f"observations_{split}.csv"),
        "--output-dir",
        str(individual_dir),
    )
    _run_individual_script(
        "export_vt_tiny_mot_group_windows.py",
        "--split",
        split,
        "--csv-path",
        str(trajectory_dir / f"observations_{split}.csv"),
        "--output-dir",
        str(group_dir),
        "--sample-mode",
        "window",
        "--window-size",
        str(window_size),
        "--stride",
        str(stride),
    )
    return {
        "individual_jsonl": individual_dir / f"individual_trajectories_{split}.jsonl",
        "group_windows_jsonl": group_dir / f"group_windows_{split}.jsonl",
    }


def _experiments_for_split(
    experiments: list[dict[str, Any]],
    split_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    rewritten = copy.deepcopy(experiments)
    for experiment in rewritten:
        if "seed" in experiment:
            experiment["seed"] = int(seed)
        _replace_if_present(
            experiment,
            "score_jsonl",
            {
                "fused_trajectories_val.jsonl": f"fused_trajectories_{split_name}.jsonl",
                "fused_trajectories_val_clean.jsonl": f"fused_trajectories_{split_name}_clean.jsonl",
            },
        )
        _replace_if_present(
            experiment,
            "score_windows",
            {
                "group_windows_val.jsonl": f"group_windows_{split_name}.jsonl",
                "group_windows_val_clean.jsonl": f"group_windows_{split_name}_clean.jsonl",
            },
        )
    return rewritten


def _replace_if_present(
    experiment: dict[str, Any],
    field: str,
    replacements: dict[str, str],
) -> None:
    value = experiment.get(field)
    if isinstance(value, str) and value in replacements:
        experiment[field] = replacements[value]


def _holdout_matrix_config(
    label_file: str,
    experiments: list[dict[str, Any]],
    split_name: str,
    seed: int,
    key_fields: list[str] | None = None,
    require_unique_keys: bool = False,
    require_score_key_match: bool = False,
) -> dict[str, Any]:
    return {
        "split": split_name,
        "seed": int(seed),
        "label_file": label_file,
        "k": 100,
        "key_fields": list(key_fields or ["sample_id"]),
        "require_unique_keys": bool(require_unique_keys),
        "require_score_key_match": bool(require_score_key_match),
        "experiments": experiments,
    }


if __name__ == "__main__":
    raise SystemExit(main())
