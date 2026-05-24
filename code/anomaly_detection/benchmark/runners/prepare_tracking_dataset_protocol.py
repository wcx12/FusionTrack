from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
ANOMALY_ROOT = BENCHMARK_ROOT.parent
INDIVIDUAL_ROOT = ANOMALY_ROOT / "individual"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.io import load_jsonl, write_jsonl  # noqa: E402
from fusiontrack.fused_trajectories import build_fused_trajectories  # noqa: E402
from runners.prepare_vt_tiny_mot_protocol import (  # noqa: E402
    DEFAULT_EXPERIMENTS,
    _run_benchmark_script,
    _run_individual_script,
    _split_windows_by_sequence,
    _write_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare FusionTrack anomaly benchmark protocol files from an "
            "already-normalized observations_<split>.csv file. This is the "
            "dataset-generic entrypoint for M3OT and MOT-family adapters."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset name for manifests.")
    parser.add_argument(
        "--mode",
        choices=("validation", "holdout"),
        default="validation",
        help="validation splits one observations CSV by sequence; holdout uses separate train/eval CSVs.",
    )
    parser.add_argument(
        "--observations-csv",
        type=Path,
        default=None,
        help="Single source observations CSV for validation mode.",
    )
    parser.add_argument(
        "--train-observations-csv",
        type=Path,
        default=None,
        help="Clean training observations CSV for holdout mode.",
    )
    parser.add_argument(
        "--eval-observations-csv",
        type=Path,
        default=None,
        help="Held-out evaluation observations CSV for holdout mode.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Intermediate export root. Defaults to <output-root>/_prepared.",
    )
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--split-name",
        default=None,
        help="Evaluation split suffix. Defaults to val for validation mode and --eval-split for holdout mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--individual-anomaly-fraction", type=float, default=0.1)
    parser.add_argument("--group-anomaly-fraction", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--smoke-max-train", type=int, default=0)
    parser.add_argument("--smoke-max-eval", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    work_root = (args.work_root or (output_root / "_prepared")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "validation":
        manifest = _prepare_validation(args, output_root, work_root)
    else:
        manifest = _prepare_holdout(args, output_root, work_root)
    _write_json(output_root / "protocol_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _prepare_validation(
    args: argparse.Namespace,
    output_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    if args.observations_csv is None:
        raise ValueError("--observations-csv is required in validation mode.")
    source_csv = args.observations_csv.resolve()
    _require_file(source_csv)
    source_split = str(args.source_split)
    split_name = str(args.split_name or "val")

    exports = _export_from_observations(
        observations_csv=source_csv,
        split=source_split,
        work_root=work_root,
        window_size=int(args.window_size),
        stride=int(args.stride),
    )

    _run_individual_script(
        "split_train_val_by_sequence.py",
        "--train-jsonl",
        str(exports["individual_jsonl"]),
        "--trajectory-output-dir",
        str(work_root / "individual_split"),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
    )
    split_dir = work_root / "individual_split"
    split_summary = json.loads((split_dir / "train_val_split_summary.json").read_text(encoding="utf-8"))
    val_sequences = set(split_summary["val_sequences"])

    train_trajectories = load_jsonl(split_dir / "individual_trajectories_train.jsonl")
    eval_trajectories = load_jsonl(split_dir / "individual_trajectories_val.jsonl")
    group_full = load_jsonl(exports["group_windows_jsonl"])
    train_windows, eval_windows = _split_windows_by_sequence(group_full, val_sequences)
    return _write_protocol_files(
        args=args,
        output_root=output_root,
        split_name=split_name,
        train_trajectories=train_trajectories,
        eval_trajectories=eval_trajectories,
        train_windows=train_windows,
        eval_windows=eval_windows,
        mode="validation",
        source_files={"observations_csv": str(source_csv)},
        extra_manifest={
            "source_split": source_split,
            "val_ratio": float(args.val_ratio),
            "val_sequences": sorted(val_sequences),
        },
    )


def _prepare_holdout(
    args: argparse.Namespace,
    output_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    if args.train_observations_csv is None or args.eval_observations_csv is None:
        raise ValueError("--train-observations-csv and --eval-observations-csv are required in holdout mode.")
    train_csv = args.train_observations_csv.resolve()
    eval_csv = args.eval_observations_csv.resolve()
    _require_file(train_csv)
    _require_file(eval_csv)
    split_name = str(args.split_name or args.eval_split)

    train_exports = _export_from_observations(
        observations_csv=train_csv,
        split=str(args.train_split),
        work_root=work_root / "train_source",
        window_size=int(args.window_size),
        stride=int(args.stride),
    )
    eval_exports = _export_from_observations(
        observations_csv=eval_csv,
        split=str(args.eval_split),
        work_root=work_root / "eval_source",
        window_size=int(args.window_size),
        stride=int(args.stride),
    )
    return _write_protocol_files(
        args=args,
        output_root=output_root,
        split_name=split_name,
        train_trajectories=load_jsonl(train_exports["individual_jsonl"]),
        eval_trajectories=load_jsonl(eval_exports["individual_jsonl"]),
        train_windows=load_jsonl(train_exports["group_windows_jsonl"]),
        eval_windows=load_jsonl(eval_exports["group_windows_jsonl"]),
        mode="holdout",
        source_files={
            "train_observations_csv": str(train_csv),
            "eval_observations_csv": str(eval_csv),
        },
        extra_manifest={
            "train_split": str(args.train_split),
            "eval_split": str(args.eval_split),
        },
    )


def _export_from_observations(
    observations_csv: Path,
    split: str,
    work_root: Path,
    window_size: int,
    stride: int,
) -> dict[str, Path]:
    individual_dir = work_root / "individual"
    group_dir = work_root / "group"
    _run_individual_script(
        "export_vt_tiny_mot_individual_trajectories.py",
        "--split",
        split,
        "--csv-path",
        str(observations_csv),
        "--output-dir",
        str(individual_dir),
    )
    _run_individual_script(
        "export_vt_tiny_mot_group_windows.py",
        "--split",
        split,
        "--csv-path",
        str(observations_csv),
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


def _write_protocol_files(
    args: argparse.Namespace,
    output_root: Path,
    split_name: str,
    train_trajectories: list[dict[str, Any]],
    eval_trajectories: list[dict[str, Any]],
    train_windows: list[dict[str, Any]],
    eval_windows: list[dict[str, Any]],
    mode: str,
    source_files: dict[str, str],
    extra_manifest: dict[str, Any],
) -> dict[str, Any]:
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

    _write_json(
        output_root / f"individual_{split_name}_matrix.json",
        _matrix_config(
            label_file=f"individual_labels_{split_name}.jsonl",
            experiments=_experiments_for_split(DEFAULT_EXPERIMENTS["individual"], split_name, int(args.seed)),
            split_name=split_name,
            seed=int(args.seed),
            require_unique_keys=True,
            require_score_key_match=True,
        ),
    )
    _write_json(
        output_root / f"group_{split_name}_matrix.json",
        _matrix_config(
            label_file=f"group_labels_{split_name}.jsonl",
            experiments=_experiments_for_split(DEFAULT_EXPERIMENTS["group"], split_name, int(args.seed)),
            split_name=split_name,
            seed=int(args.seed),
            key_fields=["sample_id", "window_id"],
            require_unique_keys=True,
            require_score_key_match=True,
        ),
    )

    manifest = {
        "dataset": str(args.dataset),
        "mode": mode,
        "output_root": str(output_root),
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
        "source_files": source_files,
        **extra_manifest,
    }
    return manifest


def _experiments_for_split(
    experiments: list[dict[str, Any]],
    split_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    rewritten = copy.deepcopy(experiments)
    for experiment in rewritten:
        if "seed" in experiment:
            experiment["seed"] = int(seed)
        _replace_if_present(experiment, "score_jsonl", split_name)
        _replace_if_present(experiment, "score_windows", split_name)
    return rewritten


def _replace_if_present(experiment: dict[str, Any], key: str, split_name: str) -> None:
    value = experiment.get(key)
    if not isinstance(value, str):
        return
    experiment[key] = (
        value.replace("_val_clean.", f"_{split_name}_clean.")
        .replace("_val.", f"_{split_name}.")
        .replace("_val_", f"_{split_name}_")
    )


def _matrix_config(
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


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required observations CSV not found: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
