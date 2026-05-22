from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
ANOMALY_ROOT = BENCHMARK_ROOT.parent
INDIVIDUAL_ROOT = ANOMALY_ROOT / "individual"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.io import load_jsonl, write_jsonl
from fusiontrack.fused_trajectories import build_fused_trajectories


DEFAULT_EXPERIMENTS = {
    "individual": [
        {
            "name": "fusiontrack_individual_nn",
            "task": "fusiontrack_individual",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "n_neighbors": 1,
        },
        {
            "name": "fusiontrack_individual_ensemble",
            "task": "fusiontrack_individual_ensemble",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "n_neighbors": 1,
            "nearest_weight": 0.4,
            "lof_weight": 0.35,
            "iforest_weight": 0.25,
            "contamination": 0.05,
            "seed": 42,
        },
        {
            "name": "fusiontrack_individual_ensemble_calibrated",
            "task": "fusiontrack_individual_ensemble",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "n_neighbors": 1,
            "nearest_weight": 0.4,
            "lof_weight": 0.35,
            "iforest_weight": 0.25,
            "contamination": 0.05,
            "seed": 42,
            "calibration_columns": ["mean_speed", "duration_frames", "num_points"],
            "calibration_bins": 4,
            "calibration_global_weight": 0.5,
        },
        {
            "name": "fusiontrack_individual_ensemble_tuned_auprc",
            "task": "fusiontrack_individual_ensemble",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "n_neighbors": 1,
            "nearest_weight": 0.45,
            "lof_weight": 0.45,
            "iforest_weight": 0.10,
            "contamination": 0.05,
            "seed": 42,
            "calibration_columns": ["mean_speed", "duration_frames", "num_points"],
            "calibration_bins": 4,
            "calibration_global_weight": 0.3,
            "selection_scope": "validation_score_grid",
        },
        {
            "name": "fusiontrack_individual_ensemble_tuned_topk",
            "task": "fusiontrack_individual_ensemble",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "n_neighbors": 1,
            "nearest_weight": 0.60,
            "lof_weight": 0.30,
            "iforest_weight": 0.10,
            "contamination": 0.05,
            "seed": 42,
            "calibration_columns": ["mean_speed", "duration_frames", "num_points"],
            "calibration_bins": 4,
            "calibration_global_weight": 0.3,
            "selection_scope": "validation_score_grid",
        },
        {
            "name": "fusiontrack_individual_context",
            "task": "fusiontrack_individual_context",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val_clean.jsonl",
            "n_neighbors": 1,
        },
        {
            "name": "individual_complementary_cetrajad_proxy",
            "task": "individual_complementary",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "n_neighbors": 1,
            "contamination": 0.05,
            "seed": 42,
        },
        {
            "name": "individual_trajectory_lm_ngram_proxy",
            "task": "individual_trajectory_lm",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "ngram_order": 2,
            "alpha": 1.0,
            "grid_size": 16,
            "seed": 42,
        },
        {
            "name": "individual_physics_kinematic_proxy",
            "task": "individual_physics",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
        },
        {
            "name": "individual_iforest",
            "task": "individual_classical",
            "method": "isolation_forest",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "contamination": 0.05,
            "seed": 42,
        },
        {
            "name": "individual_lof",
            "task": "individual_classical",
            "method": "lof",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "contamination": 0.05,
            "seed": 42,
        },
        {
            "name": "individual_ocsvm",
            "task": "individual_classical",
            "method": "one_class_svm",
            "train_jsonl": "fused_trajectories_train.jsonl",
            "score_jsonl": "fused_trajectories_val.jsonl",
            "contamination": 0.05,
            "seed": 42,
        },
    ],
    "group": [
        {
            "name": "group_prediction_linear",
            "task": "group_prediction",
            "score_windows": "group_windows_val.jsonl",
        },
        {
            "name": "group_iforest",
            "task": "group_classical",
            "method": "isolation_forest",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "contamination": 0.05,
            "seed": 42,
        },
        {
            "name": "group_lof",
            "task": "group_classical",
            "method": "lof",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "contamination": 0.05,
            "seed": 42,
        },
        {
            "name": "group_ocsvm",
            "task": "group_classical",
            "method": "one_class_svm",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "contamination": 0.05,
            "seed": 42,
        },
        {
            "name": "group_temporal_graph_ae_proxy",
            "task": "group_temporal_autoencoder",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "n_components": 3,
            "seed": 42,
        },
        {
            "name": "fusiontrack_group_temporal_knn",
            "task": "fusiontrack_group_temporal_knn",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "n_neighbors": 3,
        },
        {
            "name": "fusiontrack_group_hybrid",
            "task": "fusiontrack_group_hybrid",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "n_neighbors": 3,
            "k_neighbors": 3,
            "rho_p": 80.0,
            "rho_v": 20.0,
            "eta": 0.5,
            "prediction_weight": 0.6,
            "graph_weight": 0.2,
            "temporal_weight": 0.2,
            "invert_graph_rank": True,
            "invert_temporal_rank": True,
        },
        {
            "name": "fusiontrack_group_hybrid_gated",
            "task": "fusiontrack_group_hybrid",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "n_neighbors": 3,
            "k_neighbors": 3,
            "rho_p": 80.0,
            "rho_v": 20.0,
            "eta": 0.5,
            "prediction_weight": 0.6,
            "graph_weight": 0.2,
            "temporal_weight": 0.2,
            "invert_graph_rank": True,
            "invert_temporal_rank": True,
            "use_residual_gate": True,
            "residual_gate_power": 2.0,
            "residual_gate_floor": 0.05,
        },
        {
            "name": "fusiontrack_group_hybrid_tuned_auroc_topk",
            "task": "fusiontrack_group_hybrid",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "n_neighbors": 3,
            "k_neighbors": 3,
            "rho_p": 80.0,
            "rho_v": 20.0,
            "eta": 0.5,
            "prediction_weight": 0.50,
            "graph_weight": 0.25,
            "temporal_weight": 0.25,
            "invert_graph_rank": True,
            "invert_temporal_rank": True,
            "use_residual_gate": False,
            "selection_scope": "validation_score_grid",
        },
        {
            "name": "fusiontrack_group_hybrid_tuned_auprc_f1",
            "task": "fusiontrack_group_hybrid",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "n_neighbors": 3,
            "k_neighbors": 3,
            "rho_p": 80.0,
            "rho_v": 20.0,
            "eta": 0.5,
            "prediction_weight": 0.60,
            "graph_weight": 0.30,
            "temporal_weight": 0.10,
            "invert_graph_rank": True,
            "invert_temporal_rank": True,
            "use_residual_gate": False,
            "selection_scope": "validation_score_grid",
        },
        {
            "name": "fusiontrack_group_hybrid_tuned_fine_auprc",
            "task": "fusiontrack_group_hybrid",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "n_neighbors": 3,
            "k_neighbors": 3,
            "rho_p": 80.0,
            "rho_v": 20.0,
            "eta": 0.5,
            "prediction_weight": 0.47,
            "graph_weight": 0.41,
            "temporal_weight": 0.12,
            "invert_graph_rank": True,
            "invert_temporal_rank": True,
            "use_residual_gate": False,
            "selection_scope": "validation_fine_weight_search",
        },
        {
            "name": "fusiontrack_group_hybrid_tuned_fine_topk",
            "task": "fusiontrack_group_hybrid",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "n_neighbors": 3,
            "k_neighbors": 3,
            "rho_p": 80.0,
            "rho_v": 20.0,
            "eta": 0.5,
            "prediction_weight": 0.45,
            "graph_weight": 0.43,
            "temporal_weight": 0.12,
            "invert_graph_rank": True,
            "invert_temporal_rank": True,
            "use_residual_gate": False,
            "selection_scope": "validation_fine_weight_search",
        },
        {
            "name": "fusiontrack_group_hybrid_tuned_fine_f1",
            "task": "fusiontrack_group_hybrid",
            "train_windows": "group_windows_train.jsonl",
            "score_windows": "group_windows_val.jsonl",
            "n_neighbors": 3,
            "k_neighbors": 3,
            "rho_p": 80.0,
            "rho_v": 20.0,
            "eta": 0.5,
            "prediction_weight": 0.46,
            "graph_weight": 0.42,
            "temporal_weight": 0.12,
            "invert_graph_rank": True,
            "invert_temporal_rank": True,
            "use_residual_gate": False,
            "selection_scope": "validation_fine_weight_search",
        },
        {
            "name": "fusiontrack_group_graph",
            "task": "fusiontrack_group",
            "score_windows": "group_windows_val.jsonl",
            "k_neighbors": 3,
            "rho_p": 80.0,
            "rho_v": 20.0,
            "eta": 0.5,
        },
    ],
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare VT-Tiny-MOT anomaly benchmark protocol files and matrix configs."
    )
    parser.add_argument("--data-root", type=Path, default=ANOMALY_ROOT / "datasets" / "VT-Tiny-MOT")
    parser.add_argument("--output-root", type=Path, default=BENCHMARK_ROOT / "outputs" / "protocol")
    parser.add_argument("--work-root", type=Path, default=BENCHMARK_ROOT / "outputs" / "vt_tiny_mot_prepared")
    parser.add_argument(
        "--source-split",
        choices=("train", "test", "val"),
        default="train",
        help="Dataset split to extract before sequence-level train/val splitting.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--individual-anomaly-fraction", type=float, default=0.1)
    parser.add_argument("--group-anomaly-fraction", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--smoke-max-train", type=int, default=0)
    parser.add_argument("--smoke-max-val", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"VT-Tiny-MOT data root not found: {data_root}")

    output_root = args.output_root.resolve()
    work_root = args.work_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    trajectory_dir = work_root / "vt_tiny_mot_trajectories"
    individual_dir = work_root / "vt_tiny_mot_individual"
    individual_split_dir = work_root / "vt_tiny_mot_individual_split"
    group_full_dir = work_root / "vt_tiny_mot_group_full"

    _extract_and_export(args, data_root, trajectory_dir, individual_dir, individual_split_dir, group_full_dir)

    split_summary = json.loads(
        (individual_split_dir / "train_val_split_summary.json").read_text(encoding="utf-8")
    )
    val_sequences = set(split_summary["val_sequences"])

    train_trajectories = load_jsonl(individual_split_dir / "individual_trajectories_train.jsonl")
    val_trajectories = load_jsonl(individual_split_dir / "individual_trajectories_val.jsonl")
    if args.smoke_max_train > 0:
        train_trajectories = train_trajectories[: args.smoke_max_train]
    if args.smoke_max_val > 0:
        val_trajectories = val_trajectories[: args.smoke_max_val]

    fused_train = build_fused_trajectories(train_trajectories)
    fused_val_clean = build_fused_trajectories(val_trajectories)
    write_jsonl(output_root / "fused_trajectories_train.jsonl", fused_train)
    write_jsonl(output_root / "fused_trajectories_val_clean.jsonl", fused_val_clean)

    group_train_full = load_jsonl(group_full_dir / f"group_windows_{args.source_split}.jsonl")
    group_train, group_val = _split_windows_by_sequence(group_train_full, val_sequences)
    if args.smoke_max_train > 0:
        group_train = group_train[: args.smoke_max_train]
    if args.smoke_max_val > 0:
        group_val = group_val[: args.smoke_max_val]
    write_jsonl(output_root / "group_windows_train.jsonl", group_train)
    write_jsonl(output_root / "group_windows_val_clean.jsonl", group_val)

    _run_benchmark_script(
        "prepare_anomaly_data.py",
        "--level",
        "individual",
        "--input-jsonl",
        str(output_root / "fused_trajectories_val_clean.jsonl"),
        "--output-jsonl",
        str(output_root / "fused_trajectories_val.jsonl"),
        "--labels-jsonl",
        str(output_root / "individual_labels_val.jsonl"),
        "--anomaly-fraction",
        str(args.individual_anomaly_fraction),
        "--seed",
        str(args.seed),
        "--manifest-json",
        str(output_root / "individual_injection_manifest.json"),
    )
    _run_benchmark_script(
        "prepare_anomaly_data.py",
        "--level",
        "group",
        "--input-jsonl",
        str(output_root / "group_windows_val_clean.jsonl"),
        "--output-jsonl",
        str(output_root / "group_windows_val.jsonl"),
        "--labels-jsonl",
        str(output_root / "group_labels_val.jsonl"),
        "--anomaly-fraction",
        str(args.group_anomaly_fraction),
        "--seed",
        str(args.seed),
        "--manifest-json",
        str(output_root / "group_injection_manifest.json"),
    )

    individual_config = _matrix_config(
        label_file="individual_labels_val.jsonl",
        experiments=DEFAULT_EXPERIMENTS["individual"],
        seed=args.seed,
        require_unique_keys=True,
        require_score_key_match=True,
    )
    group_config = _matrix_config(
        label_file="group_labels_val.jsonl",
        experiments=DEFAULT_EXPERIMENTS["group"],
        seed=args.seed,
        key_fields=["sample_id", "window_id"],
        require_unique_keys=True,
        require_score_key_match=True,
    )
    _write_json(output_root / "individual_val_matrix.json", individual_config)
    _write_json(output_root / "group_val_matrix.json", group_config)

    manifest = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "work_root": str(work_root),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "num_fused_train": len(fused_train),
        "num_fused_val_clean": len(fused_val_clean),
        "num_group_train": len(group_train),
        "num_group_val_clean": len(group_val),
        "individual_matrix": str(output_root / "individual_val_matrix.json"),
        "group_matrix": str(output_root / "group_val_matrix.json"),
        "val_sequences": sorted(val_sequences),
    }
    _write_json(output_root / "protocol_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _extract_and_export(
    args: argparse.Namespace,
    data_root: Path,
    trajectory_dir: Path,
    individual_dir: Path,
    individual_split_dir: Path,
    group_full_dir: Path,
) -> None:
    source_split = str(args.source_split)
    _run_individual_script(
        "extract_vt_tiny_mot_trajectories.py",
        "--data-root",
        str(data_root),
        "--split",
        source_split,
        "--output-dir",
        str(trajectory_dir),
    )
    _run_individual_script(
        "export_vt_tiny_mot_individual_trajectories.py",
        "--split",
        source_split,
        "--csv-path",
        str(trajectory_dir / f"observations_{source_split}.csv"),
        "--output-dir",
        str(individual_dir),
    )
    _run_individual_script(
        "split_train_val_by_sequence.py",
        "--train-jsonl",
        str(individual_dir / f"individual_trajectories_{source_split}.jsonl"),
        "--trajectory-output-dir",
        str(individual_split_dir),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
    )
    _run_individual_script(
        "export_vt_tiny_mot_group_windows.py",
        "--split",
        source_split,
        "--csv-path",
        str(trajectory_dir / f"observations_{source_split}.csv"),
        "--output-dir",
        str(group_full_dir),
        "--sample-mode",
        "window",
        "--window-size",
        str(args.window_size),
        "--stride",
        str(args.stride),
    )


def _run_individual_script(script_name: str, *args: str) -> None:
    _run_command([sys.executable, script_name, *args], cwd=INDIVIDUAL_ROOT)


def _run_benchmark_script(script_name: str, *args: str) -> None:
    _run_command([sys.executable, str(BENCHMARK_ROOT / "runners" / script_name), *args], cwd=BENCHMARK_ROOT)


def _run_command(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}\n{result.stderr}"
        )


def _split_windows_by_sequence(
    windows: Iterable[dict[str, Any]],
    val_sequences: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for window in windows:
        sequence = str(window.get("sequence", ""))
        if sequence in val_sequences:
            val.append(window)
        else:
            train.append(window)
    return train, val


def _matrix_config(
    label_file: str,
    experiments: list[dict[str, Any]],
    seed: int,
    key_fields: list[str] | None = None,
    require_unique_keys: bool = False,
    require_score_key_match: bool = False,
) -> dict[str, Any]:
    return {
        "split": "val",
        "seed": int(seed),
        "label_file": label_file,
        "k": 100,
        "key_fields": list(key_fields or ["sample_id"]),
        "require_unique_keys": bool(require_unique_keys),
        "require_score_key_match": bool(require_score_key_match),
        "experiments": experiments,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
