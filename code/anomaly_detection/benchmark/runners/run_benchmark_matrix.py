from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from baselines.group_classical import run_classical_baseline as run_group_classical
from baselines.group_prediction import run_prediction_baseline as run_group_prediction
from baselines.group_temporal_autoencoder import run_temporal_graph_autoencoder
from baselines.complementary_trajectory import run_complementary_baseline
from baselines.individual_classical import run_classical_baseline as run_individual_classical
from baselines.physics_informed import run_physics_informed_baseline
from baselines.trajectory_language_model import run_ngram_language_model
from evaluation.io import load_jsonl, write_jsonl
from evaluation.reporting import evaluate_score_file, summarize_metric_files
from fusiontrack.context_aware_individual import run_context_aware_fusiontrack_baseline
from fusiontrack.group_scoring import score_group_windows
from fusiontrack.group_temporal_profile import (
    run_group_hybrid_fusiontrack,
    run_group_temporal_knn,
)
from fusiontrack.individual_scoring import (
    run_individual_fusiontrack_baseline,
    run_individual_fusiontrack_ensemble,
)


SUPPORTED_TASKS = (
    "existing_scores",
    "individual_complementary",
    "individual_physics",
    "individual_trajectory_lm",
    "individual_classical",
    "group_classical",
    "group_prediction",
    "group_temporal_autoencoder",
    "fusiontrack_individual",
    "fusiontrack_individual_ensemble",
    "fusiontrack_individual_context",
    "fusiontrack_group",
    "fusiontrack_group_temporal_knn",
    "fusiontrack_group_hybrid",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a JSON-defined anomaly benchmark matrix and summarize metrics."
    )
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config_json
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Benchmark matrix config must be a JSON object")

    output_dir = args.output_dir
    score_dir = output_dir / "scores"
    metric_dir = output_dir / "metrics"
    summary_csv = output_dir / "summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    label_file = _resolve_path(_required(config, "label_file"), config_path.parent)
    split = str(config.get("split", "unknown"))
    seed = int(config.get("seed", 42))
    key_fields = tuple(config.get("key_fields", ["sample_id"]))
    default_require_unique_keys = bool(config.get("require_unique_keys", False))
    default_require_score_key_match = bool(
        config.get("require_score_key_match", False)
    )
    k = config.get("k")
    experiments = config.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("Config field 'experiments' must be a list")

    manifest_runs: list[dict[str, Any]] = []
    metric_files: list[Path] = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each experiment must be a JSON object")
        name = _required(experiment, "name")
        task = _required(experiment, "task")
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task '{task}'. Expected one of {SUPPORTED_TASKS}.")

        score_path = score_dir / f"{_safe_name(name)}.jsonl"
        metric_path = metric_dir / f"{_safe_name(name)}.json"
        require_unique_keys = bool(
            experiment.get("require_unique_keys", default_require_unique_keys)
        )
        require_score_key_match = bool(
            experiment.get(
                "require_score_key_match",
                default_require_score_key_match,
            )
        )
        _run_experiment(
            experiment=experiment,
            task=task,
            score_path=score_path,
            config_dir=config_path.parent,
        )
        metrics = evaluate_score_file(
            score_path=score_path,
            label_path=label_file,
            key_fields=key_fields,
            k=int(k) if k is not None else None,
            require_unique_keys=require_unique_keys,
            require_score_key_match=require_score_key_match,
        )
        metrics.update(
            {
                "method": str(name),
                "task": str(task),
                "source": str(score_path),
                "split": split,
                "seed": seed,
            }
        )
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metric_files.append(metric_path)
        manifest_runs.append(
            {
                "name": str(name),
                "task": str(task),
                "score_file": str(score_path),
                "metrics_file": str(metric_path),
                "key_fields": list(key_fields),
                "require_unique_keys": require_unique_keys,
                "require_score_key_match": require_score_key_match,
            }
        )

    summarize_metric_files(metric_files, output_csv=summary_csv)
    manifest = {
        "config_json": str(config_path),
        "label_file": str(label_file),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "split": split,
        "seed": seed,
        "key_fields": list(key_fields),
        "require_unique_keys": default_require_unique_keys,
        "require_score_key_match": default_require_score_key_match,
        "runs": manifest_runs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _run_experiment(
    experiment: dict[str, Any],
    task: str,
    score_path: Path,
    config_dir: Path,
) -> None:
    if task == "existing_scores":
        rows = load_jsonl(_resolve_path(_required(experiment, "score_file"), config_dir))
    elif task == "individual_classical":
        rows = run_individual_classical(
            load_jsonl(_resolve_path(_required(experiment, "train_jsonl"), config_dir)),
            load_jsonl(_resolve_path(_required(experiment, "score_jsonl"), config_dir)),
            method=str(experiment.get("method", "isolation_forest")),
            seed=int(experiment.get("seed", 42)),
            contamination=float(experiment.get("contamination", 0.05)),
        )
    elif task == "individual_complementary":
        rows = run_complementary_baseline(
            load_jsonl(_resolve_path(_required(experiment, "train_jsonl"), config_dir)),
            load_jsonl(_resolve_path(_required(experiment, "score_jsonl"), config_dir)),
            seed=int(experiment.get("seed", 42)),
            contamination=float(experiment.get("contamination", 0.05)),
            n_neighbors=int(experiment.get("n_neighbors", 1)),
        )
    elif task == "individual_trajectory_lm":
        rows = run_ngram_language_model(
            load_jsonl(_resolve_path(_required(experiment, "train_jsonl"), config_dir)),
            load_jsonl(_resolve_path(_required(experiment, "score_jsonl"), config_dir)),
            ngram_order=int(experiment.get("ngram_order", 2)),
            alpha=float(experiment.get("alpha", 1.0)),
            grid_size=int(experiment.get("grid_size", 16)),
            seed=int(experiment.get("seed", 42)),
        )
    elif task == "individual_physics":
        rows = run_physics_informed_baseline(
            load_jsonl(_resolve_path(_required(experiment, "train_jsonl"), config_dir)),
            load_jsonl(_resolve_path(_required(experiment, "score_jsonl"), config_dir)),
        )
    elif task == "group_classical":
        score_windows = load_jsonl(_resolve_path(_required(experiment, "score_windows"), config_dir))
        train_windows = load_jsonl(_resolve_path(_required(experiment, "train_windows"), config_dir))
        rows = run_group_classical(
            train_windows,
            score_windows,
            method=str(experiment.get("method", "isolation_forest")),
            seed=int(experiment.get("seed", 42)),
            contamination=float(experiment.get("contamination", 0.05)),
        )
    elif task == "group_prediction":
        rows = run_group_prediction(
            load_jsonl(_resolve_path(_required(experiment, "score_windows"), config_dir))
        )
    elif task == "group_temporal_autoencoder":
        score_windows = load_jsonl(_resolve_path(_required(experiment, "score_windows"), config_dir))
        train_windows = load_jsonl(_resolve_path(_required(experiment, "train_windows"), config_dir))
        rows = run_temporal_graph_autoencoder(
            train_windows,
            score_windows,
            n_components=int(experiment.get("n_components", 3)),
            seed=int(experiment.get("seed", 42)),
        )
    elif task == "fusiontrack_individual":
        rows = run_individual_fusiontrack_baseline(
            load_jsonl(_resolve_path(_required(experiment, "train_jsonl"), config_dir)),
            load_jsonl(_resolve_path(_required(experiment, "score_jsonl"), config_dir)),
            n_neighbors=int(experiment.get("n_neighbors", 1)),
        )
    elif task == "fusiontrack_individual_ensemble":
        rows = run_individual_fusiontrack_ensemble(
            load_jsonl(_resolve_path(_required(experiment, "train_jsonl"), config_dir)),
            load_jsonl(_resolve_path(_required(experiment, "score_jsonl"), config_dir)),
            n_neighbors=int(experiment.get("n_neighbors", 1)),
            seed=int(experiment.get("seed", 42)),
            contamination=float(experiment.get("contamination", 0.05)),
            nearest_weight=float(experiment.get("nearest_weight", 0.4)),
            lof_weight=float(experiment.get("lof_weight", 0.35)),
            iforest_weight=float(experiment.get("iforest_weight", 0.25)),
            calibration_columns=_string_sequence(experiment.get("calibration_columns", ())),
            calibration_bins=int(experiment.get("calibration_bins", 4)),
            calibration_global_weight=float(
                experiment.get("calibration_global_weight", 0.7)
            ),
        )
    elif task == "fusiontrack_individual_context":
        rows = run_context_aware_fusiontrack_baseline(
            load_jsonl(_resolve_path(_required(experiment, "train_jsonl"), config_dir)),
            load_jsonl(_resolve_path(_required(experiment, "score_jsonl"), config_dir)),
            load_jsonl(_resolve_path(_required(experiment, "train_windows"), config_dir)),
            load_jsonl(_resolve_path(_required(experiment, "score_windows"), config_dir)),
            n_neighbors=int(experiment.get("n_neighbors", 1)),
        )
    elif task == "fusiontrack_group":
        rows = score_group_windows(
            load_jsonl(_resolve_path(_required(experiment, "score_windows"), config_dir)),
            k_neighbors=int(experiment.get("k_neighbors", 3)),
            rho_p=float(experiment.get("rho_p", float("inf"))),
            rho_v=float(experiment.get("rho_v", float("inf"))),
            eta=float(experiment.get("eta", 0.5)),
        )
    elif task == "fusiontrack_group_temporal_knn":
        score_windows = load_jsonl(_resolve_path(_required(experiment, "score_windows"), config_dir))
        train_windows = load_jsonl(_resolve_path(_required(experiment, "train_windows"), config_dir))
        rows = run_group_temporal_knn(
            train_windows,
            score_windows,
            n_neighbors=int(experiment.get("n_neighbors", 3)),
        )
    elif task == "fusiontrack_group_hybrid":
        score_windows = load_jsonl(_resolve_path(_required(experiment, "score_windows"), config_dir))
        train_windows = load_jsonl(_resolve_path(_required(experiment, "train_windows"), config_dir))
        rows = run_group_hybrid_fusiontrack(
            train_windows,
            score_windows,
            n_neighbors=int(experiment.get("n_neighbors", 3)),
            k_neighbors=int(experiment.get("k_neighbors", 3)),
            rho_p=float(experiment.get("rho_p", 80.0)),
            rho_v=float(experiment.get("rho_v", 20.0)),
            eta=float(experiment.get("eta", 0.5)),
            prediction_weight=float(experiment.get("prediction_weight", 0.6)),
            graph_weight=float(experiment.get("graph_weight", 0.2)),
            temporal_weight=float(experiment.get("temporal_weight", 0.2)),
            invert_graph_rank=bool(experiment.get("invert_graph_rank", True)),
            invert_temporal_rank=bool(experiment.get("invert_temporal_rank", True)),
            use_residual_gate=bool(experiment.get("use_residual_gate", False)),
            residual_gate_power=float(experiment.get("residual_gate_power", 1.0)),
            residual_gate_floor=float(experiment.get("residual_gate_floor", 0.0)),
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported task '{task}'")

    write_jsonl(score_path, rows)


def _resolve_path(value: Any, config_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else config_dir / path


def _required(mapping: dict[str, Any], field: str) -> Any:
    value = mapping.get(field)
    if value in (None, ""):
        raise ValueError(f"Missing required config field '{field}'")
    return value


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, Sequence):
        return tuple(str(part) for part in value if str(part))
    return (str(value),)


def _safe_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Experiment name cannot be empty")
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text)


if __name__ == "__main__":
    raise SystemExit(main())
