"""Launch MPS-GAF final eval after a training run exits.

The watcher is intentionally small and repository-relative. It does not decide
convergence itself; it waits for the configured training PID to exit, then runs
the explicit eval command against the best checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path, PurePosixPath, PureWindowsPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for MPS-GAF training and launch eval.")
    parser.add_argument("--train_pid_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", default="modelnet40")
    parser.add_argument("--noise_type", default="crop")
    parser.add_argument("--num_sources_per_ref", type=int, default=2)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--groups_per_batch", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=20)
    parser.add_argument("--num_eval_iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--best_metric", default="pose")
    parser.add_argument("--pose_trans_weight", type=float, default=50.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default="python")
    parser.add_argument("--poll_seconds", type=int, default=60)
    return parser.parse_args()


def _is_policy_absolute_path(path_value: str) -> bool:
    return (
        Path(path_value).is_absolute()
        or PurePosixPath(path_value).is_absolute()
        or PureWindowsPath(path_value).is_absolute()
    )


def validate_relative_paths(args: argparse.Namespace) -> None:
    for key in ("train_pid_file", "checkpoint", "output_dir", "dataset_path"):
        value = getattr(args, key)
        if _is_policy_absolute_path(str(value)):
            raise ValueError(f"{key} must be repository-relative.")


def read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    args = parse_args()
    validate_relative_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "auto_eval_status.jsonl"
    summary_path = output_dir / "comparison_schema_summary.json"

    while True:
        pid = read_pid(Path(args.train_pid_file))
        alive = pid is not None and process_is_alive(pid)
        log_path.open("a", encoding="utf-8").write(
            json.dumps({"status": "waiting", "train_pid": pid, "alive": alive}) + "\n"
        )
        if not alive:
            break
        time.sleep(max(5, int(args.poll_seconds)))

    if summary_path.exists():
        log_path.open("a", encoding="utf-8").write(
            json.dumps({"status": "skipped_existing_summary", "summary": str(summary_path)}) + "\n"
        )
        return

    command = [
        args.python,
        "code/registration/mps_gaf_run.py",
        "--mode",
        "eval",
        "--dataset_path",
        args.dataset_path,
        "--output_dir",
        args.output_dir,
        "--checkpoint",
        args.checkpoint,
        "--dataset_name",
        args.dataset_name,
        "--noise_type",
        args.noise_type,
        "--num_sources_per_ref",
        str(args.num_sources_per_ref),
        "--num_points",
        str(args.num_points),
        "--groups_per_batch",
        str(args.groups_per_batch),
        "--num_workers",
        str(args.num_workers),
        "--max_eval_batches",
        str(args.max_eval_batches),
        "--num_eval_iter",
        str(args.num_eval_iter),
        "--seed",
        str(args.seed),
        "--best_metric",
        args.best_metric,
        "--pose_trans_weight",
        str(args.pose_trans_weight),
        "--device",
        args.device,
    ]
    log_path.open("a", encoding="utf-8").write(
        json.dumps({"status": "launching_eval", "command": command}) + "\n"
    )
    subprocess.run(command, check=True)
    log_path.open("a", encoding="utf-8").write(json.dumps({"status": "eval_done"}) + "\n")


if __name__ == "__main__":
    main()
