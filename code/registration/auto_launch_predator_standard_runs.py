"""Auto-launch standard PREDATOR/GeoTransformer 3DMatch-family benchmarks.

The script waits for the official PREDATOR data package to finish extracting,
discovers the standard 3DMatch/3DLoMatch metadata files, resolves the matching
point-cloud root, and launches project-schema benchmark runs with relative paths.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
import urllib.request
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable


GEOTRANSFORMER_METADATA_URLS = {
    "3dmatch": "https://raw.githubusercontent.com/qinzheng93/GeoTransformer/main/data/3DMatch/metadata/3DMatch.pkl",
    "3dlomatch": "https://raw.githubusercontent.com/qinzheng93/GeoTransformer/main/data/3DMatch/metadata/3DLoMatch.pkl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch standard 3DMatch-family benchmark runs.")
    parser.add_argument("--predator_root", default="datasets/predator_official")
    parser.add_argument("--done_file", default="runs/data_downloads/predator_data.done")
    parser.add_argument("--output_root", default="runs/predator_standard_20260527")
    parser.add_argument("--methods", default="mac,sc2_pcr")
    parser.add_argument("--datasets", default="3dmatch,3dlomatch")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--groups_per_batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--max_wait_minutes", type=int, default=0)
    parser.add_argument("--python", default="python")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _is_policy_absolute_path(path_value: str) -> bool:
    return (
        Path(path_value).is_absolute()
        or PurePosixPath(path_value).is_absolute()
        or PureWindowsPath(path_value).is_absolute()
    )


def validate_relative_paths(args: argparse.Namespace) -> None:
    for key in ("predator_root", "done_file", "output_root"):
        value = getattr(args, key)
        if _is_policy_absolute_path(str(value)):
            raise ValueError(f"{key} must be repository-relative.")


def wait_for_data(done_file: Path, predator_root: Path, poll_seconds: int, max_wait_minutes: int) -> None:
    start = time.time()
    while True:
        metadata_ready = bool(list(predator_root.rglob("3DMatch.pkl"))) and bool(
            list(predator_root.rglob("3DLoMatch.pkl"))
        )
        download_in_progress = (predator_root / "data.zip.aria2").exists()
        archive_still_present = (predator_root / "data.zip").exists()
        if done_file.exists() or (metadata_ready and not download_in_progress and not archive_still_present):
            return
        if max_wait_minutes > 0 and time.time() - start > max_wait_minutes * 60:
            raise TimeoutError("Timed out waiting for PREDATOR data package.")
        print(
            json.dumps(
                {
                    "status": "waiting_for_predator_data",
                    "done_file": str(done_file),
                    "predator_root": str(predator_root),
                }
            ),
            flush=True,
        )
        time.sleep(max(5, int(poll_seconds)))


def find_metadata(predator_root: Path, dataset_name: str) -> Path:
    target = "3DLoMatch.pkl" if dataset_name == "3dlomatch" else "3DMatch.pkl"
    matches = sorted(predator_root.rglob(target), key=lambda item: (len(item.parts), str(item)))
    if matches:
        return matches[0]

    url = GEOTRANSFORMER_METADATA_URLS[dataset_name]
    indoor_root = predator_root / "data" / "indoor"
    metadata_dir = indoor_root / "metadata" if indoor_root.is_dir() else predator_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    destination = metadata_dir / target
    print(json.dumps({"status": "downloading_standard_metadata", "dataset": dataset_name, "target": str(destination)}))
    urllib.request.urlretrieve(url, destination)
    return destination


def load_first_relative_cloud(metadata_path: Path) -> str:
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"Metadata file is empty or unsupported: {metadata_path}")
    first = metadata[0]
    if not isinstance(first, dict) or "pcd0" not in first:
        raise ValueError(f"Metadata file does not contain pcd0 entries: {metadata_path}")
    return str(first["pcd0"])


def candidate_roots(predator_root: Path, metadata_path: Path) -> Iterable[Path]:
    seen = set()
    for item in [metadata_path.parent, *metadata_path.parents, predator_root, *predator_root.iterdir()]:
        if item.is_dir() and item not in seen:
            seen.add(item)
            yield item


def resolve_dataset_root(predator_root: Path, metadata_path: Path) -> Path:
    rel_cloud = load_first_relative_cloud(metadata_path)
    rel = Path(rel_cloud)
    if _is_policy_absolute_path(rel_cloud):
        raise ValueError(f"Metadata contains absolute point-cloud path: {rel_cloud}")
    for root in candidate_roots(predator_root, metadata_path):
        if (root / rel).is_file():
            return root

    basename_matches = list(predator_root.rglob(rel.name))
    for match in basename_matches[:200]:
        try:
            suffix_len = len(rel.parts)
            if match.parts[-suffix_len:] == rel.parts:
                return Path(*match.parts[:-suffix_len])
        except ValueError:
            continue
    raise FileNotFoundError(f"Could not resolve point-cloud root for {rel_cloud}")


def run_dataset(args: argparse.Namespace, dataset_name: str, metadata_path: Path, dataset_root: Path) -> dict:
    split_root = metadata_path.parent
    output_dir = Path(args.output_root) / f"{dataset_name}_{args.methods.replace(',', '_')}"
    command = [
        args.python,
        "-u",
        "code/registration/run_registration_benchmark.py",
        "--dataset_path",
        str(dataset_root),
        "--dataset_name",
        dataset_name,
        "--split_root",
        str(split_root),
        "--pair_list",
        str(metadata_path),
        "--output_dir",
        str(output_dir),
        "--methods",
        args.methods,
        "--num_points",
        str(args.num_points),
        "--groups_per_batch",
        str(args.groups_per_batch),
        "--num_workers",
        str(args.num_workers),
        "--no_estimate_normals",
    ]
    payload = {
        "dataset": dataset_name,
        "metadata_path": str(metadata_path),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "command": command,
    }
    print(json.dumps({"launch": payload}, indent=2), flush=True)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
    return payload


def main() -> None:
    args = parse_args()
    validate_relative_paths(args)
    predator_root = Path(args.predator_root)
    done_file = Path(args.done_file)
    wait_for_data(done_file, predator_root, args.poll_seconds, args.max_wait_minutes)

    launched = []
    for dataset_name in [item.strip().lower() for item in args.datasets.split(",") if item.strip()]:
        if dataset_name not in {"3dmatch", "3dlomatch"}:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        metadata_path = find_metadata(predator_root, dataset_name)
        dataset_root = resolve_dataset_root(predator_root, metadata_path)
        launched.append(run_dataset(args, dataset_name, metadata_path, dataset_root))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "auto_launch_manifest.json").write_text(
        json.dumps({"launched": launched}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
