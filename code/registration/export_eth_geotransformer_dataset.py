"""Export ETH laser pairs into the GeoTransformer 3DMatch-style data schema.

The export keeps the official ETH ``gt.log`` pairs and only changes the on-disk
format so GeoTransformer's existing 3DMatch pair dataset can read the benchmark.
Point sampling matches ``ETHPairDataset`` in ``mps_gaf_data_pipeline.py``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


SCENES = ("gazebo_summer", "gazebo_winter", "wood_autmn", "wood_summer")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth_root", required=True, help="Directory containing the four ETH scene folders.")
    parser.add_argument("--output_root", required=True, help="GeoTransformer-format output dataset root.")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark_name", default="ETH")
    parser.add_argument("--calibration_pairs", type=int, default=32)
    return parser


def read_log(path: Path) -> List[Tuple[int, int, np.ndarray]]:
    pairs: List[Tuple[int, int, np.ndarray]] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    idx = 0
    while idx + 4 < len(lines):
        header = lines[idx].split()
        if len(header) < 2:
            raise ValueError(f"Invalid ETH gt.log header in {path}: {lines[idx]}")
        ref_id = int(header[0])
        src_id = int(header[1])
        matrix = np.asarray(
            [[float(value) for value in lines[idx + row].split()] for row in range(1, 5)],
            dtype=np.float32,
        )
        pairs.append((ref_id, src_id, matrix))
        idx += 5
    return pairs


def load_points(path: Path) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected ETH point cloud with shape [N, 3] in {path}")
    return points


def sample_points(points: np.ndarray, num_points: int, rng: np.random.RandomState) -> np.ndarray:
    if num_points <= points.shape[0]:
        indices = rng.choice(points.shape[0], num_points, replace=False)
    else:
        indices = np.concatenate(
            [
                rng.choice(points.shape[0], points.shape[0], replace=False),
                rng.choice(points.shape[0], num_points - points.shape[0], replace=True),
            ],
            axis=0,
        )
    return points[indices].astype(np.float32)


def collect_eth_pairs(eth_root: Path, scenes: Iterable[str]) -> List[Dict[str, object]]:
    pairs: List[Dict[str, object]] = []
    for scene in scenes:
        scene_dir = eth_root / scene
        log_path = scene_dir / "gt.log"
        if not scene_dir.is_dir() or not log_path.is_file():
            continue
        for ref_id, src_id, transform in read_log(log_path):
            pairs.append(
                {
                    "scene": scene,
                    "ref_id": ref_id,
                    "src_id": src_id,
                    "ref_path": scene_dir / f"Hokuyo_{ref_id}.ply",
                    "src_path": scene_dir / f"Hokuyo_{src_id}.ply",
                    "transform": transform,
                }
            )
    return pairs


def export_dataset(args: argparse.Namespace) -> Dict[str, object]:
    import torch

    eth_root = Path(args.eth_root)
    output_root = Path(args.output_root)
    data_root = output_root / "data" / "eth_pairs"
    metadata_root = output_root / "metadata"
    data_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    pairs = collect_eth_pairs(eth_root, SCENES)
    if not pairs:
        raise ValueError(f"No ETH pairs found under {eth_root}")

    point_cache: Dict[Path, np.ndarray] = {}
    metadata: List[Dict[str, object]] = []
    for idx, pair in enumerate(pairs):
        ref_path = Path(pair["ref_path"])
        src_path = Path(pair["src_path"])
        if ref_path not in point_cache:
            point_cache[ref_path] = load_points(ref_path)
        if src_path not in point_cache:
            point_cache[src_path] = load_points(src_path)

        rng_src = np.random.RandomState((args.seed + idx * 1_000_003 + 31) % (2**32 - 1))
        rng_ref = np.random.RandomState((args.seed + idx * 1_000_003 + 37) % (2**32 - 1))
        rel_ref = Path("eth_pairs") / f"{idx:06d}_ref.pth"
        rel_src = Path("eth_pairs") / f"{idx:06d}_src.pth"
        torch.save(sample_points(point_cache[ref_path], args.num_points, rng_ref), data_root / rel_ref.name)
        torch.save(sample_points(point_cache[src_path], args.num_points, rng_src), data_root / rel_src.name)

        transform = np.asarray(pair["transform"], dtype=np.float32)
        metadata.append(
            {
                "scene_name": str(pair["scene"]),
                "frag_id0": int(pair["ref_id"]),
                "frag_id1": int(pair["src_id"]),
                "overlap": 1.0,
                "rotation": transform[:3, :3],
                "translation": transform[:3, 3],
                "pcd0": str(rel_ref).replace("\\", "/"),
                "pcd1": str(rel_src).replace("\\", "/"),
            }
        )

    with (metadata_root / f"{args.benchmark_name}.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    calibration_count = max(1, min(int(args.calibration_pairs), len(metadata)))
    with (metadata_root / "train.pkl").open("wb") as handle:
        pickle.dump(metadata[:calibration_count], handle)

    manifest = {
        "benchmark_name": args.benchmark_name,
        "num_pairs": len(metadata),
        "num_points": args.num_points,
        "seed": args.seed,
        "scenes": list(SCENES),
        "calibration_pairs": calibration_count,
        "format_note": "Pair-level sampled source/reference files for GeoTransformer inference.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = make_parser().parse_args()
    manifest = export_dataset(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
