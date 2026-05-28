import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline():
    module_path = Path(__file__).resolve().parents[2] / "code" / "registration" / "mps_gaf_data_pipeline.py"
    spec = importlib.util.spec_from_file_location("mps_gaf_data_pipeline_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load_pipeline()


def test_kitti_pair_dataset_reads_geotransformer_metadata(tmp_path: Path) -> None:
    root = tmp_path / "kitti"
    points_dir = root / "downsampled" / "08"
    points_dir.mkdir(parents=True)
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    np.save(points_dir / "000000.npy", points)
    np.save(points_dir / "000001.npy", points + 1.0)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    metadata = [
        {
            "seq_id": 8,
            "frame0": 1,
            "frame1": 0,
            "transform": transform,
            "pcd0": "downsampled/08/000001.npy",
            "pcd1": "downsampled/08/000000.npy",
        }
    ]
    metadata_path = tmp_path / "test.pkl"
    metadata_path.write_bytes(pickle.dumps(metadata))

    dataset = pipeline.KittiPairDataset(
        str(root),
        str(metadata_path),
        num_points=6,
        base_seed=0,
        estimate_normals=False,
    )
    sample = dataset[0]

    assert sample["points_src"].shape == (6, 6)
    assert sample["points_ref"].shape == (6, 6)
    assert sample["transform_gt"].shape == (3, 4)
    assert sample["transform_gt"][:, 3] == pytest.approx([1.0, 2.0, 3.0])
    assert sample["scene_name"] == "8"


def test_eth_log_parser_reads_standard_gt_log(tmp_path: Path) -> None:
    log_path = tmp_path / "gt.log"
    log_path.write_text(
        "\n".join(
            [
                "0 1 2",
                "1 0 0 0.1",
                "0 1 0 0.2",
                "0 0 1 0.3",
                "0 0 0 1",
            ]
        ),
        encoding="utf-8",
    )

    pairs = pipeline.ETHPairDataset._read_log(str(log_path))

    assert len(pairs) == 1
    ref_id, src_id, transform = pairs[0]
    assert (ref_id, src_id) == (0, 1)
    assert transform.shape == (4, 4)
    assert transform[:3, 3] == pytest.approx([0.1, 0.2, 0.3])
