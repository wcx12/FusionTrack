import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_exporter():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "code"
        / "registration"
        / "export_eth_geotransformer_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("export_eth_geotransformer_dataset_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = _load_exporter()


def test_read_log_parses_eth_pair_transform(tmp_path: Path) -> None:
    log_path = tmp_path / "gt.log"
    log_path.write_text(
        "\n".join(
            [
                "0 1 32",
                "1 0 0 0.5",
                "0 1 0 -0.25",
                "0 0 1 1.25",
                "0 0 0 1",
            ]
        ),
        encoding="utf-8",
    )

    pairs = exporter.read_log(log_path)

    assert len(pairs) == 1
    ref_id, src_id, transform = pairs[0]
    assert (ref_id, src_id) == (0, 1)
    assert transform[:3, 3].tolist() == pytest.approx([0.5, -0.25, 1.25])


def test_sample_points_is_deterministic_for_fixed_rng() -> None:
    points = np.arange(30, dtype=np.float32).reshape(10, 3)

    first = exporter.sample_points(points, 4, np.random.RandomState(123))
    second = exporter.sample_points(points, 4, np.random.RandomState(123))

    assert first.tolist() == second.tolist()
    assert first.shape == (4, 3)
