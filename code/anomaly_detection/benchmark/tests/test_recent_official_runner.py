from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not hasattr(torch, "no_grad"):
    pytest.skip("A full PyTorch installation is required", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runners.run_recent_official_fusiontrack import (
    SequenceSample,
    _aggregate_time_scores,
    _cap_features_len,
    _patch_config,
    _resample,
    _score_row,
)


def test_recent_runner_uses_expected_cutaddpaste_feature_lengths() -> None:
    assert _cap_features_len(16) == 4
    assert _cap_features_len(32) == 6
    assert _cap_features_len(64) == 10


def test_recent_runner_patch_config_stays_inside_window() -> None:
    patch_size, patch_stride = _patch_config(16)

    assert patch_size == 16
    assert patch_stride == 8


def test_recent_runner_resamples_short_sequences_to_fixed_window() -> None:
    values = np.asarray([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)

    resampled = _resample(values, 3)

    assert resampled.shape == (3, 2)
    assert resampled[1].tolist() == [5.0, 10.0]


def test_recent_runner_aggregates_top_fraction_scores() -> None:
    scores = torch.tensor([[1.0, 2.0, 10.0, 4.0]])

    aggregated = _aggregate_time_scores(scores, 0.5)

    assert aggregated.tolist() == [7.0]


def test_recent_runner_score_row_keeps_group_window_id() -> None:
    args = SimpleNamespace(method="timemixer", task="group")
    sample = SequenceSample(
        sample_id="seq:track",
        sequence="seq",
        track_id="track",
        values=np.zeros((2, 2), dtype=np.float32),
        metadata={"window_id": "w1"},
    )

    row = _score_row(args, sample, 0.25)

    assert row["sample_id"] == "seq:track"
    assert row["window_id"] == "w1"
    assert row["source"] == "official_timemixer:group"
    assert row["score"] == 0.25
