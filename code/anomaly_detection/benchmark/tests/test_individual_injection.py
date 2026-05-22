from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol.inject_individual import inject_individual_anomalies


def _trajectory_centers(trajectory: dict) -> list[tuple[float, float]]:
    centers = []
    for point in trajectory["points"]:
        for modality in ("fused", "rgb", "thermal"):
            center = point.get(modality, {}).get("center_xy")
            if center is not None:
                centers.append((center[0], center[1]))
    return centers


def _trajectories() -> list[dict]:
    return [
        {
            "sample_id": "seq_a:track_1",
            "sequence": "seq_a",
            "track_id": "track_1",
            "points": [
                {
                    "frame_id": 1,
                    "fused": {"center_xy": [0.0, 0.0]},
                    "rgb": {"center_xy": [0.0, 0.0]},
                    "thermal": {"center_xy": [0.0, 0.0]},
                },
                {
                    "frame_id": 2,
                    "fused": {"center_xy": [1.0, 1.0]},
                    "rgb": {"center_xy": [1.0, 1.0]},
                    "thermal": {"center_xy": [1.0, 1.0]},
                },
                {
                    "frame_id": 3,
                    "fused": {"center_xy": [2.0, 2.0]},
                    "rgb": {"center_xy": [2.0, 2.0]},
                    "thermal": {"center_xy": [2.0, 2.0]},
                },
            ],
        },
        {
            "sample_id": "seq_a:track_2",
            "sequence": "seq_a",
            "track_id": "track_2",
            "points": [
                {"frame_id": 1, "fused": {"center_xy": [10.0, 10.0]}},
                {"frame_id": 2, "fused": {"center_xy": [11.0, 11.0]}},
            ],
        },
    ]


def test_individual_injection_does_not_mutate_input_and_is_deterministic() -> None:
    trajectories = _trajectories()
    original = deepcopy(trajectories)

    injected_first, labels_first = inject_individual_anomalies(
        trajectories,
        anomaly_fraction=0.5,
        seed=7,
        anomaly_types=["route_shift"],
    )
    injected_second, labels_second = inject_individual_anomalies(
        trajectories,
        anomaly_fraction=0.5,
        seed=7,
        anomaly_types=["route_shift"],
    )

    assert trajectories == original
    assert injected_first == injected_second
    assert [label.to_dict() for label in labels_first] == [
        label.to_dict() for label in labels_second
    ]
    assert len(labels_first) == 1

    changed_ids = {label.sample_id for label in labels_first}
    assert any(
        before["sample_id"] in changed_ids
        and _trajectory_centers(before) != _trajectory_centers(after)
        for before, after in zip(trajectories, injected_first)
    )


def test_modal_offset_skips_fused_only_trajectory_without_label() -> None:
    trajectories = [
        {
            "sample_id": "seq_a:track_fused_only",
            "sequence": "seq_a",
            "track_id": "track_fused_only",
            "points": [
                {"frame_id": 1, "fused": {"center_xy": [10.0, 10.0]}},
                {"frame_id": 2, "fused": {"center_xy": [11.0, 11.0]}},
            ],
        }
    ]
    original = deepcopy(trajectories)

    injected, labels = inject_individual_anomalies(
        trajectories,
        anomaly_fraction=1.0,
        seed=17,
        anomaly_types=["modal_offset"],
    )

    assert trajectories == original
    assert injected == original
    assert labels == []


def test_stop_or_slowdown_skips_single_point_trajectory_without_label() -> None:
    trajectories = [
        {
            "sample_id": "seq_a:track_single",
            "sequence": "seq_a",
            "track_id": "track_single",
            "points": [
                {
                    "frame_id": 1,
                    "fused": {"center_xy": [1.0, 1.0]},
                    "rgb": {"center_xy": [1.0, 1.0]},
                }
            ],
        }
    ]
    original = deepcopy(trajectories)

    injected, labels = inject_individual_anomalies(
        trajectories,
        anomaly_fraction=1.0,
        seed=19,
        anomaly_types=["stop_or_slowdown"],
    )

    assert trajectories == original
    assert injected == original
    assert labels == []
