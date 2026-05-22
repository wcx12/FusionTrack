from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol.inject_group import inject_group_anomalies


def _object_centers(obj: dict) -> list[tuple[float, float]]:
    centers = []
    for state in obj["states"]:
        for modality in ("fused", "rgb", "thermal"):
            center = state.get(modality, {}).get("center_xy")
            if center is not None:
                centers.append((center[0], center[1]))
    return centers


def _windows() -> list[dict]:
    return [
        {
            "window_id": "seq_a:1-3",
            "sequence": "seq_a",
            "frame_start": 1,
            "frame_end": 3,
            "objects": [
                {
                    "sample_id": "seq_a:track_1",
                    "track_id": "track_1",
                    "states": [
                        {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
                        {"frame_id": 2, "fused": {"center_xy": [1.0, 1.0]}},
                        {"frame_id": 3, "fused": {"center_xy": [2.0, 2.0]}},
                    ],
                },
                {
                    "sample_id": "seq_a:track_2",
                    "track_id": "track_2",
                    "states": [
                        {"frame_id": 1, "fused": {"center_xy": [5.0, 5.0]}},
                        {"frame_id": 2, "fused": {"center_xy": [6.0, 6.0]}},
                        {"frame_id": 3, "fused": {"center_xy": [7.0, 7.0]}},
                    ],
                },
            ],
        },
        {
            "window_id": "seq_b:1-2",
            "sequence": "seq_b",
            "frame_start": 1,
            "frame_end": 2,
            "objects": [
                {
                    "sample_id": "seq_b:track_3",
                    "track_id": "track_3",
                    "states": [
                        {"frame_id": 1, "fused": {"center_xy": [10.0, 10.0]}},
                        {"frame_id": 2, "fused": {"center_xy": [11.0, 11.0]}},
                    ],
                }
            ],
        },
    ]


def test_group_injection_does_not_mutate_input_and_is_deterministic() -> None:
    windows = _windows()
    original = deepcopy(windows)

    injected_first, labels_first = inject_group_anomalies(
        windows,
        anomaly_fraction=0.5,
        seed=5,
        anomaly_types=["dispersion_change"],
    )
    injected_second, labels_second = inject_group_anomalies(
        windows,
        anomaly_fraction=0.5,
        seed=5,
        anomaly_types=["dispersion_change"],
    )

    assert windows == original
    assert injected_first == injected_second
    assert [label.to_dict() for label in labels_first] == [
        label.to_dict() for label in labels_second
    ]
    assert len(labels_first) == 1

    labeled = {(label.sequence, label.track_id) for label in labels_first}
    assert any(
        (window["sequence"], obj["track_id"]) in labeled
        and _object_centers(obj) != _object_centers(injected_obj)
        for window, injected_window in zip(windows, injected_first)
        for obj, injected_obj in zip(window["objects"], injected_window["objects"])
    )


def test_population_change_changes_object_count_and_emits_one_label() -> None:
    windows = [_windows()[0]]
    original = deepcopy(windows)

    injected, labels = inject_group_anomalies(
        windows,
        anomaly_fraction=1.0,
        seed=23,
        anomaly_types=["population_change"],
    )

    assert windows == original
    assert len(labels) == 1
    assert len(injected[0]["objects"]) != len(windows[0]["objects"])


def test_group_labels_reference_objects_remaining_in_injected_windows() -> None:
    windows = _windows()

    injected, labels = inject_group_anomalies(
        windows,
        anomaly_fraction=1.0,
        seed=23,
        anomaly_types=["population_change"],
    )

    assert labels
    windows_by_id = {window["window_id"]: window for window in injected}
    for label in labels:
        window = windows_by_id[label.metadata["window_id"]]
        sample_ids = {obj.get("sample_id") for obj in window["objects"]}
        assert label.sample_id in sample_ids


def test_group_anomaly_skips_objects_without_centers_without_label() -> None:
    windows = [
        {
            "window_id": "seq_no_center:1-2",
            "sequence": "seq_no_center",
            "frame_start": 1,
            "frame_end": 2,
            "objects": [
                {
                    "sample_id": "seq_no_center:track_1",
                    "track_id": "track_1",
                    "states": [{"frame_id": 1}, {"frame_id": 2}],
                }
            ],
        }
    ]
    original = deepcopy(windows)

    injected, labels = inject_group_anomalies(
        windows,
        anomaly_fraction=1.0,
        seed=29,
        anomaly_types=["leave_group"],
    )

    assert windows == original
    assert injected == original
    assert labels == []
