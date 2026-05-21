from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusiontrack.group_graph import (
    build_spatial_edges,
    compute_relative_displacements,
    connected_components,
    extract_object_states,
)


def _window() -> dict:
    return {
        "window_id": "w1",
        "sample_id": "window_sample",
        "sequence": "seq_a",
        "objects": [
            {
                "sample_id": "a_sample",
                "track_id": "a",
                "category_name": "person",
                "states": [
                    {"frame_id": 1, "fused": {"center_xy": [0.0, 0.0]}},
                    {"frame_id": 2, "fused": {"center_xy": [10.0, 0.0]}},
                ],
            },
            {
                "track_id": "b",
                "category_id": 1,
                "states": [
                    {"frame_id": 1, "rgb": {"center_xy": [0.0, 2.0]}},
                    {"frame_id": 2, "rgb": {"center_xy": [10.0, 2.0]}},
                ],
            },
            {
                "track_id": "c",
                "states": [
                    {"frame_id": 1, "thermal": {"center_xy": [20.0, 0.0]}},
                    {"frame_id": 2, "thermal": {"center_xy": [35.0, 0.0]}},
                ],
            },
        ],
    }


def test_extract_object_states_prefers_fused_and_falls_back_to_modal_centers() -> None:
    states = extract_object_states(_window())

    assert [(state["frame_id"], state["track_id"]) for state in states] == [
        (1, "a"),
        (1, "b"),
        (1, "c"),
        (2, "a"),
        (2, "b"),
        (2, "c"),
    ]
    assert states[0]["center_xy"] == [0.0, 0.0]
    assert states[1]["center_xy"] == [0.0, 2.0]
    assert states[2]["center_xy"] == [20.0, 0.0]
    assert states[0]["sample_id"] == "a_sample"
    assert states[1]["sample_id"] == "seq_a:b"


def test_relative_displacements_remove_frame_median_motion() -> None:
    states = compute_relative_displacements(extract_object_states(_window()))
    frame2 = {state["track_id"]: state for state in states if state["frame_id"] == 2}

    assert frame2["a"]["velocity"] == [10.0, 0.0]
    assert frame2["b"]["velocity"] == [10.0, 0.0]
    assert frame2["c"]["velocity"] == [15.0, 0.0]
    assert frame2["a"]["rel_velocity"] == [0.0, 0.0]
    assert frame2["b"]["rel_velocity"] == [0.0, 0.0]
    assert frame2["c"]["rel_velocity"] == [5.0, 0.0]


def test_build_spatial_edges_and_connected_components_use_knn_thresholds() -> None:
    states = compute_relative_displacements(extract_object_states(_window()))

    edges_by_frame = build_spatial_edges(states, k_neighbors=1, rho_p=3.0, rho_v=1.0)

    assert edges_by_frame[1] == {("a", "b")}
    assert edges_by_frame[2] == {("a", "b")}
    components = connected_components({"a", "b", "c"}, edges_by_frame[1])
    assert {frozenset(component) for component in components} == {
        frozenset({"a", "b"}),
        frozenset({"c"}),
    }
