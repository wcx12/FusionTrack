from __future__ import annotations

from mtf_ba.group_interface import GroupWindow

from fusiontrack.group_baseline import LightweightGroupAnomalyDetector


def make_object(track_id: str, centers: list[tuple[float, float]]) -> dict:
    return {
        "sample_id": f"S1:{track_id}",
        "sequence": "S1",
        "track_id": track_id,
        "category_id": 1,
        "category_name": "ship",
        "states": [
            {
                "frame_id": idx,
                "rgb": {"center_xy": [x, y]},
                "thermal": {"center_xy": [x + 1.0, y + 1.0]},
                "modal": {"offset_distance": 1.414},
            }
            for idx, (x, y) in enumerate(centers)
        ],
    }


def test_group_baseline_scores_leaving_target_higher() -> None:
    window = GroupWindow(
        window_id="S1:0-4",
        sequence="S1",
        frame_start=0,
        frame_end=4,
        frames=[0, 1, 2, 3, 4],
        objects=[
            make_object("1", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]),
            make_object("2", [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
            make_object("3", [(0, 2), (1, 2), (10, 8), (20, 15), (30, 20)]),
        ],
    )

    records = list(LightweightGroupAnomalyDetector().score_windows([window]))
    scores = {record.track_id: record.score for record in records}

    assert scores["3"] > scores["1"]
    assert scores["3"] > scores["2"]
