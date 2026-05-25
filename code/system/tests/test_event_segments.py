from __future__ import annotations

from fusiontrack.event_segments import event_segments_from_frame_scores, normalize_frame_event_scores


def test_normalize_frame_event_scores_filters_and_sorts_rows() -> None:
    rows = [
        {"frame_id": "4", "score": "0.5", "dominant_reason": "leave", "component_scores": {"graph_leave": "0.5"}},
        {"frame": "bad", "score": 1.0},
        {"frame": 2, "score": None},
        {"frame": 3, "score": "nan"},
    ]

    normalized = normalize_frame_event_scores(rows, source="group")

    assert normalized == [
        {"frame": 2, "score": 0.0, "source": "group"},
        {
            "frame": 4,
            "score": 0.5,
            "dominant_reason": "leave",
            "component_scores": {"graph_leave": 0.5},
            "source": "group",
        },
    ]


def test_event_segments_from_frame_scores_merges_small_gaps_and_keeps_peak_reason() -> None:
    rows = [
        {"frame": 1, "score": 0.0, "dominant_reason": "normal"},
        {"frame": 2, "score": 0.45, "dominant_reason": "speed", "component_scores": {"speed": 0.45}},
        {"frame": 4, "score": 0.72, "dominant_reason": "leave", "component_scores": {"speed": 0.2, "leave": 0.72}},
        {"frame": 8, "score": 0.4, "dominant_reason": "dispersion", "component_scores": {"dispersion": 0.4}},
    ]

    segments = event_segments_from_frame_scores(rows, threshold=0.3, max_gap=2, source="group")

    assert segments == [
        {
            "frame_start": 2,
            "frame_end": 4,
            "score": 0.72,
            "dominant_reason": "leave",
            "num_frames": 2,
            "component_scores": {"speed": 0.45, "leave": 0.72},
            "source": "group",
        },
        {
            "frame_start": 8,
            "frame_end": 8,
            "score": 0.4,
            "dominant_reason": "dispersion",
            "num_frames": 1,
            "component_scores": {"dispersion": 0.4},
            "source": "group",
        },
    ]
