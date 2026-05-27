from __future__ import annotations

from fusiontrack.explanation_schema import build_explanation_schema


def test_build_explanation_schema_prefers_peak_event_reason_and_records_policy() -> None:
    explanation = build_explanation_schema(
        {
            "score": 0.73,
            "event_score": 0.82,
            "component_scores": {
                "S_ind": 0.35,
                "S_grp": 0.91,
                "S_event": 0.82,
                "S_fused": 0.73,
                "group_graph_leave": 0.91,
                "individual_speed": 0.35,
            },
            "frame_event_scores": [
                {"frame": 5, "score": 0.42, "dominant_reason": "speed"},
                {
                    "frame": 7,
                    "score": 0.82,
                    "dominant_reason": "leave",
                    "component_scores": {"group_graph_leave": 0.82},
                },
            ],
        },
        threshold=0.5,
        max_gap=2,
        min_length=1,
    )

    assert explanation["schema_version"] == 1
    assert explanation["top_reason"] == "leave"
    assert explanation["evidence_source"] == "frame_event_scores"
    assert explanation["policy"] == {"event_threshold": 0.5, "max_gap": 2, "min_length": 1}
    assert explanation["peak_event"]["frame_start"] == 7
    assert explanation["peak_event"]["score"] == 0.82
    assert explanation["score_components"][0] == {
        "name": "group_graph_leave",
        "value": 0.91,
        "family": "group",
    }
    assert explanation["score_components"][1]["name"] == "S_event"


def test_build_explanation_schema_falls_back_to_component_reason_without_events() -> None:
    explanation = build_explanation_schema(
        {
            "score": 0.4,
            "component_scores": {"route_score": 0.64, "shape_score": 0.32},
        },
        threshold=0.5,
    )

    assert explanation["top_reason"] == "route_score"
    assert explanation["evidence_source"] == "component_scores"
    assert explanation["peak_event"] is None
    assert explanation["score_components"][0]["name"] == "route_score"
