from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fusiontrack.visualization import _apply_image_coordinate_axes, build_visual_report


def test_background_axis_keeps_image_top_origin() -> None:
    _, ax = plt.subplots()
    try:
        ax.set_xlim(0, 640)
        ax.set_ylim(0, 512)

        _apply_image_coordinate_axes(ax, background_size=(640, 512))

        assert ax.get_xlim() == (0.0, 640.0)
        assert ax.get_ylim() == (512.0, 0.0)
    finally:
        plt.close("all")


def test_axis_without_background_uses_image_coordinate_direction() -> None:
    _, ax = plt.subplots()
    try:
        ax.plot([10, 20], [30, 40])

        _apply_image_coordinate_axes(ax, background_size=None)

        y0, y1 = ax.get_ylim()
        assert y0 > y1
    finally:
        plt.close("all")


def test_visual_report_includes_top_target_sequence(tmp_path: Path) -> None:
    fused_jsonl = tmp_path / "fused.jsonl"
    scores_csv = tmp_path / "scores.csv"
    report_dir = tmp_path / "report"
    trajectories = []
    score_rows = []
    for index, (sequence, track_id, score) in enumerate(
        [
            ("S1", "1", 35.0),
            ("S1", "2", 35.0),
            ("S1", "3", 35.0),
            ("S2", "9", 90.0),
        ]
    ):
        sample_id = f"{sequence}:{track_id}"
        trajectories.append(
            {
                "sample_id": sample_id,
                "sequence": sequence,
                "track_id": track_id,
                "category_id": 1,
                "category_name": "ship",
                "points": [
                    {"frame_id": 0, "fused": {"center_xy": [index, index], "confidence": 0.8}},
                    {"frame_id": 1, "fused": {"center_xy": [index + 10, index + 10], "confidence": 0.8}},
                ],
            }
        )
        score_rows.append(
            {
                "sample_id": sample_id,
                "sequence": sequence,
                "track_id": track_id,
                "category_id": 1,
                "category_name": "ship",
                "score": score,
                "used_sources": "individual",
            }
        )
    fused_jsonl.write_text("\n".join(json.dumps(item) for item in trajectories) + "\n", encoding="utf-8")
    with scores_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "sequence", "track_id", "category_id", "category_name", "score", "used_sources"],
        )
        writer.writeheader()
        writer.writerows(score_rows)

    summary = build_visual_report(
        fused_jsonl=fused_jsonl,
        final_scores_csv=scores_csv,
        data_root=tmp_path / "data",
        output_dir=report_dir,
        top_sequences=1,
    )

    visualized_sequences = {asset["sequence"] for asset in summary["sequence_assets"]}
    assert visualized_sequences == {"S1", "S2"}
    report_html = (report_dir / "index.html").read_text(encoding="utf-8")
    assert 'class="tab-button active" data-sequence="S1"' in report_html
    assert 'data-sequence="S2"' in report_html


def test_visual_report_writes_index_and_assets(tmp_path: Path) -> None:
    fused_jsonl = tmp_path / "fused.jsonl"
    scores_csv = tmp_path / "scores.csv"
    report_dir = tmp_path / "report"
    frame_dir = tmp_path / "data" / "test2017" / "S1" / "00"
    frame_dir.mkdir(parents=True)
    plt.imsave(frame_dir / "000000.jpg", np.zeros((12, 16, 3), dtype=np.uint8))
    plt.imsave(frame_dir / "000001.jpg", np.ones((12, 16, 3), dtype=np.uint8) * 180)
    fused_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "S1:1",
                "sequence": "S1",
                "track_id": "1",
                "category_id": 1,
                "category_name": "ship",
                "points": [
                    {
                        "frame_id": 0,
                        "rgb": {"center_xy": [0, 0], "file": "S1/00/000000.jpg"},
                        "thermal": {"center_xy": [1, 1]},
                        "modal": {"offset_distance": 1.4},
                        "fused": {"center_xy": [0, 0], "confidence": 0.9, "component_scores": {"modal_offset_distance": 1.0}},
                    },
                    {
                        "frame_id": 1,
                        "rgb": {"center_xy": [10, 10], "file": "S1/00/000001.jpg"},
                        "thermal": {"center_xy": [13, 11]},
                        "modal": {"offset_distance": 3.2},
                        "fused": {"center_xy": [10, 10], "confidence": 0.8, "component_scores": {"modal_offset_distance": 2.0}},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with scores_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "sequence", "track_id", "category_id", "category_name", "score", "used_sources"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "S1:1",
                "sequence": "S1",
                "track_id": "1",
                "category_id": 1,
                "category_name": "ship",
                "score": 0.75,
                "used_sources": "individual",
            }
        )

    summary = build_visual_report(
        fused_jsonl=fused_jsonl,
        final_scores_csv=scores_csv,
        data_root=tmp_path / "data",
        output_dir=report_dir,
        top_sequences=1,
    )

    assert summary["report_html"] == str(report_dir / "index.html")
    assert (report_dir / "index.html").exists()
    assert (report_dir / "assets" / "trajectory_S1.png").exists()
    assert (report_dir / "assets" / "heatmap_S1.png").exists()
    assert (report_dir / "assets" / "timeline_S1.png").exists()
    assert (report_dir / "assets" / "modal_S1.png").exists()
    assert (report_dir / "assets" / "background_S1.jpg").exists()
    assert (report_dir / "assets" / "playback_S1.json").exists()
    playback = json.loads((report_dir / "assets" / "playback_S1.json").read_text(encoding="utf-8"))
    assert playback["sequence"] == "S1"
    assert playback["background"] == "assets/background_S1.jpg"
    assert playback["background_frames"] == [
        {"frame": 0, "src": "assets/background_S1_000000.jpg"},
        {"frame": 1, "src": "assets/background_S1_000001.jpg"},
    ]
    assert playback["frame_range"] == [0, 1]
    assert playback["tracks"][0]["score"] == 0.75
    assert playback["tracks"][0]["duration"] == 2
    assert playback["tracks"][0]["avg_confidence"] == 0.85
    assert playback["tracks"][0]["max_modal_offset"] == 3.2
    assert playback["tracks"][0]["confidence_drop"] == 0.1
    assert [item["key"] for item in playback["tracks"][0]["reason_breakdown"]] == [
        "score",
        "source",
        "motion",
        "modal",
        "confidence",
    ]
    assert playback["tracks"][0]["points"][1]["frame"] == 1
    report_html = (report_dir / "index.html").read_text(encoding="utf-8")
    assert "FusionTrack" in report_html
    assert "Heatmap" in report_html
    assert "Modal consistency" in report_html
    assert 'id="targetSearch"' in report_html
    assert 'id="minScore"' in report_html
    assert 'id="sequenceTabs"' in report_html
    assert 'id="playbackCanvas"' in report_html
    assert 'id="playPause"' in report_html
    assert 'id="frameScrubber"' in report_html
    assert 'id="speedSelect"' in report_html
    assert 'id="targetDetail"' in report_html
    assert 'id="demoMode"' in report_html
    assert 'id="autoTour"' in report_html
    assert "renderTargetDetail" in report_html
    assert "renderReasonBreakdown" in report_html
    assert "backgroundForFrame" in report_html
    assert "ensureBackgroundForFrame" in report_html
    assert "handleTimelineClick" in report_html
    assert "evidence-tab" in report_html
    assert "data-evidence-panel" in report_html
    assert "showSequence(initialTarget.dataset.sequence" in report_html
    assert "demo-mode" in report_html
    assert 'id="lightbox"' in report_html
    assert 'data-sequence="S1"' in report_html
    assert "Top anomaly table" not in report_html
    assert "File index" not in report_html
