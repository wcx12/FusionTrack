from __future__ import annotations

from pathlib import Path
import re

from fusiontrack.config import FusionTrackPaths
from fusiontrack.pipeline import build_experiment_report, build_extraction_command, build_final_results_report
from test_final_results import _build_small_final_result_tree, _write_jsonl


def test_extraction_command_uses_relative_server_paths() -> None:
    paths = FusionTrackPaths.defaults()
    command = build_extraction_command(paths, "test")
    command_text = " ".join(command)

    assert "data/VT-Tiny-MOT" in command_text.replace("\\", "/")
    assert "runs/fusiontrack_v1/trajectories" in command_text.replace("\\", "/")
    assert re.search(r"[A-Za-z]:[\\/]", command_text) is None


def test_build_experiment_report_from_result_manifest(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"
    paths = FusionTrackPaths.defaults(data_root=tmp_path / "data", work_root=work_root)
    fused_jsonl = tmp_path / "fused.jsonl"
    fused_jsonl.write_text(
        "\n".join(
            [
                '{"sample_id":"S1:7","sequence":"S1","track_id":"7","category_id":1,'
                '"category_name":"ship","points":[{"frame_id":1,"fused":{"center_xy":[1,2],'
                '"confidence":0.9}},{"frame_id":2,"fused":{"center_xy":[3,4],"confidence":0.8}}]}'
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result_dir = tmp_path / "result"
    (result_dir / "scores").mkdir(parents=True)
    (result_dir / "metrics").mkdir(parents=True)
    (result_dir / "scores" / "method_a.jsonl").write_text(
        '{"sample_id":"S1:7","sequence":"S1","track_id":"7","score":0.82}\n',
        encoding="utf-8",
    )
    (result_dir / "labels.jsonl").write_text(
        '{"sample_id":"S1:7","sequence":"S1","track_id":"7","frame_start":1,'
        '"frame_end":2,"label":1,"anomaly_type":"speed_spike","injection_seed":42}\n',
        encoding="utf-8",
    )
    (result_dir / "metrics" / "method_a.json").write_text(
        '{"method":"method_a","auroc":0.91,"seed":42}',
        encoding="utf-8",
    )
    manifest_path = result_dir / "manifest.json"
    manifest_path.write_text(
        '{"split":"test","seed":42,"label_file":"labels.jsonl","runs":[{"name":"method_a",'
        '"task":"individual","score_file":"scores/method_a.jsonl",'
        '"metrics_file":"metrics/method_a.json"}]}',
        encoding="utf-8",
    )

    summary = build_experiment_report(
        paths=paths,
        result_manifest=manifest_path,
        split="test",
        fused_jsonl=fused_jsonl,
        top_sequences=1,
    )

    assert summary["mode"] == "experiment_report"
    assert summary["experiment"]["method_name"] == "method_a"
    assert "labels_by_sample" not in summary["experiment"]
    assert (paths.final_dir / "experiment_scores_method_a.csv").exists()
    report_html = paths.report_dir / "index.html"
    assert report_html.exists()
    assert "method_a" in report_html.read_text(encoding="utf-8")


def test_build_final_results_report_from_summary_files(tmp_path: Path) -> None:
    paths = FusionTrackPaths.defaults(data_root=tmp_path / "data", work_root=tmp_path / "runs")
    final_root, score_root, individual_labels, group_labels = _build_small_final_result_tree(tmp_path)
    fused_jsonl = tmp_path / "fused.jsonl"
    _write_jsonl(
        fused_jsonl,
        [
            {
                "sample_id": "S1:1",
                "sequence": "S1",
                "track_id": "1",
                "category_name": "plane",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [1, 2], "confidence": 0.9}},
                    {"frame_id": 2, "fused": {"center_xy": [3, 4], "confidence": 0.8}},
                ],
            }
        ],
    )

    summary = build_final_results_report(
        paths=paths,
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
        fused_jsonl=fused_jsonl,
        top_sequences=1,
        top_k=2,
        case_limit=3,
        sync_remote_report=False,
    )

    assert summary["mode"] == "final_results_dashboard"
    assert summary["dashboard"]["num_methods"] == 3
    assert (paths.work_root / "final_dashboard" / "index.html").exists()
