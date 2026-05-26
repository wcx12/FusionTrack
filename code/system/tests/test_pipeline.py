from __future__ import annotations

import json
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
    assert summary["dataset_manifest"]["status"] == "missing_data_root"
    assert (paths.work_root / "dataset_manifest_all.json").exists()
    assert (paths.work_root / "final_dashboard" / "index.html").exists()


def test_build_final_results_report_links_suite_manifest(tmp_path: Path) -> None:
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
    suite_dir = tmp_path / "suite"
    matrix_dir = suite_dir / "individual_matrix"
    matrix_dir.mkdir(parents=True)
    aggregate_summary = suite_dir / "aggregate_summary.csv"
    aggregate_summary.write_text("matrix,method,auroc\nindividual,fusiontrack,0.91\n", encoding="utf-8")
    matrix_summary = matrix_dir / "summary.csv"
    matrix_summary.write_text("method,auroc\nfusiontrack,0.91\n", encoding="utf-8")
    matrix_manifest = matrix_dir / "manifest.json"
    matrix_manifest.write_text('{"runs":[{"name":"fusiontrack"}]}', encoding="utf-8")
    suite_manifest = suite_dir / "suite_manifest.json"
    suite_manifest.write_text(
        (
            "{"
            '"suite_name":"paper_suite",'
            '"generated_at_utc":"2026-05-26T00:00:00Z",'
            f'"aggregate_summary_csv":{str(aggregate_summary)!r},'
            '"matrices":[{'
            '"name":"individual",'
            '"split":"val",'
            '"num_runs":3,'
            f'"summary_csv":{str(matrix_summary)!r},'
            f'"manifest_json":{str(matrix_manifest)!r}'
            "}]"
            "}"
        ).replace("'", '"'),
        encoding="utf-8",
    )

    summary = build_final_results_report(
        paths=paths,
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
        fused_jsonl=fused_jsonl,
        suite_manifest=suite_manifest,
        top_sequences=1,
        top_k=2,
        case_limit=3,
        sync_remote_report=False,
    )

    assert summary["suite_manifest_path"] == str(suite_manifest)
    assert summary["suite_manifest"]["suite_name"] == "paper_suite"
    assert summary["suite_manifest"]["matrices"][0]["num_runs"] == 3
    dashboard_data = json.loads(
        (paths.work_root / "final_dashboard" / "assets" / "final_dashboard_data.json").read_text(encoding="utf-8")
    )
    suite = dashboard_data["provenance"]["suite"]
    assert suite["name"] == "paper_suite"
    assert suite["matrix_count"] == 1
    assert suite["run_count"] == 3
    assert suite["aggregate_summary_csv"] == "aggregate_summary.csv"
    assert suite["matrices"][0]["summary_csv"] == "summary.csv"
    assert str(tmp_path) not in json.dumps(suite)
