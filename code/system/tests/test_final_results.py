from __future__ import annotations

import csv
import json
from pathlib import Path

from fusiontrack.final_dashboard import build_final_dashboard
from fusiontrack.final_results import load_final_results_dashboard


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_small_final_result_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    final_root = tmp_path / "final"
    score_root = tmp_path / "scores"
    individual_labels = tmp_path / "labels" / "individual_labels_val.jsonl"
    group_labels = tmp_path / "labels" / "group_labels_val.jsonl"

    _write_jsonl(
        individual_labels,
        [
            {
                "sample_id": "S1:1",
                "sequence": "S1",
                "track_id": "1",
                "frame_start": 10,
                "frame_end": 20,
                "label": 1,
                "anomaly_type": "speed_spike",
                "injection_seed": 42,
            },
            {
                "sample_id": "S1:2",
                "sequence": "S1",
                "track_id": "2",
                "frame_start": 12,
                "frame_end": 18,
                "label": 1,
                "anomaly_type": "route_shift",
                "injection_seed": 42,
            },
            {
                "sample_id": "S2:3",
                "sequence": "S2",
                "track_id": "3",
                "frame_start": 0,
                "frame_end": 4,
                "label": 0,
                "anomaly_type": "normal",
                "injection_seed": 42,
            },
        ],
    )
    _write_jsonl(
        group_labels,
        [
            {
                "sample_id": "G1:1",
                "sequence": "G1",
                "track_id": "1",
                "frame_start": 1,
                "frame_end": 8,
                "label": 1,
                "anomaly_type": "population_change",
                "injection_seed": 42,
            },
            {
                "sample_id": "G1:2",
                "sequence": "G1",
                "track_id": "2",
                "frame_start": 1,
                "frame_end": 8,
                "label": 0,
                "anomaly_type": "normal",
                "injection_seed": 42,
            },
        ],
    )
    _write_jsonl(
        score_root / "individual" / "scores" / "fusiontrack_individual_nn.jsonl",
        [
            {"sample_id": "S1:1", "sequence": "S1", "track_id": "1", "score": 0.95},
            {"sample_id": "S2:3", "sequence": "S2", "track_id": "3", "score": 0.91},
            {"sample_id": "S1:2", "sequence": "S1", "track_id": "2", "score": 0.12},
        ],
    )
    _write_jsonl(
        score_root / "individual" / "scores" / "individual_lof.jsonl",
        [
            {"sample_id": "S1:2", "sequence": "S1", "track_id": "2", "score": 0.88},
            {"sample_id": "S2:3", "sequence": "S2", "track_id": "3", "score": 0.50},
            {"sample_id": "S1:1", "sequence": "S1", "track_id": "1", "score": 0.20},
        ],
    )
    _write_jsonl(
        score_root / "group" / "scores" / "group_prediction_linear.jsonl",
        [
            {"sample_id": "G1:2", "sequence": "G1", "track_id": "2", "score": 0.75},
            {"sample_id": "G1:1", "sequence": "G1", "track_id": "1", "score": 0.25},
        ],
    )
    _write_csv(
        final_root / "final_individual_all_methods_summary.csv",
        [
            {
                "method": "fusiontrack_individual_nn",
                "source": "/remote/results/individual/scores/fusiontrack_individual_nn.jsonl",
                "split": "val",
                "seed": 42,
                "auroc": 0.7,
                "auprc": 0.4,
                "f1": 0.5,
                "precision_at_k": 0.5,
                "recall_at_k": 0.5,
                "num_score_rows": 3,
                "num_missing_score_keys": 0,
            },
            {
                "method": "individual_lof",
                "source": "/remote/results/individual/scores/individual_lof.jsonl",
                "split": "val",
                "seed": 42,
                "auroc": 0.8,
                "auprc": 0.45,
                "f1": 0.55,
                "precision_at_k": 0.5,
                "recall_at_k": 0.5,
                "num_score_rows": 3,
                "num_missing_score_keys": 0,
            },
        ],
    )
    _write_csv(
        final_root / "final_individual_all_methods_categorized.csv",
        [
            {
                "method": "fusiontrack_individual_nn",
                "owner": "our_method",
                "source_type": "fusiontrack",
                "learning_type": "learning_nearest_neighbor_profile",
                "method_family": "KNN_feature_profile",
                "role": "proposed_method",
                "auroc": 0.7,
                "auprc": 0.4,
                "f1": 0.5,
                "precision_at_k": 0.5,
                "recall_at_k": 0.5,
                "num_score_rows": 3,
                "num_missing_score_keys": 0,
            },
            {
                "method": "individual_lof",
                "owner": "classic_baseline",
                "source_type": "local_classic_algorithm",
                "learning_type": "learning_classical_ml",
                "method_family": "LOF",
                "role": "main_baseline",
                "auroc": 0.8,
                "auprc": 0.45,
                "f1": 0.55,
                "precision_at_k": 0.5,
                "recall_at_k": 0.5,
                "num_score_rows": 3,
                "num_missing_score_keys": 0,
            },
        ],
    )
    _write_csv(
        final_root / "final_group_all_methods_summary.csv",
        [
            {
                "method": "group_prediction_linear",
                "source": "/remote/results/group/scores/group_prediction_linear.jsonl",
                "split": "val",
                "seed": 42,
                "auroc": 0.62,
                "auprc": 0.2,
                "f1": 0.3,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "num_score_rows": 2,
                "num_missing_score_keys": 0,
            }
        ],
    )
    _write_csv(
        final_root / "final_group_all_methods_categorized.csv",
        [
            {
                "method": "group_prediction_linear",
                "owner": "classic_baseline",
                "source_type": "local_classic_rule_baseline",
                "learning_type": "non_learning_rule_or_residual",
                "method_family": "linear_prediction_residual",
                "role": "main_baseline",
                "auroc": 0.62,
                "auprc": 0.2,
                "f1": 0.3,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "num_score_rows": 2,
                "num_missing_score_keys": 0,
            }
        ],
    )
    return final_root, score_root, individual_labels, group_labels


def test_load_final_results_dashboard_builds_leaderboards_type_stats_and_cases(tmp_path: Path) -> None:
    final_root, score_root, individual_labels, group_labels = _build_small_final_result_tree(tmp_path)

    dashboard = load_final_results_dashboard(
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
        top_k=2,
        case_limit=3,
    )

    assert dashboard.tasks["individual"].leaderboard[0]["method"] == "individual_lof"
    assert dashboard.tasks["individual"].leaderboard[1]["is_our_method"] is True
    assert dashboard.tasks["individual"].methods["fusiontrack_individual_nn"].score_path.name == "fusiontrack_individual_nn.jsonl"
    speed_row = next(
        row for row in dashboard.tasks["individual"].anomaly_type_rows
        if row["method"] == "fusiontrack_individual_nn" and row["anomaly_type"] == "speed_spike"
    )
    assert speed_row["hits_at_k"] == 1
    assert speed_row["total_positive"] == 1
    cases = dashboard.tasks["individual"].case_rows["fusiontrack_individual_nn"]
    assert cases["true_positive"][0]["sample_id"] == "S1:1"
    assert cases["false_positive"][0]["sample_id"] == "S2:3"
    assert cases["false_negative"][0]["sample_id"] == "S1:2"
    assert dashboard.tasks["group"].anomaly_type_rows[0]["anomaly_type"] == "population_change"


def test_build_final_dashboard_writes_method_switching_html(tmp_path: Path) -> None:
    final_root, score_root, individual_labels, group_labels = _build_small_final_result_tree(tmp_path)
    dashboard = load_final_results_dashboard(
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
        top_k=2,
        case_limit=3,
    )
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
                    {"frame_id": 10, "fused": {"center_xy": [10, 20], "confidence": 0.9}},
                    {"frame_id": 20, "fused": {"center_xy": [20, 30], "confidence": 0.8}},
                ],
            },
            {
                "sample_id": "S1:2",
                "sequence": "S1",
                "track_id": "2",
                "category_name": "plane",
                "points": [
                    {"frame_id": 10, "fused": {"center_xy": [30, 40], "confidence": 0.9}},
                    {"frame_id": 20, "fused": {"center_xy": [40, 50], "confidence": 0.8}},
                ],
            },
        ],
    )

    summary = build_final_dashboard(
        dashboard=dashboard,
        output_dir=tmp_path / "dashboard",
        fused_jsonl=fused_jsonl,
        data_root=tmp_path / "data",
        top_sequences=1,
    )

    assert summary["num_tasks"] == 2
    assert summary["num_methods"] == 3
    assert summary["playback_sequences"] == ["S1"]
    html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
    playback_data = json.loads((tmp_path / "dashboard" / "assets" / "final_playback_data.json").read_text(encoding="utf-8"))
    assert playback_data["S1"]["stats"] == {
        "sequence_sample_count": 2,
        "sequence_anomaly_count": 2,
        "frame_start": 10,
        "frame_end": 20,
        "visualized_tracks": 2,
    }
    assert "FusionTrack 最终结果看板" in html
    assert "总标签数" in html
    assert "总异常数" in html
    assert "当前序列样本数" in html
    assert "当前序列异常数" in html
    assert "当前序列帧范围" in html
    assert "可视化轨迹数" in html
    assert "sequenceStats" in html
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "control-surface" in html
    assert "section-heading" in html
    assert "focus-visible" in html
    assert "prefers-reduced-motion" in html
    assert ".view-mode-button { min-height: 44px" in html
    assert ".layer-button { min-height: 44px" in html
    assert "min-width: 760px" in html
    assert "四画面对比" in html
    assert "单画面模式" in html
    assert "originalCanvas" in html
    assert "heatmapCanvas" in html
    assert "tracksCanvas" in html
    assert "bothCanvas" in html
    assert "singleCanvas" in html
    assert "drawComparisonView" in html
    assert "drawCanvasLayer" in html
    assert "languageSelector" in html
    assert "localStorage" in html
    assert "translations" in html
    assert "data-analysis-panel=\"leaderboard\"" in html
    assert "data-analysis-panel=\"types\"" in html
    assert "data-analysis-panel=\"cases\"" in html
    assert "analysis-tab active" in html
    assert "<aside" not in html
    assert "methodCards" not in html
    assert html.index("Interactive playback") < html.index("data-analysis-panel=\"leaderboard\"")
    assert "典型案例" in html
    assert "truePositive" in html
    assert "methodSelector" in html
    assert "sequenceSelector" in html
    assert "frameSlider" in html
    assert "backgroundForFrame" in html
    assert "drawHeatmap" in html
    assert "heatOpacity" in html
    assert "heatWindow" in html
    assert "Heat + Tracks" in html
    assert "renderMethodView" in html
    assert "playbackData" in html
    assert "fusiontrack_individual_nn" in html
