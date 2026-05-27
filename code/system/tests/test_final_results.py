from __future__ import annotations

import csv
import json
from pathlib import Path

from fusiontrack.final_dashboard import _score_decomposition, build_final_dashboard
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


def _write_tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00"
        b"\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
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
            {
                "sample_id": "G1:2",
                "sequence": "G1",
                "track_id": "2",
                "score": 0.75,
                "event_score": 0.8,
                "component_scores": {"graph_leave": 0.8},
                "event_segments": [{"frame_start": 4, "frame_end": 8, "score": 0.8, "dominant_reason": "leave"}],
                "frame_event_scores": [
                    {"frame": 1, "score": 0.0, "dominant_reason": "object_group"},
                    {"frame": 8, "score": 0.8, "dominant_reason": "leave"},
                ],
            },
            {
                "sample_id": "G1:1",
                "sequence": "G1",
                "track_id": "1",
                "score": 0.25,
                "event_score": 0.55,
                "frame_event_scores": [
                    {"frame": 2, "score": 0.55, "dominant_reason": "dispersion", "component_scores": {"graph_dispersion": 0.55}},
                    {"frame": 3, "score": 0.25, "dominant_reason": "dispersion", "component_scores": {"graph_dispersion": 0.25}},
                ],
            },
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


def test_final_results_falls_back_to_central_method_registry(tmp_path: Path) -> None:
    final_root, score_root, individual_labels, group_labels = _build_small_final_result_tree(tmp_path)
    (final_root / "final_individual_all_methods_categorized.csv").unlink()
    (final_root / "final_group_all_methods_categorized.csv").unlink()

    dashboard = load_final_results_dashboard(
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
    )

    individual_method = dashboard.tasks["individual"].methods["fusiontrack_individual_nn"]
    assert individual_method.category["owner"] == "our_method"
    assert individual_method.category["role"] == "component"
    assert individual_method.category["method_family"] == "fusiontrack_nearest_neighbor"

    group_method = dashboard.tasks["group"].methods["group_prediction_linear"]
    assert group_method.category["owner"] == "classic_baseline"
    assert group_method.category["method_family"] == "linear_prediction_residual"


def test_central_method_registry_overrides_stale_categorized_fields(tmp_path: Path) -> None:
    final_root, score_root, individual_labels, group_labels = _build_small_final_result_tree(tmp_path)

    dashboard = load_final_results_dashboard(
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
    )

    method = dashboard.tasks["individual"].methods["fusiontrack_individual_nn"]
    assert method.category["role"] == "component"
    assert method.category["method_family"] == "fusiontrack_nearest_neighbor"
    assert method.category["registry_status"] == "registered"


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


def test_final_results_dashboard_exposes_schema_diagnostics(tmp_path: Path) -> None:
    final_root, score_root, individual_labels, group_labels = _build_small_final_result_tree(tmp_path)
    summary_csv = final_root / "final_individual_all_methods_summary.csv"
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["schema_diagnostics"] = {
        "schema_diagnostics_version": 1,
        "status": "warning",
        "key_fields": ["sample_id"],
        "label": {"num_rows": 3, "num_unique_keys": 3, "num_duplicate_keys": 0},
        "score": {"num_rows": 3, "num_unique_keys": 2, "num_duplicate_keys": 1},
        "alignment": {"num_missing_score_keys": 1, "num_extra_score_keys": 0},
        "warnings": ["missing_score_keys", "duplicate_score_keys"],
    }
    (final_root / "final_individual_all_methods_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False),
        encoding="utf-8",
    )

    dashboard = load_final_results_dashboard(
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
        top_k=2,
        case_limit=3,
    )

    method = dashboard.tasks["individual"].methods["fusiontrack_individual_nn"]
    diagnostics = method.metrics["schema_diagnostics"]
    assert diagnostics["status"] == "warning"
    assert diagnostics["alignment"]["num_missing_score_keys"] == 1
    leaderboard_row = next(
        row for row in dashboard.tasks["individual"].leaderboard
        if row["method"] == "fusiontrack_individual_nn"
    )
    assert leaderboard_row["schema_diagnostics"]["warnings"] == ["missing_score_keys", "duplicate_score_keys"]

    build_final_dashboard(
        dashboard=dashboard,
        output_dir=tmp_path / "dashboard",
        top_sequences=1,
    )

    dashboard_payload = json.loads(
        (tmp_path / "dashboard" / "assets" / "final_dashboard_data.json").read_text(encoding="utf-8")
    )
    public_metrics = dashboard_payload["tasks"]["individual"]["methods"]["fusiontrack_individual_nn"]["metrics"]
    assert public_metrics["schema_diagnostics"]["score"]["num_duplicate_keys"] == 1
    html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "schemaStatusHeader" in html
    assert "schemaWarningHeader" in html
    assert "missing_score_keys" in html
    assert "schemaDiagnosticsSummary" in html
    assert "exportViewJson" in html
    assert "exportLeaderboardCsv" in html
    assert "exportSequenceJson" in html
    assert "exportPlaybackPng" in html
    assert "fusiontrack_dashboard_view_export_v1" in html
    assert "rowsToCsv" in html
    assert "buildPlaybackPngCanvas" in html
    assert "handlePlaybackPngExport" in html


def test_score_decomposition_reads_fused_metadata_sources_and_explicit_components() -> None:
    row = {
        "score": 0.62,
        "event_score": 0.91,
        "component_scores": {
            "S_ind": 0.37,
            "S_grp": 0.82,
            "S_event": 0.91,
            "S_fused": 0.62,
            "individual_speed_score": 0.37,
            "group_graph_leave": 0.82,
        },
        "metadata": {
            "used_sources": ["individual", "group"],
            "alpha": 0.55,
            "individual_raw_score": 2.5,
            "group_raw_score": 4.0,
        },
    }

    decomp = _score_decomposition(row)

    assert decomp["S_ind"] == 0.37
    assert decomp["S_grp"] == 0.82
    assert decomp["S_event"] == 0.91
    assert decomp["S_fused"] == 0.62
    assert decomp["individual_source"] == 1.0
    assert decomp["group_source"] == 1.0
    assert decomp["alpha"] == 0.55


def test_score_decomposition_infers_single_branch_source_family() -> None:
    group_decomp = _score_decomposition(
        {
            "source": "fusiontrack_group_hybrid",
            "score": 0.73,
            "component_scores": {"graph_leave": 0.41},
        }
    )
    individual_decomp = _score_decomposition(
        {
            "source": "fusiontrack_individual:nearest_feature",
            "score": 0.58,
            "component_scores": {"route_score": 0.34},
        }
    )

    assert group_decomp["S_grp"] == 0.73
    assert group_decomp["group_source"] == 1.0
    assert individual_decomp["S_ind"] == 0.58
    assert individual_decomp["individual_source"] == 1.0


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
                    {
                        "frame_id": 10,
                        "fused": {"center_xy": [10, 20], "confidence": 0.9},
                        "rgb": {"file": "S1/00/00010.jpg"},
                        "thermal": {"file": "S1/01/00010.jpg"},
                        "modal": {"offset_distance": 2.5},
                    },
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
            {
                "sample_id": "G1:1",
                "sequence": "G1",
                "track_id": "1",
                "category_name": "plane",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [15, 25], "confidence": 0.9}},
                    {"frame_id": 8, "fused": {"center_xy": [25, 35], "confidence": 0.8}},
                ],
            },
            {
                "sample_id": "G1:2",
                "sequence": "G1",
                "track_id": "2",
                "category_name": "plane",
                "points": [
                    {"frame_id": 1, "fused": {"center_xy": [35, 45], "confidence": 0.9}},
                    {"frame_id": 8, "fused": {"center_xy": [45, 55], "confidence": 0.8}},
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
        provenance={
            "mode": "final_results_dashboard",
            "generated_at_utc": "2026-05-26T00:00:00+00:00",
            "dataset_manifest": {
                "dataset_name": "VT-Tiny-MOT",
                "status": "ok",
                "dataset_fingerprint": "fingerprint-for-test",
                "splits": {
                    "test": {
                        "requested_split": "test",
                        "quality": {
                            "status": "partial",
                            "num_observation_keys": 3,
                            "num_rgb_annotations": 2,
                            "num_thermal_annotations": 2,
                            "num_paired_annotations": 1,
                            "num_missing_rgb_annotations": 1,
                            "num_missing_thermal_annotations": 1,
                            "rgb_annotation_coverage": 0.666667,
                            "thermal_annotation_coverage": 0.666667,
                            "paired_annotation_coverage": 0.333333,
                            "modal_offset_mean": 5.0,
                            "modal_offset_max": 5.0,
                        },
                    }
                },
            },
            "dataset_manifest_path": tmp_path / "work" / "dataset_manifest_all.json",
            "final_results_root": final_root,
            "individual_label_file": individual_labels,
            "group_label_file": group_labels,
            "score_search_roots": [score_root, tmp_path / "official_scores"],
            "fused_jsonl": fused_jsonl,
            "registration_manifest": tmp_path / "registration" / "manifest.json",
            "top_sequences": 1,
            "top_k": 2,
            "case_limit": 3,
        },
    )

    assert summary["num_tasks"] == 2
    assert summary["num_methods"] == 3
    assert summary["playback_sequences"] == ["S1", "G1"]
    html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
    dashboard_payload = json.loads((tmp_path / "dashboard" / "assets" / "final_dashboard_data.json").read_text(encoding="utf-8"))
    playback_data = json.loads((tmp_path / "dashboard" / "assets" / "final_playback_data.json").read_text(encoding="utf-8"))
    provenance = dashboard_payload["provenance"]
    assert provenance["dataset"]["fingerprint"] == "fingerprint-for-test"
    assert provenance["dataset"]["status"] == "ok"
    assert provenance["dataset"]["quality"]["num_observation_keys"] == 3
    assert provenance["dataset"]["quality"]["rgb_annotation_coverage"] == 0.666667
    assert provenance["inputs"]["score_search_root_count"] == 2
    assert provenance["inputs"]["final_results_root"] == "final"
    assert provenance["inputs"]["registration_manifest"] == "manifest.json"
    assert str(tmp_path) not in json.dumps(provenance)
    assert "{dashboard_json}" not in html
    assert "{playback_json}" not in html
    assert '"tasks"' in html
    assert playback_data["S1"]["stats"] == {
        "sequence_sample_count": 2,
        "sequence_anomaly_count": 2,
        "frame_start": 10,
        "frame_end": 20,
        "visualized_tracks": 2,
    }
    assert playback_data["S1"]["stats_by_task"]["group"]["sequence_sample_count"] == 0
    assert playback_data["S1"]["modality_audit"]["point_count"] == 4
    assert playback_data["S1"]["modality_audit"]["rgb_point_count"] == 1
    assert playback_data["S1"]["modality_audit"]["thermal_point_count"] == 1
    assert playback_data["S1"]["modality_audit"]["missing_thermal_points"] == 3
    assert playback_data["S1"]["modality_audit"]["background_status"] == "missing"
    assert playback_data["S1"]["media"]["kind"] == "track_only_missing_background"
    assert playback_data["S1"]["media"]["has_original_background"] is False
    assert playback_data["S1"]["media"]["explanation_key"] == "sequenceNoVideoBackground"
    assert playback_data["S1"]["modality_audit"]["modal_offset_mean"] == 2.5
    group_tracks = {track["sample_id"]: track for track in playback_data["G1"]["tracks"]}
    assert playback_data["G1"]["stats_by_task"]["group"] == {
        "sequence_sample_count": 2,
        "sequence_anomaly_count": 1,
        "frame_start": 1,
        "frame_end": 8,
        "visualized_tracks": 2,
    }
    assert group_tracks["G1:2"]["task_scores"]["group"]["group_prediction_linear"] == 0.75
    group_components = group_tracks["G1:2"]["task_score_components"]["group"]["group_prediction_linear"]
    assert group_components["event_segments"] == [
        {"frame_start": 4, "frame_end": 8, "score": 0.8, "dominant_reason": "leave"}
    ]
    assert group_components["frame_event_scores"][1]["frame"] == 8
    assert group_components["frame_event_scores"][1]["dominant_reason"] == "leave"
    assert group_components["explanation_schema"]["top_reason"] == "leave"
    assert group_components["explanation_schema"]["evidence_source"] == "event_segments"
    assert group_components["explanation_schema"]["score_components"][0]["name"] == "graph_leave"
    derived_components = group_tracks["G1:1"]["task_score_components"]["group"]["group_prediction_linear"]
    assert derived_components["event_segments"] == [
        {
            "frame_start": 2,
            "frame_end": 3,
            "score": 0.55,
            "dominant_reason": "dispersion",
            "num_frames": 2,
            "component_scores": {"graph_dispersion": 0.55},
        }
    ]
    assert derived_components["explanation_schema"]["top_reason"] == "dispersion"
    assert derived_components["explanation_schema"]["evidence_source"] == "frame_event_scores"
    assert group_tracks["G1:1"]["task_labels"]["group"]["label"] == 1
    assert "FusionTrack 最终结果看板" in html
    assert "总标签数" in html
    assert "总异常数" in html
    assert "当前序列样本数" in html
    assert "当前序列异常数" in html
    assert "当前序列帧范围" in html
    assert "可视化轨迹数" in html
    assert "sequenceStats" in html
    assert "helpButton" in html
    assert "helpDialog" in html
    assert "异常协议" in html
    assert "synthetic anomaly injection" in html
    assert "individualProtocol" in html
    assert "groupProtocol" in html
    assert "trackRankList" in html
    assert "explanationPanel" in html
    assert "groupInsightPanel" in html
    assert "methodStatusTable" in html
    assert "算法接入" in html
    assert "sequencesForTask" in html
    assert "compareSequencesForTask" in html
    assert "Number(label.num_windows || 0) > 0" not in html
    assert "const hasLabel = Number(label.label || 0) === 1;" in html
    assert "trackScores" in html
    assert "trackScoresForTask" in html
    assert "trackLabelValue" in html
    assert "drawGroupRelations" in html
    assert "pickTrackFromCanvas" in html
    assert "renderProtocolOverview" in html
    assert "renderMethodStatus" in html
    assert "renderTrackInsights" in html
    assert "explanationSchemaReason" in html
    assert "renderDataFlowAudit" in html
    assert "renderProvenanceAudit" in html
    assert "datasetQualityPanel" in html
    assert "provenanceDatasetQuality" in html
    assert "provenancePanel" in html
    assert "provenanceDatasetFingerprint" in html
    assert "fingerprint-for-test" in html
    assert "modality_audit" in html
    assert "序列数据审计" in html
    assert "dataFlowRgbCoverage" in html
    assert "submoduleCurve" in html
    assert "aggregateGroupEvents" in html
    assert "dataFlowPanel" in html
    assert "anomalyDescriptions" in html
    assert 'state.task !== "individual"' not in html
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
    assert "presentationModeSelector" in html
    assert "fusiontrack.finalDashboard.presentationMode" in html
    assert "data-i18n-option" in html
    assert "demoSummary" in html
    assert "demoPipelineSummary" in html
    assert "demoResultSummary" in html
    assert "demoEvidenceSummary" in html
    assert "systemCoveragePanel" in html
    assert "renderSystemCoverage" in html
    assert "moduleCoverageTitle" in html
    assert "moduleStatusDone" in html
    assert "moduleRegistration" in html
    assert "moduleVisualization" in html
    assert "audit-only" in html
    assert "applyPresentationMode" in html
    assert "renderDemoSummary" in html
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
    assert "eventThreshold" in html
    assert "eventThresholdReadout" in html
    assert "frameEventScoresForWindow" in html
    assert "windowEventSummary" in html
    assert "windowSegmentSummary" in html
    assert "renderWindowEventEvidence" in html
    assert "windowEventTitle" in html
    assert "windowEventSourceSegment" in html
    assert "eventSegmentsFromFrameScores" in html
    assert "handlePlaybackKeydown" in html
    assert "shouldIgnorePlaybackShortcut" in html
    assert "ArrowRight" in html
    assert "Space" in html
    assert "Digit1" in html
    assert "Heat + Tracks" in html
    assert "renderMethodView" in html
    assert "playbackData" in html
    assert "fusiontrack_individual_nn" in html


def test_final_dashboard_background_frames_include_fallback_source(tmp_path: Path) -> None:
    final_root, score_root, individual_labels, group_labels = _build_small_final_result_tree(tmp_path)
    dashboard = load_final_results_dashboard(
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
        top_k=2,
        case_limit=3,
    )
    data_root = tmp_path / "data"
    _write_tiny_png(data_root / "S1" / "00" / "00010.png")
    _write_tiny_png(data_root / "S1" / "00" / "00020.png")
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
                    {
                        "frame_id": 10,
                        "fused": {"center_xy": [10, 20], "confidence": 0.9},
                        "rgb": {"file": "S1/00/00010.png"},
                    },
                    {
                        "frame_id": 20,
                        "fused": {"center_xy": [20, 30], "confidence": 0.8},
                        "rgb": {"file": "S1/00/00020.png"},
                    },
                ],
            }
        ],
    )

    build_final_dashboard(
        dashboard=dashboard,
        output_dir=tmp_path / "dashboard",
        fused_jsonl=fused_jsonl,
        data_root=data_root,
        top_sequences=1,
    )

    html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
    playback_data = json.loads(
        (tmp_path / "dashboard" / "assets" / "final_playback_data.json").read_text(encoding="utf-8")
    )
    assert playback_data["S1"]["background"] == "assets/background_S1.png"
    assert playback_data["S1"]["background_frames"] == [
        {"frame": 10, "src": "assets/background_S1_000010.png", "fallback_src": "assets/background_S1.png"},
        {"frame": 20, "src": "assets/background_S1_000020.png", "fallback_src": "assets/background_S1.png"},
    ]
    assert "fallback_src" in html
    assert "backgroundFallbackForFrame" in html
    assert "backgroundLoadFailed" in html


def test_final_dashboard_publishes_holdout_multiseed_provenance(tmp_path: Path) -> None:
    final_root, score_root, individual_labels, group_labels = _build_small_final_result_tree(tmp_path)
    dashboard = load_final_results_dashboard(
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root],
        top_k=2,
        case_limit=3,
    )
    holdout_dir = tmp_path / "holdout"
    aggregate_csv = holdout_dir / "aggregate.csv"
    best_by_metric_json = holdout_dir / "best_by_metric.json"
    manifest_path = holdout_dir / "manifest.json"
    _write_csv(
        aggregate_csv,
        [
            {
                "level": "individual",
                "method": "individual_lof",
                "task": "individual_classical",
                "num_runs": 3,
                "seeds": "42,43,44",
                "auroc_mean": 0.61,
                "auroc_std": 0.02,
                "auprc_mean": 0.19,
                "auprc_std": 0.04,
                "f1_mean": 0.23,
                "f1_std": 0.03,
            },
            {
                "level": "group",
                "method": "fusiontrack_group_hybrid",
                "task": "fusiontrack_group_hybrid",
                "num_runs": 3,
                "seeds": "42,43,44",
                "auroc_mean": 0.79,
                "auroc_std": 0.01,
                "auprc_mean": 0.09,
                "auprc_std": 0.02,
                "f1_mean": 0.16,
                "f1_std": 0.02,
            },
        ],
    )
    best_by_metric_json.parent.mkdir(parents=True, exist_ok=True)
    best_by_metric_json.write_text(
        json.dumps(
            {
                "auroc": {
                    "level": "group",
                    "method": "fusiontrack_group_hybrid",
                    "task": "fusiontrack_group_hybrid",
                    "num_runs": 3,
                    "auroc_mean": 0.79,
                    "auroc_std": 0.01,
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "aggregate_csv": "/root/autodl-tmp/fusiontrack_holdout/aggregate.csv",
        "all_runs_csv": "/root/autodl-tmp/fusiontrack_holdout/all_runs.csv",
        "best_by_metric_json": "/root/autodl-tmp/fusiontrack_holdout/best_by_metric.json",
        "split_name": "test",
        "train_source_split": "train",
        "eval_source_split": "test",
        "levels": ["individual", "group"],
        "seeds": [42, 43, 44],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    build_final_dashboard(
        dashboard=dashboard,
        output_dir=tmp_path / "dashboard",
        top_sequences=1,
        provenance={
            "mode": "final_results_dashboard",
            "dataset_manifest": {"dataset_name": "VT-Tiny-MOT", "status": "ok", "splits": {}},
            "holdout_manifest": manifest,
            "holdout_manifest_path": manifest_path,
        },
    )

    dashboard_payload = json.loads(
        (tmp_path / "dashboard" / "assets" / "final_dashboard_data.json").read_text(encoding="utf-8")
    )
    holdout = dashboard_payload["provenance"]["holdout"]
    assert holdout["seeds"] == [42, 43, 44]
    assert holdout["levels"] == ["individual", "group"]
    assert holdout["split_name"] == "test"
    assert holdout["aggregate_csv"] == "aggregate.csv"
    assert holdout["best_by_metric_json"] == "best_by_metric.json"
    assert holdout["top_methods"][0]["method"] == "fusiontrack_group_hybrid"
    assert holdout["best_by_metric"]["auroc"]["method"] == "fusiontrack_group_hybrid"
    assert str(tmp_path) not in json.dumps(holdout)
    html = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "holdoutPanel" in html
    assert "provenanceHoldout" in html
    assert "fusiontrack_group_hybrid" in html


def test_final_dashboard_includes_registration_playback_without_labels(tmp_path: Path) -> None:
    final_root, score_root, individual_labels, group_labels = _build_small_final_result_tree(tmp_path)
    registration_root = tmp_path / "registration_work"
    registration_score = registration_root / "registration_scores" / "icp_registration_scores.jsonl"
    registration_metrics = registration_root / "registration_metrics" / "icp_registration_metrics.json"
    registration_manifest = registration_root / "registration_artifacts" / "registration_experiment_manifest.json"
    _write_jsonl(
        registration_score,
        [
            {
                "sample_id": "icp:0:0:0",
                "sequence": "R1",
                "track_id": "0",
                "score": 0.41,
                "used_sources": "registration",
                "rotation_error_deg": 4.2,
                "translation_error": 0.12,
                "chamfer_distance": 0.08,
                "runtime_sec": 0.03,
                "success": True,
                "skipped": False,
                "component_scores": {"registration_error_score": 0.41},
            },
            {
                "sample_id": "icp:0:1:0",
                "sequence": "R1",
                "track_id": "1",
                "score": 1.4,
                "used_sources": "registration",
                "rotation_error_deg": 18.0,
                "translation_error": 0.7,
                "chamfer_distance": 0.34,
                "runtime_sec": 0.04,
                "success": False,
                "skipped": False,
                "component_scores": {"registration_error_score": 1.4},
            },
        ],
    )
    registration_metrics.parent.mkdir(parents=True, exist_ok=True)
    registration_metrics.write_text(
        json.dumps(
            {
                "success_rate": 0.5,
                "num_pairs": 2,
                "num_successful_pairs": 1,
                "num_failed_pairs": 1,
                "rotation_error_deg_mean": 11.1,
                "translation_error_mean": 0.41,
                "chamfer_distance_mean": 0.21,
                "runtime_sec_mean": 0.035,
            }
        ),
        encoding="utf-8",
    )
    registration_manifest.parent.mkdir(parents=True, exist_ok=True)
    registration_manifest.write_text(
        json.dumps(
            {
                "task": "registration",
                "split": "test",
                "runs": [
                    {
                        "name": "icp",
                        "score_file": "registration_scores/icp_registration_scores.jsonl",
                        "metrics_file": "registration_metrics/icp_registration_metrics.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dashboard = load_final_results_dashboard(
        final_results_root=final_root,
        individual_label_file=individual_labels,
        group_label_file=group_labels,
        score_search_roots=[score_root, registration_root],
        registration_manifest=registration_manifest,
        top_k=2,
        case_limit=3,
    )
    fused_jsonl = tmp_path / "fused_with_registration.jsonl"
    _write_jsonl(
        fused_jsonl,
        [
            {
                "sample_id": "S1:1",
                "sequence": "S1",
                "track_id": "1",
                "points": [{"frame_id": 10, "fused": {"center_xy": [10, 20], "confidence": 0.9}}],
            },
            {
                "sample_id": "G1:1",
                "sequence": "G1",
                "track_id": "1",
                "points": [{"frame_id": 1, "fused": {"center_xy": [15, 25], "confidence": 0.9}}],
            },
            {
                "sample_id": "icp:0:0:0",
                "sequence": "R1",
                "track_id": "0",
                "points": [{"frame_id": 0, "fused": {"center_xy": [20, 30], "confidence": 0.41}}],
            },
            {
                "sample_id": "icp:0:1:0",
                "sequence": "R1",
                "track_id": "1",
                "points": [{"frame_id": 0, "fused": {"center_xy": [40, 50], "confidence": 1.4}}],
            },
        ],
    )

    summary = build_final_dashboard(
        dashboard=dashboard,
        output_dir=tmp_path / "registration_dashboard",
        fused_jsonl=fused_jsonl,
        data_root=tmp_path / "data",
        top_sequences=1,
    )

    playback_data = json.loads(
        (tmp_path / "registration_dashboard" / "assets" / "final_playback_data.json").read_text(encoding="utf-8")
    )
    assert "registration" in dashboard.tasks
    assert "R1" in summary["playback_sequences"]
    assert playback_data["R1"]["stats_by_task"]["registration"]["sequence_sample_count"] == 2
    assert playback_data["R1"]["tracks"][0]["task_score_components"]["registration"]["icp"]["rotation_error_deg"] is not None
    html = (tmp_path / "registration_dashboard" / "index.html").read_text(encoding="utf-8")
    assert "registrationMetricRotation" in html
    assert "配准任务展示非学习基线" in html
    assert "registration3DTitle" in html
    assert "renderRegistrationPointCloud" in html
    assert "backgroundNotice" in html
    assert "hasVideoBackground" in html
    assert "registrationNoVideoBackground" in html
    assert "syncPlaybackModeForData" in html
    assert "canvasPlaceholderText" in html
    assert "playbackMediaKind" in html
    assert "mediaStatusStrip" in html
    assert "sequenceOptionLabel" in html
    assert "mediaStatusOriginalBackground" in html
    assert "mediaStatusRegistrationBackground" in html
    assert "sequenceOptionPointCloud" in html
    assert "mediaKindRegistration" in html
    assert "registrationPlaybackView" in html
    assert "registrationCanvas" in html
    assert "drawRegistrationPlayback" in html
    assert "setPlaybackSurfaceForTask" in html
    assert playback_data["R1"]["media"]["kind"] == "registration_point_cloud"
    assert playback_data["R1"]["media"]["has_original_background"] is False
    assert playback_data["R1"]["media"]["explanation_key"] == "registrationNoVideoBackground"
