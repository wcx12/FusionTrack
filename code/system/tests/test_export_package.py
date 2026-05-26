from __future__ import annotations

import json
import zipfile
from pathlib import Path

from fusiontrack.export_package import build_analysis_export_package


def test_build_analysis_export_package_collects_report_and_sanitizes_paths(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"
    report_dir = work_root / "final_dashboard"
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True)
    (report_dir / "index.html").write_text("<html>FusionTrack</html>", encoding="utf-8")
    (assets_dir / "final_dashboard_data.json").write_text('{"tasks":{}}', encoding="utf-8")
    (assets_dir / "curve.png").write_bytes(b"png")
    summary_path = work_root / "pipeline_summary_final_dashboard.json"
    manifest_path = work_root / "pipeline_manifest_final_dashboard_all.json"
    dataset_manifest_path = work_root / "dataset_manifest_all.json"
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    suite_manifest_path = suite_dir / "suite_manifest.json"
    aggregate_summary_path = suite_dir / "aggregate_summary.csv"
    aggregate_summary_path.write_text("matrix,method,auroc\nindividual,fusiontrack,0.91\n", encoding="utf-8")
    suite_manifest_path.write_text(
        json.dumps(
            {
                "suite_name": "paper_suite",
                "aggregate_summary_csv": str(aggregate_summary_path),
                "matrices": [],
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "mode": "final_results_dashboard",
        "work_root": str(work_root),
        "data_root": str(tmp_path / "data" / "VT-Tiny-MOT"),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "suite_manifest_path": str(suite_manifest_path),
        "suite_manifest": {
            "suite_name": "paper_suite",
            "aggregate_summary_csv": str(aggregate_summary_path),
            "matrices": [],
        },
        "dataset_manifest": {"status": "ok", "dataset_fingerprint": "abc123"},
        "dashboard": {
            "report_html": str(report_dir / "index.html"),
            "assets_dir": str(assets_dir),
            "num_tasks": 2,
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps({"mode": "final_results_dashboard"}), encoding="utf-8")
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "dataset_fingerprint": "abc123",
                "data_root": str(tmp_path / "data" / "VT-Tiny-MOT"),
                "annotation_dir": str(tmp_path / "data" / "VT-Tiny-MOT" / "annotations"),
            }
        ),
        encoding="utf-8",
    )

    package_path = tmp_path / "exports" / "fusiontrack_export.zip"

    result = build_analysis_export_package(summary, package_path)

    assert result["package_path"] == str(package_path)
    assert result["num_files"] >= 5
    assert package_path.exists()
    with zipfile.ZipFile(package_path) as archive:
        names = set(archive.namelist())
        assert "report/index.html" in names
        assert "report/assets/final_dashboard_data.json" in names
        assert "report/assets/curve.png" in names
        assert "summary/pipeline_summary.json" in names
        assert "summary/pipeline_manifest.json" in names
        assert "artifacts/work_root/dataset_manifest_all.json" in names
        assert "artifacts/suite_root/suite_manifest.json" in names
        assert "artifacts/suite_root/aggregate_summary.csv" in names
        assert "export_manifest.json" in names
        assert not any(name.startswith("artifacts/work_root/pipeline_summary") for name in names)
        dataset_manifest_text = archive.read("artifacts/work_root/dataset_manifest_all.json").decode("utf-8")
        assert str(tmp_path) not in dataset_manifest_text
        assert "${data_root}/annotations" in dataset_manifest_text
        manifest = json.loads(archive.read("export_manifest.json").decode("utf-8"))
        manifest_text = json.dumps(manifest, ensure_ascii=False)
        assert str(tmp_path) not in manifest_text
        assert "${work_root}/final_dashboard/index.html" in manifest_text
        assert manifest["package_format"] == "fusiontrack_analysis_export_v1"
