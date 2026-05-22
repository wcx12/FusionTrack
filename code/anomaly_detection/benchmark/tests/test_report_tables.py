from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.tables import (
    build_latex_table,
    export_report_tables,
    filter_rows,
    rank_rows,
)


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_latex_table_bolds_best_metric_and_escapes_method_names() -> None:
    rows = [
        {"method": "plain", "auroc": "0.7", "auprc": "0.8", "best_f1": "0.6"},
        {"method": "best_method", "auroc": "0.8", "auprc": "0.95", "best_f1": "0.7"},
    ]

    table = build_latex_table(rows, metric="auprc", caption="Main", label="tab:main")

    assert "best\\_method" in table
    assert "\\textbf{0.9500}" in table
    assert "\\caption{Main}" in table
    assert "\\label{tab:main}" in table


def test_rank_rows_sorts_empty_metric_last_and_latex_renders_empty_as_dash() -> None:
    rows = [
        {"method": "missing", "auprc": "", "auroc": ""},
        {"method": "not_finite", "auprc": "nan", "auroc": "inf"},
        {"method": "low", "auprc": "0.1", "auroc": "0.2"},
        {"method": "high", "auprc": "0.9", "auroc": "0.8"},
    ]

    ranked = rank_rows(rows, metric="auprc")
    table = build_latex_table(ranked, metric="auprc")

    assert [row["method"] for row in ranked] == ["high", "low", "missing", "not_finite"]
    assert "missing & -- & --" in table
    assert "not\\_finite & -- & --" in table


def test_filter_rows_infers_individual_and_group_from_task_or_method() -> None:
    rows = [
        {"method": "ocsvm", "task": "individual_anomaly"},
        {"method": "group_graph", "task": "benchmark"},
        {"method": "fusiontrack_group_graph", "task": "fusiontrack_group"},
        {"method": "fusiontrack_individual", "source": "scores.jsonl"},
        {"method": "unknown", "source": "somewhere/group/scores.jsonl"},
        {"method": "unknown", "source": "scores.jsonl"},
    ]

    individual = filter_rows(rows, level="individual")
    group = filter_rows(rows, level="group")

    assert [row["method"] for row in individual] == ["ocsvm", "fusiontrack_individual"]
    assert [row["method"] for row in group] == [
        "group_graph",
        "fusiontrack_group_graph",
        "unknown",
    ]
    assert filter_rows(rows, level=None) == rows


def test_export_report_tables_writes_expected_files(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    output_dir = tmp_path / "tables"
    _write_summary(
        summary_csv,
        [
            {
                "method": "individual_iforest",
                "task": "individual",
                "auroc": "0.8",
                "auprc": "0.7",
                "best_f1": "0.6",
                "precision_at_k": "0.5",
                "recall_at_k": "0.4",
            },
            {
                "method": "group_graph",
                "task": "group",
                "auroc": "0.7",
                "auprc": "0.9",
                "best_f1": "0.8",
                "precision_at_k": "0.6",
                "recall_at_k": "0.5",
            },
        ],
    )

    manifest = export_report_tables(summary_csv, output_dir, metric="auprc")

    assert set(manifest) == {
        "all_methods_md",
        "all_methods_tex",
        "individual_main_tex",
        "group_main_tex",
        "best_by_metric_csv",
    }
    for path in manifest.values():
        assert path.exists()
    assert "group\\_graph" in manifest["group_main_tex"].read_text(encoding="utf-8")


def test_export_report_tables_cli_direct_script_generates_manifest_and_files(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    output_dir = tmp_path / "tables"
    repo_root = Path(__file__).resolve().parents[4]
    script = Path("code/anomaly_detection/benchmark/runners/export_report_tables.py")
    _write_summary(
        summary_csv,
        [
            {"method": "individual_iforest", "task": "individual", "auprc": "0.7"},
            {"method": "group_graph", "task": "group", "auprc": "0.9"},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--summary-csv",
            str(summary_csv),
            "--output-dir",
            str(output_dir),
            "--metric",
            "auprc",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["all_methods_tex"] == str(output_dir / "all_methods.tex")
    assert (output_dir / "best_by_metric.csv").exists()
