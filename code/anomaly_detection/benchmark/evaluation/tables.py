from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable


TABLE_COLUMNS = [
    ("method", "Method"),
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("best_f1", "F1"),
    ("precision_at_k", "Precision@K"),
    ("recall_at_k", "Recall@K"),
]

LEVEL_TOKENS = {
    "individual": ("individual", "person", "single"),
    "group": ("group", "graph", "collective", "team"),
}

LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def load_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def filter_rows(rows: Iterable[dict[str, Any]], level: str | None = None) -> list[dict[str, Any]]:
    row_list = list(rows)
    if level is None:
        return row_list
    if level not in LEVEL_TOKENS:
        raise ValueError(f"level must be one of {sorted(LEVEL_TOKENS)} or None")
    return [row for row in row_list if _infer_level(row) == level]


def rank_rows(rows: Iterable[dict[str, Any]], metric: str = "auprc") -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[bool, float]:
        value = _row_metric_value(row, metric)
        return value is not None, value if value is not None else float("-inf")

    return sorted(
        list(rows),
        key=sort_key,
        reverse=True,
    )


def build_markdown_table(rows: Iterable[dict[str, Any]], metric: str = "auprc") -> str:
    ranked = rank_rows(rows, metric=metric)
    headers = [label for _, label in TABLE_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in ranked:
        values = [_markdown_cell(row, key) for key, _ in TABLE_COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_latex_table(
    rows: Iterable[dict[str, Any]],
    metric: str = "auprc",
    caption: str | None = None,
    label: str | None = None,
) -> str:
    ranked = rank_rows(rows, metric=metric)
    best_metric = _best_metric_value(ranked, metric)
    column_spec = "l" + "r" * (len(TABLE_COLUMNS) - 1)
    lines = [
        rf"\begin{{table}}[htbp]",
        r"\centering",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(label for _, label in TABLE_COLUMNS) + r" \\",
        r"\hline",
    ]
    for row in ranked:
        values = [
            _latex_cell(
                row,
                key,
                bold=key == metric
                and best_metric is not None
                and _row_metric_value(row, metric) == best_metric,
            )
            for key, _ in TABLE_COLUMNS
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    if caption is not None:
        lines.append(rf"\caption{{{_latex_escape(caption)}}}")
    if label is not None:
        lines.append(rf"\label{{{_latex_escape(label)}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def export_report_tables(
    summary_csv: Path,
    output_dir: Path,
    metric: str = "auprc",
) -> dict[str, Path]:
    rows = load_summary_rows(summary_csv)
    ranked = rank_rows(rows, metric=metric)
    individual_rows = filter_rows(rows, level="individual")
    group_rows = filter_rows(rows, level="group")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "all_methods_md": output_dir / "all_methods.md",
        "all_methods_tex": output_dir / "all_methods.tex",
        "individual_main_tex": output_dir / "individual_main.tex",
        "group_main_tex": output_dir / "group_main.tex",
        "best_by_metric_csv": output_dir / "best_by_metric.csv",
    }

    manifest["all_methods_md"].write_text(
        build_markdown_table(ranked, metric=metric),
        encoding="utf-8",
    )
    manifest["all_methods_tex"].write_text(
        build_latex_table(
            ranked,
            metric=metric,
            caption="All benchmark methods",
            label="tab:all-methods",
        ),
        encoding="utf-8",
    )
    manifest["individual_main_tex"].write_text(
        build_latex_table(
            individual_rows,
            metric=metric,
            caption="Individual anomaly detection benchmark",
            label="tab:individual-main",
        ),
        encoding="utf-8",
    )
    manifest["group_main_tex"].write_text(
        build_latex_table(
            group_rows,
            metric=metric,
            caption="Group anomaly detection benchmark",
            label="tab:group-main",
        ),
        encoding="utf-8",
    )
    _write_ranked_csv(manifest["best_by_metric_csv"], ranked)
    return manifest


def _infer_level(row: dict[str, Any]) -> str | None:
    task = str(row.get("task", "")).lower()
    if task.startswith("individual") or "_individual" in task:
        return "individual"
    if task.startswith("group") or "_group" in task:
        return "group"

    method = str(row.get("method", "")).lower()
    if method.startswith("individual_") or "_individual_" in method:
        return "individual"
    if method.startswith("group_") or "_group_" in method:
        return "group"

    haystack = " ".join(
        (
            task,
            method,
            str(row.get("source", "")).lower().replace("\\", "/"),
        )
    )
    matches = [
        level
        for level, tokens in LEVEL_TOKENS.items()
        if any(token in haystack for token in tokens)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _metric_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _best_metric_value(rows: Iterable[dict[str, Any]], metric: str) -> float | None:
    values = [_row_metric_value(row, metric) for row in rows]
    valid_values = [value for value in values if value is not None]
    return max(valid_values) if valid_values else None


def _markdown_cell(row: dict[str, Any], key: str) -> str:
    if key == "method":
        return str(row.get(key) or "--")
    return _format_metric_cell(row, key, latex=False)


def _latex_cell(row: dict[str, Any], key: str, bold: bool = False) -> str:
    if key == "method":
        return _latex_escape(str(row.get(key) or "--"))
    value = _format_metric_cell(row, key, latex=True)
    if bold and value != "--":
        return rf"\textbf{{{value}}}"
    return value


def _format_metric_cell(row: dict[str, Any], key: str, latex: bool) -> str:
    value = _row_metric_value(row, key)
    if value is None:
        return "--"
    std_value = _row_metric_std(row, key)
    formatted = _format_number(value)
    if std_value is None:
        return formatted
    separator = r" $\pm$ " if latex else " +/- "
    return f"{formatted}{separator}{_format_number(std_value)}"


def _format_number(value: Any) -> str:
    numeric_value = _metric_value(value)
    if numeric_value is None:
        return "--"
    return f"{numeric_value:.4f}"


def _row_value(row: dict[str, Any], key: str) -> Any:
    if key == "best_f1" and key not in row:
        return row.get("f1")
    return row.get(key)


def _row_metric_value(row: dict[str, Any], key: str) -> float | None:
    value = _metric_value(_row_value(row, key))
    if value is not None:
        return value
    return _metric_value(_row_aggregate_value(row, key, "mean"))


def _row_metric_std(row: dict[str, Any], key: str) -> float | None:
    return _metric_value(_row_aggregate_value(row, key, "std"))


def _row_aggregate_value(row: dict[str, Any], key: str, suffix: str) -> Any:
    field = f"{key}_{suffix}"
    if field in row:
        return row.get(field)
    if key == "best_f1":
        return row.get(f"f1_{suffix}")
    return None


def _latex_escape(value: str) -> str:
    return "".join(LATEX_REPLACEMENTS.get(char, char) for char in value)


def _write_ranked_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    priority = [
        "method",
        "task",
        "source",
        "split",
        "seed",
        "auroc",
        "auprc",
        "best_f1",
        "precision_at_k",
        "recall_at_k",
    ]
    if not rows:
        return priority
    seen = set()
    fieldnames: list[str] = []
    for field in priority:
        if any(field in row for row in rows):
            fieldnames.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)
    return fieldnames
