from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


TRAINING_MONITORS = ("train_loss", "loss")


def summarize_convergence(
    history: Iterable[dict[str, Any]],
    requested_epochs: int,
    monitor: str = "val_loss",
    patience: int = 5,
    min_delta: float = 0.001,
    restore_best_checkpoint: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in history]
    metric_name, monitor_source, limitation = _select_monitor(rows, monitor)
    values = _metric_values(rows, metric_name) if metric_name is not None else []
    summary: dict[str, Any] = {
        "status": "not-evaluated",
        "requested_epochs": int(requested_epochs),
        "final_epoch": _final_epoch(rows),
        "best_epoch": None,
        "best_value": None,
        "monitor": metric_name,
        "monitor_source": monitor_source,
        "patience": int(patience),
        "min_delta": float(min_delta),
        "epochs_since_best": None,
        "restore_best_checkpoint": bool(restore_best_checkpoint),
    }
    if limitation:
        summary["limitation"] = limitation
    if extra:
        summary.update(extra)
    if not values:
        summary["status"] = "no-loss-history"
        summary["early_stop_reason"] = "no finite monitored loss values were recorded"
        return summary

    best_epoch, best_value, epochs_since_best = _best_epoch(values, min_delta)
    summary.update(
        {
            "best_epoch": best_epoch,
            "best_value": best_value,
            "epochs_since_best": epochs_since_best,
        }
    )
    if summary["final_epoch"] < int(requested_epochs):
        summary["status"] = "early-stopped"
        summary["early_stop_reason"] = "runner stopped before the requested epoch budget"
    elif epochs_since_best >= int(patience):
        summary["status"] = "converged"
        summary["early_stop_reason"] = "monitored loss plateaued within the requested budget"
    else:
        summary["status"] = "max-budget-not-converged"
        summary["early_stop_reason"] = "monitored loss had not plateaued by the requested epoch budget"
    return summary


def write_convergence_artifacts(
    output_dir: Path,
    history: Iterable[dict[str, Any]],
    requested_epochs: int,
    monitor: str = "val_loss",
    patience: int = 5,
    min_delta: float = 0.001,
    restore_best_checkpoint: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in history]
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "loss_history.json"
    history_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = summarize_convergence(
        rows,
        requested_epochs=requested_epochs,
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_checkpoint=restore_best_checkpoint,
        extra=extra,
    )
    summary["loss_history"] = str(history_path)
    summary_path = output_dir / "convergence_summary.json"
    summary["convergence_summary"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _select_monitor(
    rows: list[dict[str, Any]],
    preferred: str,
) -> tuple[str | None, str | None, str | None]:
    if _metric_values(rows, preferred):
        return preferred, "validation" if preferred == "val_loss" else "custom", None
    for metric in TRAINING_MONITORS:
        if _metric_values(rows, metric):
            return (
                metric,
                "training",
                f"validation loss unavailable; using {metric}. This is weaker than validation-based early stopping.",
            )
    return None, None, None


def _metric_values(rows: list[dict[str, Any]], metric: str | None) -> list[tuple[int, float]]:
    if metric is None:
        return []
    values: list[tuple[int, float]] = []
    for index, row in enumerate(rows, start=1):
        value = row.get(metric)
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            values.append((int(row.get("epoch", index)), score))
    return values


def _best_epoch(
    values: list[tuple[int, float]],
    min_delta: float,
) -> tuple[int, float, int]:
    best_epoch, best_value = values[0]
    last_improved_epoch = best_epoch
    for epoch, value in values[1:]:
        relative_target = best_value * (1.0 - float(min_delta))
        absolute_target = best_value - float(min_delta)
        if value < min(relative_target, absolute_target):
            best_epoch = epoch
            best_value = value
            last_improved_epoch = epoch
    final_epoch = values[-1][0]
    return best_epoch, best_value, final_epoch - last_improved_epoch


def _final_epoch(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    try:
        return int(rows[-1].get("epoch", len(rows)))
    except (TypeError, ValueError):
        return len(rows)
