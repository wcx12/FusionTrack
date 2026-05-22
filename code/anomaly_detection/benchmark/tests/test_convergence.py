from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.convergence import summarize_convergence, write_convergence_artifacts


def test_summarize_convergence_uses_validation_plateau() -> None:
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.0},
        {"epoch": 2, "train_loss": 0.8, "val_loss": 0.8},
        {"epoch": 3, "train_loss": 0.7, "val_loss": 0.8005},
        {"epoch": 4, "train_loss": 0.6, "val_loss": 0.8004},
    ]

    summary = summarize_convergence(
        history,
        requested_epochs=4,
        monitor="val_loss",
        patience=2,
        min_delta=0.001,
    )

    assert summary["status"] == "converged"
    assert summary["monitor"] == "val_loss"
    assert summary["best_epoch"] == 2
    assert summary["final_epoch"] == 4
    assert summary["epochs_since_best"] == 2
    assert summary["restore_best_checkpoint"] is True


def test_summarize_convergence_flags_unfinished_improvement() -> None:
    history = [
        {"epoch": 1, "loss": 1.0},
        {"epoch": 2, "loss": 0.8},
        {"epoch": 3, "loss": 0.6},
    ]

    summary = summarize_convergence(
        history,
        requested_epochs=3,
        monitor="val_loss",
        patience=2,
        min_delta=0.001,
    )

    assert summary["status"] == "max-budget-not-converged"
    assert summary["monitor"] == "loss"
    assert summary["monitor_source"] == "training"
    assert "validation loss unavailable" in summary["limitation"]


def test_write_convergence_artifacts_writes_history_and_summary(tmp_path: Path) -> None:
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.0},
        {"epoch": 2, "train_loss": 0.9, "val_loss": 0.9},
        {"epoch": 3, "train_loss": 0.8, "val_loss": 0.9001},
    ]

    summary = write_convergence_artifacts(
        tmp_path,
        history,
        requested_epochs=3,
        monitor="val_loss",
        patience=1,
        min_delta=0.001,
        extra={"gpu_name": "unit-test-gpu"},
    )

    assert summary["status"] == "converged"
    assert summary["gpu_name"] == "unit-test-gpu"
    assert json.loads((tmp_path / "loss_history.json").read_text(encoding="utf-8")) == history
    saved_summary = json.loads(
        (tmp_path / "convergence_summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary == summary
