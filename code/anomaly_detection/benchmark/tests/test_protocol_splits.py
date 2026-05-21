from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol.splits import (
    collect_sequences_from_observations,
    load_split_records,
    split_sequences,
    write_split_records,
)


def test_collect_sequences_returns_sorted_unique_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "observations.csv"
    csv_path.write_text(
        "sequence,track_id,frame_id\n"
        "seq_b,1,0\n"
        "seq_a,2,0\n"
        "seq_b,1,1\n",
        encoding="utf-8",
    )

    assert collect_sequences_from_observations(csv_path) == ["seq_a", "seq_b"]


def test_split_sequences_is_deterministic_and_excludes_test_sequences() -> None:
    sequences = ["seq_1", "seq_2", "seq_3", "seq_4", "seq_5"]

    first = split_sequences(
        sequences,
        val_fraction=0.2,
        seed=11,
        test_sequences=["seq_5"],
    )
    second = split_sequences(
        sequences,
        val_fraction=0.2,
        seed=11,
        test_sequences=["seq_5"],
    )

    assert first == second
    assert first["test"] == ["seq_5"]
    assert "seq_5" not in first["train"]
    assert "seq_5" not in first["val"]
    assert len(first["val"]) == 1
    assert sorted(first["train"] + first["val"] + first["test"]) == sorted(sequences)


def test_split_sequences_rejects_unknown_test_sequences() -> None:
    with pytest.raises(ValueError, match="test_sequences"):
        split_sequences(
            ["seq_1", "seq_2"],
            val_fraction=0.5,
            seed=11,
            test_sequences=["seq_missing"],
        )


def test_write_and_load_split_records_roundtrip(tmp_path: Path) -> None:
    splits = {"train": ["seq_1", "seq_3"], "val": ["seq_2"], "test": ["seq_4"]}
    output_csv = tmp_path / "splits.csv"

    write_split_records(splits, output_csv)

    assert load_split_records(output_csv) == splits
