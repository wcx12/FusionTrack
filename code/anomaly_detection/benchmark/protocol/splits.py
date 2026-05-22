from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Sequence

from .schemas import SplitRecord


def collect_sequences_from_observations(csv_path: Path) -> list[str]:
    sequences: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sequence" not in reader.fieldnames:
            raise ValueError(f"CSV is missing required 'sequence' column: {csv_path}")
        for row in reader:
            sequence = row.get("sequence")
            if sequence:
                sequences.add(sequence)
    return sorted(sequences)


def split_sequences(
    sequences: Sequence[str],
    val_fraction: float,
    seed: int,
    test_sequences: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    if not 0 <= val_fraction <= 1:
        raise ValueError("val_fraction must be between 0 and 1")

    sequence_set = set(sequences)
    test = sorted(set(test_sequences or []))
    test_set = set(test)
    unknown_test_sequences = sorted(test_set - sequence_set)
    if unknown_test_sequences:
        raise ValueError(
            f"test_sequences must be a subset of sequences: {unknown_test_sequences}"
        )

    candidates = sorted(sequence for sequence in sequence_set if sequence not in test_set)

    shuffled = candidates[:]
    random.Random(seed).shuffle(shuffled)

    val_count = int(len(shuffled) * val_fraction)
    if val_fraction > 0 and val_count == 0 and len(shuffled) >= 2:
        val_count = 1
    if len(shuffled) >= 2:
        val_count = min(val_count, len(shuffled) - 1)

    val = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])

    return {"train": train, "val": val, "test": test}


def write_split_records(splits: dict[str, list[str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_order = ["train", "val", "test"]
    extra_splits = [name for name in splits if name not in split_order]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "split"])
        writer.writeheader()
        for split in split_order + extra_splits:
            for sequence in splits.get(split, []):
                writer.writerow(SplitRecord(sequence=sequence, split=split).__dict__)


def load_split_records(csv_path: Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        required = {"sequence", "split"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV is missing required columns {sorted(required)}: {csv_path}")
        for row in reader:
            split = row["split"]
            splits.setdefault(split, []).append(row["sequence"])
    return splits
