from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SplitRecord:
    sequence: str
    split: str


@dataclass
class AnomalyLabel:
    sample_id: str
    sequence: str
    track_id: str
    frame_start: int
    frame_end: int
    label: int
    anomaly_type: str
    injection_seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sample_id(sequence: str, track_id: str) -> str:
    return f"{sequence}:{track_id}"
