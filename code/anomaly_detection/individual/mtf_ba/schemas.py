from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def build_sample_id(sequence: str, track_id: str | int) -> str:
    """Build the shared object-level sample ID used across the pipeline."""
    return f"{sequence}:{track_id}"


@dataclass(frozen=True)
class ObjectIdentity:
    """Stable object identity shared across individual/group/fusion stages."""

    sequence: str
    track_id: str
    category_id: int | None = None
    category_name: str | None = None

    @property
    def sample_id(self) -> str:
        return build_sample_id(self.sequence, self.track_id)


@dataclass
class ScoreRecord:
    """
    Standard anomaly score record.

    `source` is expected to be one of:
    - "individual"
    - "group"
    - "fusion"
    """

    sequence: str
    track_id: str
    source: str
    score: float
    category_id: int | None = None
    category_name: str | None = None
    component_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        return build_sample_id(self.sequence, self.track_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sequence": self.sequence,
            "track_id": self.track_id,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "source": self.source,
            "score": self.score,
            "component_scores": dict(self.component_scores),
            "metadata": dict(self.metadata),
        }
