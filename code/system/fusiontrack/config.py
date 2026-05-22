from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FusionTrackPaths:
    data_root: Path
    work_root: Path
    trajectory_dir: Path
    fusion_dir: Path
    feature_dir: Path
    model_dir: Path
    score_dir: Path
    group_dir: Path
    final_dir: Path
    heatmap_dir: Path
    report_dir: Path

    @classmethod
    def defaults(
        cls,
        data_root: str | Path = Path("data") / "VT-Tiny-MOT",
        work_root: str | Path = Path("runs") / "fusiontrack_v1",
    ) -> "FusionTrackPaths":
        work_root = Path(work_root)
        return cls(
            data_root=Path(data_root),
            work_root=work_root,
            trajectory_dir=work_root / "trajectories",
            fusion_dir=work_root / "fusion",
            feature_dir=work_root / "features",
            model_dir=work_root / "models",
            score_dir=work_root / "scores",
            group_dir=work_root / "group",
            final_dir=work_root / "final",
            heatmap_dir=work_root / "heatmaps",
            report_dir=work_root / "report",
        )

    def observations_csv(self, split: str) -> Path:
        return self.trajectory_dir / f"observations_{split}.csv"

    def fused_jsonl(self, split: str) -> Path:
        return self.fusion_dir / f"fused_trajectories_{split}.jsonl"

    def fused_states_csv(self, split: str) -> Path:
        return self.fusion_dir / f"fused_states_{split}.csv"
