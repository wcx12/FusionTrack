#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MTF_BA_ROOT = REPO_ROOT / "code" / "anomaly_detection" / "individual"
if str(MTF_BA_ROOT) not in sys.path:
    sys.path.insert(0, str(MTF_BA_ROOT))

from fusiontrack.config import FusionTrackPaths
from fusiontrack.pipeline import run_smoke_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FusionTrack v1 offline pipeline.")
    parser.add_argument("--data-root", type=Path, default=Path("data") / "VT-Tiny-MOT")
    parser.add_argument("--work-root", type=Path, default=Path("runs") / "fusiontrack_v1")
    parser.add_argument("--split", choices=["train", "test", "val"], default="test")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--top-sequences", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--device", default="cpu", help="Reserved for full training mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = FusionTrackPaths.defaults(data_root=args.data_root, work_root=args.work_root)
    if args.mode == "full":
        print("full mode is using the smoke baseline path in this v1 implementation.")
    summary = run_smoke_pipeline(
        paths=paths,
        split=args.split,
        top_sequences=args.top_sequences,
        force=args.force,
        skip_extraction=args.skip_extraction,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
