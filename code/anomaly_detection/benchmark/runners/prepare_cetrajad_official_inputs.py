from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.io import load_jsonl
from external_sources.cetrajad_adapters import write_cetrajad_official_input_bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare benchmark trajectory JSONL for an external CETrajAD checkout. "
            "The bundle includes a manifest and <dict-name>_evaluation_gps.pkl; "
            "use those paths as the official script data path according to the "
            "checked-out CETrajAD parameters."
        )
    )
    parser.add_argument("--trajectory-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dict-name", default="fusiontrack")
    parser.add_argument("--sidecar-name", default="cetrajad_sidecar.json")
    parser.add_argument("--manifest-name", default="cetrajad_manifest.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = write_cetrajad_official_input_bundle(
        load_jsonl(args.trajectory_jsonl),
        output_dir=args.output_dir,
        dict_name=args.dict_name,
        sidecar_name=args.sidecar_name,
        manifest_name=args.manifest_name,
    )
    print(
        json.dumps(
            {
                "trajectory_jsonl": str(args.trajectory_jsonl),
                "output_dir": str(args.output_dir),
                "manifest_json": str(args.output_dir / args.manifest_name),
                "sidecar_json": manifest["sidecar_json"],
                "input_pickle": manifest["input_pickle"],
                "num_trajectories": manifest["num_trajectories"],
                "dict_name": manifest["dict_name"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
