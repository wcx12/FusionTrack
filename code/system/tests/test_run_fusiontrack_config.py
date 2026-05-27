from __future__ import annotations

import json
from pathlib import Path

from run_fusiontrack import parse_args


def test_run_config_resolves_relative_paths_against_base_dir(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "dashboard.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "base_dir": "..",
                "data_root": "data/VT-Tiny-MOT",
                "work_root": "runs/final_results_dashboard",
                "final_results_root": "server_artifacts/final_results",
                "individual_label_file": "labels/individual.jsonl",
                "group_label_file": "labels/group.jsonl",
                "fused_jsonl": "runs/final/fused.jsonl",
                "score_search_roots": ["scores/a", "scores/b"],
                "top_sequences": 7,
                "top_k": 100,
                "case_limit": 12,
            }
        ),
        encoding="utf-8",
    )

    args = parse_args(["--run-config", str(config)])

    assert args.data_root == tmp_path / "data" / "VT-Tiny-MOT"
    assert args.work_root == tmp_path / "runs" / "final_results_dashboard"
    assert args.final_results_root == tmp_path / "server_artifacts" / "final_results"
    assert args.individual_label_file == tmp_path / "labels" / "individual.jsonl"
    assert args.group_label_file == tmp_path / "labels" / "group.jsonl"
    assert args.fused_jsonl == tmp_path / "runs" / "final" / "fused.jsonl"
    assert args.score_search_root == [tmp_path / "scores" / "a", tmp_path / "scores" / "b"]
    assert args.top_sequences == 7
    assert args.top_k == 100
    assert args.case_limit == 12


def test_run_config_allows_cli_overrides(tmp_path: Path) -> None:
    config = tmp_path / "dashboard.json"
    config.write_text(
        json.dumps(
            {
                "work_root": "runs/from_config",
                "score_search_roots": ["scores/from_config"],
                "top_k": 20,
            }
        ),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--run-config",
            str(config),
            "--work-root",
            str(tmp_path / "runs" / "from_cli"),
            "--score-search-root",
            str(tmp_path / "scores" / "from_cli"),
            "--top-k",
            "5",
        ]
    )

    assert args.work_root == tmp_path / "runs" / "from_cli"
    assert args.score_search_root == [tmp_path / "scores" / "from_cli"]
    assert args.top_k == 5
