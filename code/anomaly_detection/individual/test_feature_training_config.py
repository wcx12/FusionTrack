from __future__ import annotations

import ast
from pathlib import Path


def test_fused_feature_names_use_expected_columns_layers_and_lengths() -> None:
    constants = _module_constants()
    expected = {
        "route_fused": (["latitude", "longitude"], 2, 10),
        "speed_fused": (["speed"], 1, 10),
        "shape_fused": (["delta_x", "delta_y"], 2, 2),
    }

    for feature_name, (columns, num_layers, min_length) in expected.items():
        assert constants["FEATURE_COLUMNS"][feature_name] == columns
        assert constants["FEATURE_NUM_LAYERS"][feature_name] == num_layers
        assert constants["FEATURE_MIN_LENGTH_DEFAULTS"][feature_name] == min_length


def _module_constants() -> dict[str, dict]:
    module_path = Path(__file__).parent / "mtf_ba" / "feature_training.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "FEATURE_COLUMNS",
                    "FEATURE_NUM_LAYERS",
                    "FEATURE_MIN_LENGTH_DEFAULTS",
                }:
                    constants[target.id] = ast.literal_eval(node.value)
    return constants
