from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


def write_vt_tiny_mot_dataset_manifest(
    data_root: str | Path,
    output_path: str | Path,
    splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Reuse the system dataset manifest writer without importing the wrong fusiontrack package."""
    module = _load_system_dataset_manifest_module()
    return module.write_dataset_manifest(data_root=data_root, output_path=output_path, splits=splits)


def _load_system_dataset_manifest_module() -> ModuleType:
    code_root = Path(__file__).resolve().parents[3]
    module_path = code_root / "system" / "fusiontrack" / "dataset_manifest.py"
    spec = importlib.util.spec_from_file_location("_fusiontrack_system_dataset_manifest", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset manifest helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
