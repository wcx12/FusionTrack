from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROFILE_FIELDS = (
    "name",
    "task",
    "owner",
    "role",
    "method_family",
    "learning_type",
    "source_type",
    "status",
)

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "anomaly_detection"
    / "benchmark"
    / "configs"
    / "method_registry.json"
)


def method_profile(name: str, task: str, registry_path: str | Path | None = None) -> dict[str, str]:
    registry = _load_registry(Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH)
    normalized_name = str(name or "").strip()
    normalized_task = str(task or "").strip()
    key = (normalized_task, normalized_name)
    if key in registry:
        return dict(registry[key])
    for profile in registry.values():
        aliases = profile.get("aliases", ())
        if profile.get("task") == normalized_task and normalized_name in aliases:
            return {field: str(profile.get(field, "")) for field in PROFILE_FIELDS}
    return {
        "name": normalized_name,
        "task": normalized_task,
        "owner": "unregistered",
        "role": "unregistered",
        "method_family": "unregistered",
        "learning_type": "unknown",
        "source_type": "unknown",
        "status": "unregistered",
    }


def _load_registry(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods") if isinstance(payload, dict) else None
    if not isinstance(methods, list):
        return {}
    registry: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_profile in methods:
        if not isinstance(raw_profile, dict):
            continue
        name = str(raw_profile.get("name", "")).strip()
        task = str(raw_profile.get("task", "")).strip()
        if not name or not task:
            continue
        profile = {field: str(raw_profile.get(field, "") or "").strip() for field in PROFILE_FIELDS}
        profile["name"] = name
        profile["task"] = task
        profile["aliases"] = tuple(str(alias).strip() for alias in raw_profile.get("aliases", []) or [])
        registry[(task, name)] = profile
    return registry
