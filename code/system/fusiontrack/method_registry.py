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


def method_profile(name: str, task: str, registry_path: str | Path | None = None) -> dict[str, Any]:
    registry = _load_registry(Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH)
    normalized_name = str(name or "").strip()
    normalized_task = str(task or "").strip()
    key = (normalized_task, normalized_name)
    if key in registry:
        profile = dict(registry[key])
        profile["registry_status"] = "registered"
        return profile
    for profile in registry.values():
        aliases = profile.get("aliases", ())
        if profile.get("task") == normalized_task and normalized_name in aliases:
            matched = {field: str(profile.get(field, "")) for field in PROFILE_FIELDS}
            matched["aliases"] = tuple(str(alias) for alias in aliases)
            matched["registry_status"] = "registered"
            return matched
    return {
        "name": normalized_name,
        "task": normalized_task,
        "owner": "unregistered",
        "role": "unregistered",
        "method_family": "unregistered",
        "learning_type": "unknown",
        "source_type": "unknown",
        "status": "unregistered",
        "aliases": (),
        "registry_status": "unregistered",
    }


def validate_method_registry(registry_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return {
            "status": "invalid",
            "num_methods": 0,
            "num_aliases": 0,
            "errors": [f"registry file does not exist: {path}"],
            "warnings": warnings,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods") if isinstance(payload, dict) else None
    if not isinstance(methods, list):
        return {
            "status": "invalid",
            "num_methods": 0,
            "num_aliases": 0,
            "errors": ["registry payload must contain a methods list"],
            "warnings": warnings,
        }

    method_keys: dict[tuple[str, str], int] = {}
    method_names_by_task: dict[str, set[str]] = {}
    aliases_by_task: dict[tuple[str, str], str] = {}
    alias_count = 0

    for index, raw_profile in enumerate(methods):
        if not isinstance(raw_profile, dict):
            errors.append(f"method[{index}] must be an object")
            continue
        name = str(raw_profile.get("name", "") or "").strip()
        task = str(raw_profile.get("task", "") or "").strip()
        for field in PROFILE_FIELDS:
            if str(raw_profile.get(field, "") or "").strip() == "":
                errors.append(f"method[{index}] missing required field {field}")
        if task and name:
            key = (task, name)
            if key in method_keys:
                errors.append(f"duplicate method {task}:{name} at method[{index}] and method[{method_keys[key]}]")
            method_keys[key] = index
            method_names_by_task.setdefault(task, set()).add(name)

    for index, raw_profile in enumerate(methods):
        if not isinstance(raw_profile, dict):
            continue
        name = str(raw_profile.get("name", "") or "").strip()
        task = str(raw_profile.get("task", "") or "").strip()
        aliases = raw_profile.get("aliases", []) or []
        if not isinstance(aliases, list):
            errors.append(f"method[{index}] aliases must be a list")
            continue
        for alias in aliases:
            normalized_alias = str(alias or "").strip()
            if not normalized_alias:
                errors.append(f"method[{index}] contains an empty alias")
                continue
            alias_count += 1
            owner = f"{task}:{name}"
            alias_key = (task, normalized_alias)
            previous_owner = aliases_by_task.get(alias_key)
            if previous_owner is not None and previous_owner != owner:
                errors.append(
                    f"duplicate alias {task}:{normalized_alias} used by {previous_owner} and {owner}"
                )
            aliases_by_task[alias_key] = owner
            if task and normalized_alias in method_names_by_task.get(task, set()) and normalized_alias != name:
                errors.append(
                    f"alias {task}:{normalized_alias} collides with a registered method name"
                )

    return {
        "status": "invalid" if errors else "ok",
        "num_methods": len(methods),
        "num_aliases": alias_count,
        "errors": errors,
        "warnings": warnings,
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
