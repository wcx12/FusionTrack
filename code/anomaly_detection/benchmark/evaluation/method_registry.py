from __future__ import annotations

import json
from dataclasses import dataclass
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
    Path(__file__).resolve().parents[1] / "configs" / "method_registry.json"
)


@dataclass(frozen=True)
class MethodRegistry:
    schema_version: int
    source_path: Path
    profiles: dict[tuple[str, str], dict[str, str]]
    aliases: dict[tuple[str, str], tuple[str, str]]

    def profile_for(self, name: str, task: str | None = None) -> dict[str, str]:
        normalized_name = _norm(name)
        normalized_task = _norm(task) if task else None
        keys: list[tuple[str, str]] = []
        if normalized_task:
            keys.append((normalized_task, normalized_name))
            alias = self.aliases.get((normalized_task, normalized_name))
            if alias is not None:
                keys.append(alias)
        else:
            keys.extend(key for key in self.profiles if key[1] == normalized_name)
            keys.extend(
                target for (alias_task, alias_name), target in self.aliases.items()
                if alias_name == normalized_name
            )

        unique_keys = []
        for key in keys:
            if key in self.profiles and key not in unique_keys:
                unique_keys.append(key)
        if len(unique_keys) == 1:
            return dict(self.profiles[unique_keys[0]])
        if len(unique_keys) > 1:
            raise ValueError(f"Method {name!r} is ambiguous; pass task explicitly.")
        return _unknown_profile(normalized_name, normalized_task)

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": _portable_path(self.source_path),
        }


def load_method_registry(path: str | Path | None = None) -> MethodRegistry:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{registry_path} must contain a JSON object.")
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError(f"{registry_path} must contain a methods list.")

    profiles: dict[tuple[str, str], dict[str, str]] = {}
    aliases: dict[tuple[str, str], tuple[str, str]] = {}
    for index, raw_profile in enumerate(methods, start=1):
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Method registry entry {index} must be a JSON object.")
        profile = _normalize_profile(raw_profile)
        key = (profile["task"], profile["name"])
        if key in profiles:
            raise ValueError(f"Duplicate method registry entry: {key[0]}/{key[1]}")
        profiles[key] = profile
        for alias in raw_profile.get("aliases", []) or []:
            alias_key = (profile["task"], _norm(alias))
            if alias_key in aliases:
                raise ValueError(
                    f"Duplicate method registry alias: {alias_key[0]}/{alias_key[1]}"
                )
            aliases[alias_key] = key

    return MethodRegistry(
        schema_version=int(payload.get("schema_version", 1)),
        source_path=registry_path,
        profiles=profiles,
        aliases=aliases,
    )


def _normalize_profile(raw_profile: dict[str, Any]) -> dict[str, str]:
    name = _norm(raw_profile.get("name"))
    task = _norm(raw_profile.get("task"))
    if not name:
        raise ValueError("Method registry entry is missing name.")
    if not task:
        raise ValueError(f"Method registry entry {name!r} is missing task.")

    profile = {field: _norm(raw_profile.get(field)) for field in PROFILE_FIELDS}
    profile["name"] = name
    profile["task"] = task
    profile["owner"] = profile["owner"] or "unclassified"
    profile["role"] = profile["role"] or "unclassified"
    profile["method_family"] = profile["method_family"] or "unclassified"
    profile["learning_type"] = profile["learning_type"] or "unclassified"
    profile["source_type"] = profile["source_type"] or "unclassified"
    profile["status"] = profile["status"] or "planned"
    return profile


def _unknown_profile(name: str, task: str | None) -> dict[str, str]:
    return {
        "name": name,
        "task": task or "unknown",
        "owner": "unregistered",
        "role": "unregistered",
        "method_family": "unregistered",
        "learning_type": "unknown",
        "source_type": "unknown",
        "status": "unregistered",
    }


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()
