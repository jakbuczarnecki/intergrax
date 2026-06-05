# © Artur Czarnecki. All rights reserved.

"""Declarative environment loaders (Phase DX-5.3–DX-5.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_ENV_ADAPTER = TypeAdapter(ApplicationEnvironmentProfile)
_MANIFEST_ADAPTER = TypeAdapter(ApplicationManifest)


def load_environment_profile_yaml(path: Path) -> ApplicationEnvironmentProfile:
    """Load ``ApplicationEnvironmentProfile`` from YAML or JSON."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML environment files")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Environment file must be a mapping: {path}")
    return _ENV_ADAPTER.validate_python(payload)


def load_manifest_yaml(path: Path) -> ApplicationManifest:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML manifest files")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest file must be a mapping: {path}")
    return _MANIFEST_ADAPTER.validate_python(payload)


def load_agents_yaml(path: Path) -> list[AgentBinding]:
    """
    Load declarative roster entries (``agents.yaml``).

    Each item must include ``import_path`` and optional ``capabilities`` list.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML agent rosters")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"Agents file must be a list: {path}")
    bindings: list[AgentBinding] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid agents.yaml entry: {item!r}")
        import_path = item.get("import_path")
        if not isinstance(import_path, str):
            raise ValueError("agents.yaml entry requires import_path")
        caps_raw = item.get("capabilities", [])
        capabilities = [c for c in caps_raw if isinstance(c, str)] if isinstance(caps_raw, list) else []
        bindings.append(
            AgentBinding.deserialize(
                import_path=import_path,
                capabilities=capabilities,
                default=bool(item.get("default", False)),
            )
        )
    return bindings


def merge_manifest_with_files(
    manifest: ApplicationManifest,
    *,
    env_path: Path | None = None,
    agents_path: Path | None = None,
) -> ApplicationManifest:
    updates: dict[str, Any] = {}
    if env_path is not None:
        updates["environment"] = load_environment_profile_yaml(env_path)
    if agents_path is not None:
        updates["agents"] = load_agents_yaml(agents_path)
    if not updates:
        return manifest
    return manifest.model_copy(update=updates)
