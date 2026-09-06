# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only catalog manifest projection for discovery adapters."""

from __future__ import annotations

from typing import Iterator

from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.shipped_plugins import SHIPPED_SKILL_PLUGINS

_CATALOG_MANIFESTS: dict[str, SkillManifest] | None = None


def _normalize_plugin_manifests(raw: object, plugin_type: type) -> tuple[SkillManifest, ...]:
    if isinstance(raw, SkillManifest):
        return (raw,)
    if isinstance(raw, tuple):
        manifests: list[SkillManifest] = []
        for item in raw:
            if isinstance(item, SkillManifest):
                manifests.append(item)
            else:
                raise TypeError(
                    f"{plugin_type.__qualname__}.skill_manifests() must yield SkillManifest",
                )
        return tuple(manifests)
    raise TypeError(
        f"{plugin_type.__qualname__}.skill_manifests() must return SkillManifest or tuple thereof",
    )


def _build_catalog_manifest_index() -> dict[str, SkillManifest]:
    index: dict[str, SkillManifest] = {}
    for plugin_type in SHIPPED_SKILL_PLUGINS:
        for manifest in _normalize_plugin_manifests(plugin_type.skill_manifests(), plugin_type):
            skill_id = manifest.skill_id
            if skill_id in index and index[skill_id].version != manifest.version:
                raise ValueError(
                    f"Conflicting catalog manifest versions for skill '{skill_id}'",
                )
            index[skill_id] = manifest
    return index


def catalog_manifest_for_skill_id(skill_id: str) -> SkillManifest | None:
    """Return the shipped catalog manifest for ``skill_id``, if present."""
    global _CATALOG_MANIFESTS
    if _CATALOG_MANIFESTS is None:
        _CATALOG_MANIFESTS = _build_catalog_manifest_index()
    return _CATALOG_MANIFESTS.get(skill_id)


def iter_catalog_skill_manifests() -> Iterator[SkillManifest]:
    """Iterate shipped catalog manifests in deterministic skill_id order."""
    global _CATALOG_MANIFESTS
    if _CATALOG_MANIFESTS is None:
        _CATALOG_MANIFESTS = _build_catalog_manifest_index()
    for skill_id in sorted(_CATALOG_MANIFESTS):
        yield _CATALOG_MANIFESTS[skill_id]


def reset_catalog_manifest_index_for_tests() -> None:
    """Clear cached catalog manifest index (tests only)."""
    global _CATALOG_MANIFESTS
    _CATALOG_MANIFESTS = None
