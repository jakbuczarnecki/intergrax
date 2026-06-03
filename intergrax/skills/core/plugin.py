# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill plugin protocol — external skill bundles (§7.1.8)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.registry.runtime import SkillRegistry


@runtime_checkable
class SkillPlugin(Protocol):
    """
    Optional class-based registration for custom skill bundles.

    Entry point group: ``intergrax.skills``.
    """

    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        """Catalog identity for this skill bundle."""

    @classmethod
    def skill_manifests(cls) -> tuple:
        """Skill manifests provided by this bundle (``SkillManifest`` instances)."""

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        """Register skill manifests on ``registry``."""


def skill_bundle_manifest_for_plugin(plugin_type: type[SkillPlugin]) -> SkillBundleManifest:
    manifest = plugin_type.skill_bundle_manifest()
    if not isinstance(manifest, SkillBundleManifest):
        raise TypeError(f"{plugin_type.__qualname__}.skill_bundle_manifest() must return SkillBundleManifest")
    return manifest
