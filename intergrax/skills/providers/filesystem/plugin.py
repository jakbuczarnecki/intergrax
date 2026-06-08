# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.filesystem.manifests import (
    FILESYSTEM_LOCAL_IO,
    FILESYSTEM_STAT_AUDITOR,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_FILESYSTEM_MANIFESTS = (
    FILESYSTEM_LOCAL_IO,
    FILESYSTEM_STAT_AUDITOR,
)


class FilesystemSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="filesystem",
            skill_ids=tuple(m.skill_id for m in _FILESYSTEM_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="filesystem skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _FILESYSTEM_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _FILESYSTEM_MANIFESTS:
            registry.register(manifest)
