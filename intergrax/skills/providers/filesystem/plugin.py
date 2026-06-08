# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.filesystem.manifests import FILESYSTEM_LOCAL_IO
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class FilesystemSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="filesystem",
            skill_ids=(FILESYSTEM_LOCAL_IO.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Filesystem skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (FILESYSTEM_LOCAL_IO)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(FILESYSTEM_LOCAL_IO)
