# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.data.manifests import DATA_SQL_ANALYST
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class DataSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="data",
            skill_ids=(DATA_SQL_ANALYST.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Data analyst skill packs (SK-EXP P2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (DATA_SQL_ANALYST,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(DATA_SQL_ANALYST)
