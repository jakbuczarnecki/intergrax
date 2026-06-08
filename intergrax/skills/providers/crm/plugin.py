# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.crm.manifests import CRM_ACCOUNT_LOOKUP
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class CrmSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="crm",
            skill_ids=(CRM_ACCOUNT_LOOKUP.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="CRM account lookup skill packs (SK-EXP3)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (CRM_ACCOUNT_LOOKUP,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(CRM_ACCOUNT_LOOKUP)
