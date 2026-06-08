# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.cost.manifests import COST_BUDGET_GUARDIAN
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class CostSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="cost",
            skill_ids=(COST_BUDGET_GUARDIAN.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Cost and budget governance skill packs (SK-EXP3)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (COST_BUDGET_GUARDIAN,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(COST_BUDGET_GUARDIAN)
