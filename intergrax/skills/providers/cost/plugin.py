# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.cost.manifests import (
    COST_BUDGET_GUARDIAN,
    COST_CHARGEBACK_REPORT,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_COST_MANIFESTS = (
    COST_BUDGET_GUARDIAN,
    COST_CHARGEBACK_REPORT,
)


class CostSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="cost",
            skill_ids=tuple(m.skill_id for m in _COST_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="cost skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _COST_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _COST_MANIFESTS:
            registry.register(manifest)
