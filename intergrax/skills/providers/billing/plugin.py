# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.billing.manifests import BILLING_USAGE_TRACKER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class BillingSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="billing",
            skill_ids=(BILLING_USAGE_TRACKER.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Billing and usage tracking skill packs (SK-EXP3)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (BILLING_USAGE_TRACKER,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(BILLING_USAGE_TRACKER)
