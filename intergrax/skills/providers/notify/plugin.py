# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.notify.manifests import NOTIFY_SCHEDULED_ALERTS
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class NotifySkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="notify",
            skill_ids=(NOTIFY_SCHEDULED_ALERTS.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Scheduled notification skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (NOTIFY_SCHEDULED_ALERTS,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(NOTIFY_SCHEDULED_ALERTS)
