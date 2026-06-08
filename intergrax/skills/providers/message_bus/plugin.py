# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.message_bus.manifests import MESSAGE_BUS_ASYNC_RUNNER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class MessageBusSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="message_bus",
            skill_ids=(MESSAGE_BUS_ASYNC_RUNNER.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Message bus async task skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (MESSAGE_BUS_ASYNC_RUNNER,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(MESSAGE_BUS_ASYNC_RUNNER)
