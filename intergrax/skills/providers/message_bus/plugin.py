# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.message_bus.manifests import (
    MESSAGE_BUS_ASYNC_RUNNER,
    MESSAGE_BUS_TASK_ADMIN,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class MessageBusSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="message_bus",
            skill_ids=(MESSAGE_BUS_ASYNC_RUNNER.skill_id, MESSAGE_BUS_TASK_ADMIN.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Message bus skill packs (SK-EXP2 + SK-EXP3)",
        )

    _MANIFESTS = (MESSAGE_BUS_ASYNC_RUNNER, MESSAGE_BUS_TASK_ADMIN)

    @classmethod
    def skill_manifests(cls) -> tuple:
        return MessageBusSkillPlugin._MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in MessageBusSkillPlugin._MANIFESTS:
            registry.register(manifest)
