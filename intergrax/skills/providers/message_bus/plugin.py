# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.message_bus.manifests import (
    MESSAGE_BUS_ASYNC_RUNNER,
    MESSAGE_BUS_TASK_ADMIN,
    MESSAGE_BUS_RETRY_HANDLER,
    MESSAGE_BUS_DEAD_LETTER,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_MESSAGE_BUS_MANIFESTS = (
    MESSAGE_BUS_ASYNC_RUNNER,
    MESSAGE_BUS_TASK_ADMIN,
    MESSAGE_BUS_RETRY_HANDLER,
    MESSAGE_BUS_DEAD_LETTER,
)


class MessageBusSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="message_bus",
            skill_ids=tuple(m.skill_id for m in _MESSAGE_BUS_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="message_bus skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _MESSAGE_BUS_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _MESSAGE_BUS_MANIFESTS:
            registry.register(manifest)
