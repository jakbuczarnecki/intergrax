# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.notify.manifests import (
    NOTIFY_SCHEDULED_ALERTS,
    NOTIFY_BATCH_DISPATCH,
    NOTIFY_ESCALATION_LADDER,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_NOTIFY_MANIFESTS = (
    NOTIFY_SCHEDULED_ALERTS,
    NOTIFY_BATCH_DISPATCH,
    NOTIFY_ESCALATION_LADDER,
)


class NotifySkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="notify",
            skill_ids=tuple(m.skill_id for m in _NOTIFY_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="notify skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _NOTIFY_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _NOTIFY_MANIFESTS:
            registry.register(manifest)
