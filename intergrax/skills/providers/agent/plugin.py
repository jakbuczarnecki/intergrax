# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.agent.manifests import (
    AGENT_ROSTER_INTROSPECT,
    AGENT_CAPABILITY_MAPPER,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_AGENT_MANIFESTS = (
    AGENT_ROSTER_INTROSPECT,
    AGENT_CAPABILITY_MAPPER,
)


class AgentSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="agent",
            skill_ids=tuple(m.skill_id for m in _AGENT_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="agent skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _AGENT_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _AGENT_MANIFESTS:
            registry.register(manifest)
