# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.agent.manifests import AGENT_ROSTER_INTROSPECT
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class AgentSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="agent",
            skill_ids=(AGENT_ROSTER_INTROSPECT.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Agent roster introspection skill packs (SK-EXP3)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (AGENT_ROSTER_INTROSPECT,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(AGENT_ROSTER_INTROSPECT)
