# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.graph.manifests import GRAPH_ENTITY_EXPLORER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class GraphSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="graph",
            skill_ids=(GRAPH_ENTITY_EXPLORER.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Knowledge graph skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (GRAPH_ENTITY_EXPLORER,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(GRAPH_ENTITY_EXPLORER)
