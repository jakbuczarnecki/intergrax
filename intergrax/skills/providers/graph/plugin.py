# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.graph.manifests import (
    GRAPH_ENTITY_EXPLORER,
    GRAPH_PATH_FINDER,
    GRAPH_KNOWLEDGE_LINKER,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_GRAPH_MANIFESTS = (
    GRAPH_ENTITY_EXPLORER,
    GRAPH_PATH_FINDER,
    GRAPH_KNOWLEDGE_LINKER,
)


class GraphSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="graph",
            skill_ids=tuple(m.skill_id for m in _GRAPH_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="graph skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _GRAPH_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _GRAPH_MANIFESTS:
            registry.register(manifest)
