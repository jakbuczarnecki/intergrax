# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.knowledge.manifests import KNOWLEDGE_OPENAI_STRICT, KNOWLEDGE_WIKI_NAVIGATOR
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_KNOWLEDGE_MANIFESTS = (KNOWLEDGE_OPENAI_STRICT, KNOWLEDGE_WIKI_NAVIGATOR)


class KnowledgeSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="knowledge",
            skill_ids=tuple(m.skill_id for m in _KNOWLEDGE_MANIFESTS),
            status=SkillBundleStatus.BETA,
            description="Knowledge retrieval skills (OpenAI hosted, wiki, and related packs)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _KNOWLEDGE_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _KNOWLEDGE_MANIFESTS:
            registry.register(manifest)
