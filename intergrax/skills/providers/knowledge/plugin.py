# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.knowledge.manifests import KNOWLEDGE_OPENAI_STRICT
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class KnowledgeSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="knowledge",
            skill_ids=(KNOWLEDGE_OPENAI_STRICT.skill_id,),
            status=SkillBundleStatus.BETA,
            description="Knowledge retrieval skills (OpenAI hosted and related packs)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (KNOWLEDGE_OPENAI_STRICT,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(KNOWLEDGE_OPENAI_STRICT)
