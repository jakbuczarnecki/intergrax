# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.rag.manifests import RAG_DOCUMENT_INGEST, RAG_HYBRID_QA
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_RAG_MANIFESTS = (RAG_HYBRID_QA, RAG_DOCUMENT_INGEST)


class RagSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="rag",
            skill_ids=tuple(m.skill_id for m in _RAG_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="RAG retrieval and document ingest skill packs (SK-EXP P0)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _RAG_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _RAG_MANIFESTS:
            registry.register(manifest)
