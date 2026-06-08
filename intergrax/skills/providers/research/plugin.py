# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.research.manifests import (
    RESEARCH_CITATION_SYNTHESIS,
    RESEARCH_LITERATURE_SCAN,
    RESEARCH_WEB_EVIDENCE,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_RESEARCH_MANIFESTS = (
    RESEARCH_LITERATURE_SCAN,
    RESEARCH_WEB_EVIDENCE,
    RESEARCH_CITATION_SYNTHESIS,
)


class ResearchSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="research",
            skill_ids=tuple(m.skill_id for m in _RESEARCH_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Research domain skill packs",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _RESEARCH_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _RESEARCH_MANIFESTS:
            registry.register(manifest)
