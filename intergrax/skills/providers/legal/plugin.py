# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.legal.manifests import (
    LEGAL_CASE_RESEARCH,
    LEGAL_CLAUSE_COMPARE,
    LEGAL_CONTRACT_REVIEW,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_LEGAL_MANIFESTS = (
    LEGAL_CONTRACT_REVIEW,
    LEGAL_CLAUSE_COMPARE,
    LEGAL_CASE_RESEARCH,
)


class LegalSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="legal",
            skill_ids=tuple(m.skill_id for m in _LEGAL_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Legal domain skill packs",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _LEGAL_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _LEGAL_MANIFESTS:
            registry.register(manifest)
