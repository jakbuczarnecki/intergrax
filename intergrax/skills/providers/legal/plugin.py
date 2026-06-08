# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.legal.manifests import (
    LEGAL_CONTRACT_REVIEW,
    LEGAL_CLAUSE_COMPARE,
    LEGAL_CASE_RESEARCH,
    LEGAL_REDLINE_DRAFT,
    LEGAL_REGULATORY_SCAN,
    LEGAL_OBLIGATION_TRACKER,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_LEGAL_MANIFESTS = (
    LEGAL_CONTRACT_REVIEW,
    LEGAL_CLAUSE_COMPARE,
    LEGAL_CASE_RESEARCH,
    LEGAL_REDLINE_DRAFT,
    LEGAL_REGULATORY_SCAN,
    LEGAL_OBLIGATION_TRACKER,
)


class LegalSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="legal",
            skill_ids=tuple(m.skill_id for m in _LEGAL_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="legal skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _LEGAL_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _LEGAL_MANIFESTS:
            registry.register(manifest)
