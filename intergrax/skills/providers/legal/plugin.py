# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.legal.manifests import LEGAL_CONTRACT_REVIEW
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class LegalSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="legal",
            skill_ids=(LEGAL_CONTRACT_REVIEW.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Legal domain skill packs",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (LEGAL_CONTRACT_REVIEW,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(LEGAL_CONTRACT_REVIEW)
