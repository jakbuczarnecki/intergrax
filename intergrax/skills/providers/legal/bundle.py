# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.legal.manifests import LEGAL_CONTRACT_REVIEW
from intergrax.skills.registry.catalog import SkillBundleEntry, SkillBundleStatus, register_skill_bundle
from intergrax.skills.registry.runtime import SkillRegistry


def _register_legal_skills(registry: SkillRegistry) -> None:
    registry.register(LEGAL_CONTRACT_REVIEW)


def register_legal_skill_bundle(*, override: bool = False) -> None:
    register_skill_bundle(
        SkillBundleEntry(
            bundle_id="legal",
            skill_ids=(LEGAL_CONTRACT_REVIEW.skill_id,),
            register=_register_legal_skills,
            status=SkillBundleStatus.STABLE,
            description="Legal domain skill packs",
        ),
        override=override,
    )
