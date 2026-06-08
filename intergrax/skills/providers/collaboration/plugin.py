# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.collaboration.manifests import COLLABORATION_OUTREACH
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class CollaborationSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="collaboration",
            skill_ids=(COLLABORATION_OUTREACH.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Collaboration suite skill packs (SK-EXP P1)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (COLLABORATION_OUTREACH,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(COLLABORATION_OUTREACH)
