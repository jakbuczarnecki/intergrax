# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.interaction.manifests import INTERACTION_SESSION_HANDLER, INTERACTION_INPUT_CAPTURE
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class InteractionSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="interaction",
            skill_ids=(INTERACTION_SESSION_HANDLER.skill_id, INTERACTION_INPUT_CAPTURE.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Interaction skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (INTERACTION_SESSION_HANDLER, INTERACTION_INPUT_CAPTURE)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(INTERACTION_SESSION_HANDLER)
        registry.register(INTERACTION_INPUT_CAPTURE)
