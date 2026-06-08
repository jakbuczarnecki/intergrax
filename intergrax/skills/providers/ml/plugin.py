# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.ml.manifests import ML_EXPLAIN_PREDICT
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class MlSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="ml",
            skill_ids=(ML_EXPLAIN_PREDICT.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Ml skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (ML_EXPLAIN_PREDICT)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(ML_EXPLAIN_PREDICT)
