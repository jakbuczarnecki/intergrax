# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.eval.manifests import EVAL_SCORE_LOGGER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class EvalSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="eval",
            skill_ids=(EVAL_SCORE_LOGGER.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Evaluation and scoring skill packs (SK-EXP2)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (EVAL_SCORE_LOGGER,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(EVAL_SCORE_LOGGER)
