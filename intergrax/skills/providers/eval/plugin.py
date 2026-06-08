# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.eval.manifests import (
    EVAL_RELEASE_COMPARE,
    EVAL_SCORE_LOGGER,
    EVAL_TRAJECTORY_JUDGE,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class EvalSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="eval",
            skill_ids=(
                EVAL_SCORE_LOGGER.skill_id,
                EVAL_TRAJECTORY_JUDGE.skill_id,
                EVAL_RELEASE_COMPARE.skill_id,
            ),
            status=SkillBundleStatus.STABLE,
            description="Evaluation and scoring skill packs (SK-EXP2 + SK-EXP3)",
        )

    _MANIFESTS = (EVAL_SCORE_LOGGER, EVAL_TRAJECTORY_JUDGE, EVAL_RELEASE_COMPARE)

    @classmethod
    def skill_manifests(cls) -> tuple:
        return EvalSkillPlugin._MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in EvalSkillPlugin._MANIFESTS:
            registry.register(manifest)
