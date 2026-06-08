# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.eval.manifests import (
    EVAL_SCORE_LOGGER,
    EVAL_TRAJECTORY_JUDGE,
    EVAL_RELEASE_COMPARE,
    EVAL_OBSERVATION_BROWSER,
    EVAL_BASELINE_RUNNER,
    EVAL_REGRESSION_GUARD,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_EVAL_MANIFESTS = (
    EVAL_SCORE_LOGGER,
    EVAL_TRAJECTORY_JUDGE,
    EVAL_RELEASE_COMPARE,
    EVAL_OBSERVATION_BROWSER,
    EVAL_BASELINE_RUNNER,
    EVAL_REGRESSION_GUARD,
)


class EvalSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="eval",
            skill_ids=tuple(m.skill_id for m in _EVAL_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="eval skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _EVAL_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _EVAL_MANIFESTS:
            registry.register(manifest)
