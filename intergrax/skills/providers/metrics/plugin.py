# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.metrics.manifests import METRICS_RUN_OBSERVER
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class MetricsSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="metrics",
            skill_ids=(METRICS_RUN_OBSERVER.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Metrics and monitoring skill packs (SK-EXP3)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (METRICS_RUN_OBSERVER,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(METRICS_RUN_OBSERVER)
