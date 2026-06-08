# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.data.manifests import (
    DATA_SQL_ANALYST,
    DATA_RECORDS_QUERY,
    DATA_SQL_MUTATOR,
    DATA_RECORDS_ADMIN,
    DATA_PIPELINE_PROBE,
    DATA_SCHEMA_DOCUMENTER,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_DATA_MANIFESTS = (
    DATA_SQL_ANALYST,
    DATA_RECORDS_QUERY,
    DATA_SQL_MUTATOR,
    DATA_RECORDS_ADMIN,
    DATA_PIPELINE_PROBE,
    DATA_SCHEMA_DOCUMENTER,
)


class DataSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="data",
            skill_ids=tuple(m.skill_id for m in _DATA_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="data skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _DATA_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _DATA_MANIFESTS:
            registry.register(manifest)
