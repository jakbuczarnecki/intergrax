# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.catalog.manifests import CATALOG_TOOL_INTROSPECT
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class CatalogSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="catalog",
            skill_ids=(CATALOG_TOOL_INTROSPECT.skill_id),
            status=SkillBundleStatus.STABLE,
            description="Catalog skill packs (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (CATALOG_TOOL_INTROSPECT)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(CATALOG_TOOL_INTROSPECT)
