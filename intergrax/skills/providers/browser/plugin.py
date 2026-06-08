# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.browser.manifests import BROWSER_RESEARCH_FETCH
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry


class BrowserSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="browser",
            skill_ids=(BROWSER_RESEARCH_FETCH.skill_id,),
            status=SkillBundleStatus.STABLE,
            description="Browser automation research skill packs (SK-EXP P1)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return (BROWSER_RESEARCH_FETCH,)

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        registry.register(BROWSER_RESEARCH_FETCH)
