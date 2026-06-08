# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.browser.manifests import (
    BROWSER_RESEARCH_FETCH,
    BROWSER_INTERACTIVE_RUN,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_BROWSER_MANIFESTS = (
    BROWSER_RESEARCH_FETCH,
    BROWSER_INTERACTIVE_RUN,
)


class BrowserSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="browser",
            skill_ids=tuple(m.skill_id for m in _BROWSER_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Browser automation research skill packs (SK-EXP P1)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _BROWSER_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _BROWSER_MANIFESTS:
            registry.register(manifest)
