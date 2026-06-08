# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.sandbox.manifests import (
    SANDBOX_CODE_EXEC,
    SANDBOX_TEST_RUNNER,
    SANDBOX_REFACTOR_LOOP,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_SANDBOX_MANIFESTS = (
    SANDBOX_CODE_EXEC,
    SANDBOX_TEST_RUNNER,
    SANDBOX_REFACTOR_LOOP,
)


class SandboxSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="sandbox",
            skill_ids=tuple(m.skill_id for m in _SANDBOX_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="sandbox skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _SANDBOX_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _SANDBOX_MANIFESTS:
            registry.register(manifest)
