# © Artur Czarnecki. All rights reserved.

"""Harness platform skills as :class:`SkillPlugin`."""

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.harness.manifests import (
    HARNESS_CONTEXT_DEMO,
    HARNESS_MODALITY_SMOKE,
    HARNESS_SKILL_REGISTRY,
    HARNESS_STACK_DEMO,
    HARNESS_TOOL_SMOKE,
    HARNESS_TRACE_READ,
    HARNESS_VISION_QA,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_HARNESS_MANIFESTS = (
    HARNESS_TOOL_SMOKE,
    HARNESS_CONTEXT_DEMO,
    HARNESS_TRACE_READ,
    HARNESS_SKILL_REGISTRY,
    HARNESS_MODALITY_SMOKE,
    HARNESS_VISION_QA,
    HARNESS_STACK_DEMO,
)


class HarnessSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="harness",
            skill_ids=tuple(m.skill_id for m in _HARNESS_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Platform harness capability packs (Phase S)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _HARNESS_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _HARNESS_MANIFESTS:
            registry.register(manifest)
