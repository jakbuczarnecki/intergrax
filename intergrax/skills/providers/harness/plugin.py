# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.harness.manifests import (
    HARNESS_TOOL_SMOKE,
    HARNESS_CONTEXT_DEMO,
    HARNESS_TRACE_READ,
    HARNESS_MODALITY_SMOKE,
    HARNESS_VISION_QA,
    HARNESS_SKILL_REGISTRY,
    HARNESS_INTEGRATION_BRIDGE_SMOKE,
    HARNESS_RELIABILITY_SMOKE,
    HARNESS_POLICY_SMOKE,
    HARNESS_STACK_DEMO,
    HARNESS_RUN_COMPARATOR,
    HARNESS_RUN_EXPORTER,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_HARNESS_MANIFESTS = (
    HARNESS_TOOL_SMOKE,
    HARNESS_CONTEXT_DEMO,
    HARNESS_TRACE_READ,
    HARNESS_MODALITY_SMOKE,
    HARNESS_VISION_QA,
    HARNESS_SKILL_REGISTRY,
    HARNESS_INTEGRATION_BRIDGE_SMOKE,
    HARNESS_RELIABILITY_SMOKE,
    HARNESS_POLICY_SMOKE,
    HARNESS_STACK_DEMO,
    HARNESS_RUN_COMPARATOR,
    HARNESS_RUN_EXPORTER,
)


class HarnessSkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="harness",
            skill_ids=tuple(m.skill_id for m in _HARNESS_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="Platform harness capability packs (Phase S) (SK-EXP4)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _HARNESS_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _HARNESS_MANIFESTS:
            registry.register(manifest)
