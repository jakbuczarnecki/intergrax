# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.harness.manifests import (
    HARNESS_CONTEXT_DEMO,
    HARNESS_SKILL_REGISTRY,
    HARNESS_TOOL_SMOKE,
    HARNESS_TRACE_READ,
    HARNESS_VISION_QA,
)
from intergrax.skills.registry.catalog import SkillBundleEntry, SkillBundleStatus, register_skill_bundle
from intergrax.skills.registry.runtime import SkillRegistry


def _register_harness_skills(registry: SkillRegistry) -> None:
    registry.register(HARNESS_TOOL_SMOKE)
    registry.register(HARNESS_CONTEXT_DEMO)
    registry.register(HARNESS_TRACE_READ)
    registry.register(HARNESS_SKILL_REGISTRY)
    registry.register(HARNESS_VISION_QA)


def register_harness_skill_bundle(*, override: bool = False) -> None:
    register_skill_bundle(
        SkillBundleEntry(
            bundle_id="harness",
            skill_ids=(
                HARNESS_TOOL_SMOKE.skill_id,
                HARNESS_CONTEXT_DEMO.skill_id,
                HARNESS_TRACE_READ.skill_id,
                HARNESS_SKILL_REGISTRY.skill_id,
                HARNESS_VISION_QA.skill_id,
            ),
            register=_register_harness_skills,
            status=SkillBundleStatus.STABLE,
            description="Platform harness capability packs (Phase S)",
        ),
        override=override,
    )
