# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.providers.harness.manifests import (
    HARNESS_CONTEXT_DEMO,
    HARNESS_TOOL_SMOKE,
    HARNESS_TRACE_READ,
)
from intergrax.skills.registry.catalog import SkillBundleEntry, SkillBundleStatus, register_skill_bundle
from intergrax.skills.registry.runtime import SkillRegistry


def _register_harness_skills(registry: SkillRegistry) -> None:
    registry.register(HARNESS_TOOL_SMOKE)
    registry.register(HARNESS_CONTEXT_DEMO)
    registry.register(HARNESS_TRACE_READ)


def register_harness_skill_bundle(*, override: bool = False) -> None:
    register_skill_bundle(
        SkillBundleEntry(
            bundle_id="harness",
            skill_ids=(
                HARNESS_TOOL_SMOKE.skill_id,
                HARNESS_CONTEXT_DEMO.skill_id,
                HARNESS_TRACE_READ.skill_id,
            ),
            register=_register_harness_skills,
            status=SkillBundleStatus.STABLE,
            description="Platform harness capability packs (Phase S)",
        ),
        override=override,
    )
