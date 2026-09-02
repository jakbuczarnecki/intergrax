# © Artur Czarnecki. All rights reserved.

"""Validate skill tool requirements against host ToolProfile during composition."""

from __future__ import annotations

from intergrax.skills.registry.factory import build_registry_from_profile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.registry.tool_requirements import (
    SkillToolRequirementError,
    SkillToolRequirementResolution,
    assert_skill_tool_requirements_satisfied,
    available_tool_ids_for_profile,
    resolve_skill_tool_requirements,
)
from intergrax.tools.registry.profile import ToolProfile

__all__ = [
    "SkillToolRequirementError",
    "SkillToolRequirementResolution",
    "assert_skill_tool_requirements_for_profile",
    "resolve_skill_tool_requirements_for_profile",
]


def _skill_profile_has_selection(skill_profile: SkillProfile) -> bool:
    return bool(
        skill_profile.enabled
        or skill_profile.enabled_bundles
        or skill_profile.register_all_catalog_bundles
    )


def resolve_skill_tool_requirements_for_profile(
    tool_profile: ToolProfile,
    skill_profile: SkillProfile,
    *,
    skill_registry: SkillRegistry | None = None,
) -> SkillToolRequirementResolution:
    """Resolve enabled skill tool requirements against ``tool_profile`` availability."""
    if not _skill_profile_has_selection(skill_profile):
        available = available_tool_ids_for_profile(tool_profile)
        return SkillToolRequirementResolution(
            required_tool_ids=(),
            available_tool_ids=available,
            satisfied_tool_ids=(),
            missing_tool_ids=(),
            is_satisfied=True,
            violations=(),
        )

    registry = skill_registry or build_registry_from_profile(skill_profile)
    return resolve_skill_tool_requirements(
        registry,
        available_tool_ids_for_profile(tool_profile),
    )


def assert_skill_tool_requirements_for_profile(
    tool_profile: ToolProfile,
    skill_profile: SkillProfile,
    *,
    skill_registry: SkillRegistry | None = None,
) -> SkillToolRequirementResolution:
    """Fail closed when skill requirements exceed host tool availability."""
    if not _skill_profile_has_selection(skill_profile):
        return resolve_skill_tool_requirements_for_profile(
            tool_profile,
            skill_profile,
            skill_registry=skill_registry,
        )

    registry = skill_registry or build_registry_from_profile(skill_profile)
    return assert_skill_tool_requirements_satisfied(registry, tool_profile)
