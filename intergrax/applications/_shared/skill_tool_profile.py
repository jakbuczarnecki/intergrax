# © Artur Czarnecki. All rights reserved.

"""Enable catalog tools referenced by enabled skill bundles."""

from __future__ import annotations

from intergrax.skills.registry.bootstrap import register_default_skills
from intergrax.skills.registry.factory import build_registry_from_profile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile


def tool_ids_referenced_by_skill_profile(skill_profile: SkillProfile) -> tuple[str, ...]:
    """Return sorted tool_ids declared on skills selected by ``skill_profile``."""
    if (
        not skill_profile.enabled
        and not skill_profile.enabled_bundles
        and not skill_profile.register_all_catalog_bundles
    ):
        return ()

    register_default_skills()
    registry = build_registry_from_profile(skill_profile)
    referenced: set[str] = set()
    for skill_id in registry.skill_ids():
        manifest = registry.get(skill_id).manifest
        referenced.update(tid.strip() for tid in manifest.tool_ids if tid.strip())
    return tuple(sorted(referenced))


def extend_tool_profile_for_skills(
    tool_profile: ToolProfile,
    skill_profile: SkillProfile,
) -> ToolProfile:
    """Append skill-declared tool_ids so SkillResolver validation can succeed."""
    additions = tool_ids_referenced_by_skill_profile(skill_profile)
    if not additions:
        return tool_profile

    enabled = list(tool_profile.enabled)
    for tool_id in additions:
        if tool_id not in enabled:
            enabled.append(tool_id)
    return tool_profile.model_copy(update={"enabled": enabled})
