# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.registry.catalog import iter_bundles
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry


def build_registry_from_profile(
    profile: SkillProfile,
    *,
    registry: SkillRegistry | None = None,
) -> SkillRegistry:
    target = registry or SkillRegistry()
    for entry in iter_bundles():
        if not profile.should_register_bundle(entry.bundle_id, skill_ids=entry.skill_ids):
            continue
        entry.register_bundle(target)

    if profile.enabled and not profile.register_all_catalog_bundles:
        enabled = set(profile.enabled)
        for skill_id in list(target.skill_ids()):
            if skill_id not in enabled:
                target.unregister(skill_id)

    return target


def enabled_skill_ids_for_profile(profile: SkillProfile) -> list[str]:
    """Enabled skill ids — explicit ``enabled`` / ``enabled_bundles`` only (not catalog registration)."""
    if not profile.enabled and not profile.enabled_bundles:
        return []
    ids: set[str] = set(profile.enabled)
    enabled_set = set(profile.enabled)
    for entry in iter_bundles():
        normalized = entry.bundle_id.strip().lower()
        if normalized in profile.enabled_bundles:
            ids.update(entry.skill_ids)
        elif profile.enabled and any(sid in enabled_set for sid in entry.skill_ids):
            ids.update(entry.skill_ids)
    if profile.enabled:
        ids &= enabled_set
    return sorted(ids)
