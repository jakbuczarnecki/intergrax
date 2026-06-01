# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.registry.catalog import iter_bundles, list_catalog_skill_ids
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
    if profile.register_all_catalog_bundles:
        return list_catalog_skill_ids()
    ids: set[str] = set(profile.enabled)
    for entry in iter_bundles():
        if profile.should_register_bundle(entry.bundle_id, skill_ids=entry.skill_ids):
            ids.update(entry.skill_ids)
    if profile.enabled:
        ids &= set(profile.enabled)
    return sorted(ids)
