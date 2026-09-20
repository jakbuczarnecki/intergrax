# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export — canonical: skills.contracts.skill_profile."""

from __future__ import annotations

from intergrax.skills.contracts.skill_profile import SkillProfile
from intergrax.skills.registry.catalog import get_bundle


def _catalog_is_skill_enabled(self: SkillProfile, skill_id: str) -> bool:
    if self.register_all_catalog_bundles:
        return True
    if skill_id in self.enabled:
        return True
    if not self.enabled and not self.enabled_bundles:
        return False
    for bundle_id in self.enabled_bundles:
        try:
            entry = get_bundle(bundle_id)
        except KeyError:
            continue
        if skill_id in entry.skill_ids:
            return True
    return False


setattr(SkillProfile, "is_skill_enabled", _catalog_is_skill_enabled)

__all__ = ["SkillProfile"]
