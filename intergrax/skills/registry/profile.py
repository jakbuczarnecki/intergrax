# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export plus catalog-aware skill enablement evaluator."""

from __future__ import annotations

from intergrax.skills.contracts.skill_profile import SkillProfile
from intergrax.skills.registry.catalog import get_bundle


def is_skill_enabled(profile: SkillProfile, skill_id: str) -> bool:
    """Evaluate enablement including catalog bundle membership."""
    if profile.register_all_catalog_bundles:
        return True
    if skill_id in profile.enabled:
        return True
    if not profile.enabled and not profile.enabled_bundles:
        return False
    for bundle_id in profile.enabled_bundles:
        try:
            entry = get_bundle(bundle_id)
        except KeyError:
            continue
        if skill_id in entry.skill_ids:
            return True
    return False


__all__ = ["SkillProfile", "is_skill_enabled"]
