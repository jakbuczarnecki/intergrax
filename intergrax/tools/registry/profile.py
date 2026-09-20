# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export plus catalog-aware tool enablement evaluator."""

from __future__ import annotations

from intergrax.tools.contracts.tool_profile import ToolProfile, default_lab_tool_profile
from intergrax.tools.registry.catalog import get_bundle


def is_tool_enabled(profile: ToolProfile, tool_id: str) -> bool:
    """Evaluate enablement including catalog bundle membership."""
    if profile.register_all_catalog_bundles:
        return True
    if tool_id in profile.enabled:
        return True
    if not profile.enabled and not profile.enabled_bundles:
        return False
    for bundle_id in profile.enabled_bundles:
        try:
            entry = get_bundle(bundle_id)
        except KeyError:
            continue
        if tool_id in entry.tool_ids:
            return True
    return False


__all__ = ["ToolProfile", "default_lab_tool_profile", "is_tool_enabled"]
