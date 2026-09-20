# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export — canonical: tools.contracts.tool_profile."""

from __future__ import annotations

from intergrax.tools.contracts.tool_profile import ToolProfile, default_lab_tool_profile
from intergrax.tools.registry.catalog import get_bundle


def _catalog_is_tool_enabled(self: ToolProfile, tool_id: str) -> bool:
    if self.register_all_catalog_bundles:
        return True
    if tool_id in self.enabled:
        return True
    if not self.enabled and not self.enabled_bundles:
        return False
    for bundle_id in self.enabled_bundles:
        try:
            entry = get_bundle(bundle_id)
        except KeyError:
            continue
        if tool_id in entry.tool_ids:
            return True
    return False


setattr(ToolProfile, "is_tool_enabled", _catalog_is_tool_enabled)

__all__ = ["ToolProfile", "default_lab_tool_profile"]
