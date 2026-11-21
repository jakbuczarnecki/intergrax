# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical host tool availability for a ``ToolProfile`` (contract-side entry point).

The tool catalog lives in ``intergrax.tools.registry``; it is imported inside the function on
purpose so this contract module carries no module-level registry edge (pinned by NT-WA-KR18-3).
"""

from __future__ import annotations

from intergrax.tools.contracts.tool_profile import ToolProfile


def available_tool_ids_for_profile(tool_profile: ToolProfile) -> tuple[str, ...]:
    """Canonical host tool availability for composition-time validation."""
    from intergrax.tools.registry.catalog import list_catalog_tool_ids
    from intergrax.tools.registry.profile import is_tool_enabled

    if tool_profile.register_all_catalog_bundles:
        return tuple(sorted(list_catalog_tool_ids()))

    available = {tool_id.strip() for tool_id in tool_profile.enabled if tool_id.strip()}
    for tool_id in list_catalog_tool_ids():
        if is_tool_enabled(tool_profile, tool_id):
            available.add(tool_id)
    return tuple(sorted(available))


__all__ = ["available_tool_ids_for_profile"]
