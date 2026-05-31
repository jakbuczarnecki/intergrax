# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Build runtime ``ToolRegistry`` instances from catalog + profile (Phase O.2)."""

from __future__ import annotations

from typing import Optional

from intergrax.tools.registry.catalog import iter_bundles, list_catalog_tool_ids
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext


def build_registry_from_profile(
    profile: ToolProfile,
    *,
    ctx: Optional[ToolWiringContext] = None,
    registry: Optional[ToolRegistry] = None,
) -> ToolRegistry:
    """
    Populate a ``ToolRegistry`` from catalog bundles selected by ``ToolProfile``.

    Bundles whose tools are not enabled are skipped. Handlers receive ``ctx``.
    """
    resolved_ctx = ctx or ToolWiringContext()
    target = registry or ToolRegistry()

    for entry in iter_bundles():
        if not profile.should_register_bundle(entry.bundle_id, tool_ids=entry.tool_ids):
            continue
        entry.register(target, resolved_ctx)

    if profile.enabled and not profile.register_all_catalog_bundles:
        _prune_disabled_tools(target, profile)

    return target


def _prune_disabled_tools(registry: ToolRegistry, profile: ToolProfile) -> None:
    """Remove tools registered by a bundle but not listed in ``profile.enabled``."""
    if profile.enabled_bundles and not profile.enabled:
        return

    enabled = set(profile.enabled)
    for tool_id in list(registry.tool_ids()):
        if tool_id not in enabled:
            registry.unregister(tool_id)


def enabled_tool_ids_for_profile(profile: ToolProfile) -> list[str]:
    """Resolve the tool ids that would be registered for a profile (planning aid)."""
    if profile.register_all_catalog_bundles:
        return list_catalog_tool_ids()

    ids: set[str] = set(profile.enabled)
    for entry in iter_bundles():
        if profile.should_register_bundle(entry.bundle_id, tool_ids=entry.tool_ids):
            ids.update(entry.tool_ids)
    if profile.enabled:
        ids &= set(profile.enabled)
    return sorted(ids)
