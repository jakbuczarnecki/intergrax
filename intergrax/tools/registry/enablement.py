# © Artur Czarnecki. All rights reserved.

"""Catalog-aware tool enablement view (tools registry / composition owned)."""

from __future__ import annotations

from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.registry.profile import is_tool_enabled


class CatalogToolEnablementView:
    """Bind a ``ToolProfile`` DTO to catalog-aware enablement evaluation.

    Structurally satisfies ``ToolEnablementProfile`` without importing the
    agent-owned protocol (dependency direction: composition binds the view).
    """

    def __init__(self, profile: ToolProfile) -> None:
        self._profile = profile

    def is_tool_enabled(self, tool_id: str) -> bool:
        return is_tool_enabled(self._profile, tool_id)


def catalog_tool_enablement(
    profile: ToolProfile | None,
) -> CatalogToolEnablementView | None:
    """Wrap a DTO for injection into consumers that require enablement capability."""
    if profile is None:
        return None
    return CatalogToolEnablementView(profile)


__all__ = [
    "CatalogToolEnablementView",
    "catalog_tool_enablement",
]
