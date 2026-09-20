# © Artur Czarnecki. All rights reserved.

"""Composition binding for catalog-aware tool enablement injection."""

from __future__ import annotations

from intergrax.agents.tool_enablement import ToolEnablementProfile
from intergrax.tools.contracts.tool_profile import ToolProfile
from intergrax.tools.registry.enablement import catalog_tool_enablement


def resolve_tool_enablement(
    injected: ToolEnablementProfile | None,
    *,
    environment_tool_profile: ToolProfile | None = None,
) -> ToolEnablementProfile | None:
    """Bind DTO profiles through the catalog view; pass custom enablement through."""
    if injected is not None:
        if isinstance(injected, ToolProfile):
            return catalog_tool_enablement(injected)
        return injected
    return catalog_tool_enablement(environment_tool_profile)


__all__ = ["resolve_tool_enablement"]
