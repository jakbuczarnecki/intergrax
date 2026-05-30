# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 tool catalog wiring helpers (Phase O.8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry import ToolProfile, ToolRegistry, ToolWiringContext, build_registry_from_profile
from intergrax.tools.registry.bootstrap import register_default_tools


@dataclass(frozen=True)
class ApplicationToolWiring:
    """Resolved tool profile + wiring context + materialized registry."""

    profile: ToolProfile
    wiring_context: ToolWiringContext
    registry: ToolRegistry


def build_application_tool_wiring(
    profile: ToolProfile,
    *,
    integration_profile: IntegrationProfile | None = None,
    wiring_context: ToolWiringContext | None = None,
    extras: dict[str, Any] | None = None,
) -> ApplicationToolWiring:
    """
    Build a catalog registry for Tier-3 hosts (lab, product, MCP export).

    Call ``register_default_tools()`` once, compose ``ToolWiringContext`` from
    integrations + runtime managers, then enable bundles via ``ToolProfile``.
    """
    register_default_tools()
    ctx = wiring_context
    if ctx is None and integration_profile is not None:
        ctx = ToolWiringContext.from_integration_profile(integration_profile, extras=extras)
    ctx = ctx or ToolWiringContext(extras=dict(extras or {}))
    registry = build_registry_from_profile(profile, ctx=ctx)
    return ApplicationToolWiring(profile=profile, wiring_context=ctx, registry=registry)
