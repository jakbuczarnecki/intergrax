# © Artur Czarnecki. All rights reserved.

"""Tier-3 wiring for ACP declarative catalog tool invoker."""

from __future__ import annotations

from intergrax.agents.persistence.catalog_declarative_invoker import (
    CatalogDeclarativeToolInvoker,
    build_catalog_declarative_invoker_from_registry,
)
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring


def build_declarative_invoker_from_tool_wiring(
    tool_wiring: ApplicationToolWiring,
) -> CatalogDeclarativeToolInvoker | None:
    """Materialize catalog invoker when host tool profile enables catalog tools."""
    if not tool_wiring.profile.enabled and not tool_wiring.profile.enabled_bundles:
        return None
    return build_catalog_declarative_invoker_from_registry(tool_wiring.registry)
