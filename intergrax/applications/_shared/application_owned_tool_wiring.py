# © Artur Czarnecki. All rights reserved.

"""Merge application-owned tool registries into environment wiring (PLATFORM-5B)."""

from __future__ import annotations

from intergrax.applications._shared.application_owned_tool_conformance import (
    declared_application_owned_tool_ids,
    merge_application_owned_tool_registry,
)
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.tools.registry.runtime import ToolRegistry


def apply_application_owned_tool_registry(
    manifest: ApplicationManifest,
    tool_wiring: ApplicationToolWiring,
    application_tool_registry: ToolRegistry | None,
) -> ApplicationToolWiring:
    """Merge a pre-registered application tool registry into wired catalog tools."""
    if application_tool_registry is None:
        return tool_wiring
    declared = declared_application_owned_tool_ids(manifest)
    if not declared and application_tool_registry.tool_ids():
        from intergrax.applications.contracts.errors import ApplicationManifestConformanceError

        raise ApplicationManifestConformanceError(
            "application tool registry provided without manifest application_owned_tools declarations",
        )
    merged_registry = merge_application_owned_tool_registry(
        catalog_registry=tool_wiring.registry,
        application_registry=application_tool_registry,
        declared_tool_ids=declared,
    )
    return ApplicationToolWiring(
        profile=tool_wiring.profile,
        wiring_context=tool_wiring.wiring_context,
        registry=merged_registry,
    )
