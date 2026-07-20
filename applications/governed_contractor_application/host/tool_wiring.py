# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for governed_contractor_application (Phase O.8)."""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.profile import ToolProfile


def wire_governed_contractor_tools(
    *,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationToolWiring:
    profile = ToolProfile(
        enabled=[
            "rag.retrieve",
            "rag.ingest_document",
            "rag.list_collections",
            "websearch.query",
            "websearch.read_url",
            "websearch.fetch_batch",
        ],
    )
    return build_application_tool_wiring(
        profile,
        integration_profile=integration_profile or IntegrationProfile(),
    )
