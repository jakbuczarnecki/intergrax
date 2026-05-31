# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for poc_template_application (Phase O.8)."""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.profile import ToolProfile


def wire_poc_template_tools(
    *,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationToolWiring:
    """PoC template — same defaults as lab (context + sandbox tools)."""
    profile = ToolProfile(
        enabled=["rag.retrieve", "websearch.query", "sandbox.exec"],
    )
    return build_application_tool_wiring(
        profile,
        integration_profile=integration_profile,
    )
