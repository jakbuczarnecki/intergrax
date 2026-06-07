# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for poc_template_application (Phase O.8)."""

from __future__ import annotations

from intergrax.applications._shared.integration_tool_profile import extend_tool_profile_for_integration
from intergrax.applications._shared.integration_tool_wiring import wire_integration_tool_context
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.wiring import ToolWiringContext


def wire_poc_template_tools(
    *,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationToolWiring:
    """PoC template — lab defaults plus P6 integration-backed catalog tools."""
    profile = ToolProfile(
        enabled=["rag.retrieve", "websearch.query", "sandbox.exec"],
    )
    profile = extend_tool_profile_for_integration(profile, integration_profile)
    wiring_context = ToolWiringContext()
    if integration_profile is not None:
        wiring_context = ToolWiringContext.from_integration_profile(integration_profile)
        wiring_context = wire_integration_tool_context(wiring_context, integration_profile)
    return build_application_tool_wiring(
        profile,
        integration_profile=integration_profile,
        wiring_context=wiring_context,
    )
