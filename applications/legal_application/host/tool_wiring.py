# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool catalog wiring for legal_application."""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.profile import ToolProfile
from legal_application.host.settings import LegalBackendSettings


def wire_legal_tools(
    *,
    settings: LegalBackendSettings,
    integration_profile: IntegrationProfile | None = None,
) -> ApplicationToolWiring:
    """
    Enable catalog tools declared in :class:`LegalBackendSettings`.

    RAG / websearch require matching ``LEGAL_ENABLE_*`` flags **and** runtime
    managers wired into ``ToolWiringContext`` (vectorstore, embeddings, websearch executor).
    """
    enabled = list(settings.enabled_tool_ids)
    profile = ToolProfile(enabled=enabled) if enabled else ToolProfile()
    return build_application_tool_wiring(
        profile,
        integration_profile=integration_profile,
    )
