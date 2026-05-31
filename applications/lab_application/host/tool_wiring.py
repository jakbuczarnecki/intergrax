# © Artur Czarnecki. All rights reserved.

"""Tool catalog wiring for lab_application (Phase O.8)."""

from __future__ import annotations

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.registry.profile import ToolProfile


def wire_lab_tools(
    *,
    integration_profile: IntegrationProfile | None = None,
    harness: bool = False,
) -> ApplicationToolWiring:
    """
    Laboratory tool wiring — context retrieval tools enabled by default.

    Pass ``integration_profile`` from ``wire_lab_integrations()`` when issue/wiki
    tools should resolve integration contracts automatically.
    """
    enabled = ["rag.retrieve", "websearch.query", "sandbox.exec"]
    if harness:
        enabled.extend(
            [
                "errors.capture",
                "observability.query_traces",
                "pagerduty.trigger_incident",
                "gitlab.create_issue",
                "braintrust.log_eval",
            ]
        )
    profile = ToolProfile(enabled=enabled)
    return build_application_tool_wiring(
        profile,
        integration_profile=integration_profile,
    )
