# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.registry.agent_registry import AgentRegistry
from lab_application.host.agent_builders import LAB_AGENT_BUILDERS
from lab_application.host.settings import LabApplicationSettings
from lab_application.host.tool_wiring import wire_lab_tools
from lab_application.manifest import build_lab_manifest


def build_lab_registry(
    *,
    settings: LabApplicationSettings | None = None,
    integration_profile=None,
) -> AgentRegistry:
    """
    Compose the lab agent registry from manifest + builders (Tier-3 unified wiring).

    Agent roster flags come from :class:`LabApplicationSettings`; instance creation
    uses :data:`LAB_AGENT_BUILDERS` (zero-arg agents today, factories when needed).
    """
    settings = settings or LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    profile = integration_profile or getattr(manifest, "integration_profile", None)
    tool_wiring = wire_lab_tools(integration_profile=profile)
    ctx = ApplicationBuildContext.for_manifest(
        manifest,
        settings=settings,
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
        policy_bundle=build_runtime_policy_bundle(),
    )
    return build_application_registry(manifest, ctx, builders=LAB_AGENT_BUILDERS)
