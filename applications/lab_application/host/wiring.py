# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.policy_wiring import build_runtime_policy_bundle
from intergrax.applications._shared.skill_wiring import build_application_skill_wiring, lab_skill_profile
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.registry.agent_registry import AgentRegistry
from lab_application.host.agent_builders import LAB_AGENT_BUILDERS
from lab_application.host.settings import LabApplicationSettings
from lab_application.host.tool_wiring import wire_lab_tools
from lab_application.manifest import build_lab_manifest


def build_lab_registry(
    *,
    settings: LabApplicationSettings | None = None,
    integration_profile=None,
    runtime_event_bus: RuntimeEventBus | None = None,
    trace_db_path=None,
) -> AgentRegistry:
    """
    Compose the lab agent registry from manifest + builders (Tier-3 unified wiring).

    Agent roster flags come from :class:`LabApplicationSettings`; instance creation
    uses :data:`LAB_AGENT_BUILDERS` (zero-arg agents today, factories when needed).
    """
    settings = settings or LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    profile = integration_profile or manifest.integration_profile
    tool_wiring = wire_lab_tools(
        integration_profile=profile,
        harness=settings.harness,
    )
    skill_wiring = build_application_skill_wiring(lab_skill_profile())
    tool_registry = tool_wiring.registry
    if not tool_wiring.profile.enabled and not tool_wiring.profile.enabled_bundles:
        tool_registry = None
    ctx = ApplicationBuildContext.for_manifest(
        manifest,
        settings=settings,
        integration_profile=profile,
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
        skill_profile=skill_wiring.profile,
        skill_registry=skill_wiring.registry,
        tool_registry=tool_registry,
        policy_bundle=build_runtime_policy_bundle(),
        runtime_event_bus=runtime_event_bus or RuntimeEventBus(),
        strict_harness=settings.strict_harness,
        trace_db_path=trace_db_path,
    )
    return build_application_registry(manifest, ctx, builders=LAB_AGENT_BUILDERS)
