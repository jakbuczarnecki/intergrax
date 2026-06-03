# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.registry.agent_registry import AgentRegistry
from lab_application.host.agent_builders import LAB_AGENT_BUILDERS
from lab_application.host.settings import LabApplicationSettings
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
    env = manifest.environment or build_lab_environment_profile(settings)
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(
        manifest,
        env,
        settings=settings,
        integration_profile=integration_profile or manifest.integration_profile,
        runtime_event_bus=runtime_event_bus or RuntimeEventBus(),
        strict_harness=settings.strict_harness,
        trace_db_path=trace_db_path,
    )
    return build_application_registry(
        manifest,
        env_wiring.build_context,
        builders=LAB_AGENT_BUILDERS,
    )
