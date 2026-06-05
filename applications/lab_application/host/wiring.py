# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.registry.agent_registry import AgentRegistry
from lab_application.host.agent_builders import LAB_AGENT_BUILDERS
from lab_application.host.integration_wiring import LabIntegrationWiring, wire_lab_integrations
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest


def bootstrap_lab_integration_wiring(
    *,
    settings: LabApplicationSettings | None = None,
    db_path: Path | None = None,
    experiments_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    harness: bool | None = None,
    otel_enabled: bool | None = None,
) -> LabIntegrationWiring:
    """Lab superset integration bootstrap (factory must not import integration_wiring directly)."""
    settings = settings or LabApplicationSettings.from_env()
    return wire_lab_integrations(
        settings=settings,
        db_path=db_path,
        experiments_db_path=experiments_db_path,
        runtime_events_db_path=runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        harness=harness if harness is not None else settings.harness,
        otel_enabled=otel_enabled if otel_enabled is not None else settings.otel_enabled,
    )


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
