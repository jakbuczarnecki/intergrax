# © Artur Czarnecki. All rights reserved.

"""Shared Tier-3 host runtime assembly (Phase DX-1.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.applications._shared.environment_wiring import (
    ApplicationEnvironmentWiring,
    wire_application_environment,
)
from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications._shared.task_memory_wiring import wire_task_memory_from_profile
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores, wire_nexus_observability
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.registry.agent_registry import AgentRegistry


@dataclass(frozen=True)
class HarnessHostRuntime:
    """Resolved Tier-3 runtime artifacts for HTTP/MCP hosts."""

    manifest: ApplicationManifest
    environment: ApplicationEnvironmentProfile
    env_wiring: ApplicationEnvironmentWiring
    registry: AgentRegistry
    observability: NexusObservabilityStores
    nexus_loop: NexusLoop


def build_harness_host_runtime(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    *,
    settings: Any = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    use_in_memory_trace: bool = False,
    builders: dict[type, Any] | None = None,
    registry: AgentRegistry | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    notification_adapter: NotificationAdapter | None = None,
) -> HarnessHostRuntime:
    """
    Single H-APP path: environment wiring → registry → observability → NexusLoop.

    Replaces per-host duplicate ``NexusLoop(...)`` construction in scaffold factories.
    """
    resolved_manifest = manifest
    if manifest.environment is None:
        resolved_manifest = manifest.model_copy(update={"environment": environment})

    env_wiring = wire_application_environment(
        resolved_manifest,
        environment,
        settings=settings,
    )
    resolved_registry = registry or build_application_registry(
        resolved_manifest,
        env_wiring.build_context,
        builders=builders,
    )
    observability = wire_nexus_observability(
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        integration_profile=environment.integration_profile,
        use_in_memory_trace=use_in_memory_trace,
    )
    task_memory = wire_task_memory_from_profile(environment)
    nexus_loop = build_nexus_loop_from_environment(
        resolved_registry,
        env=environment,
        trace_store=observability.trace_store,
        checkpoint_store=checkpoint_store,
        notification_adapter=notification_adapter,
        runtime_events_db_path=observability.runtime_events_db_path,
        task_memory_store=task_memory.store,
        task_memory_db_path=task_memory.db_path,
        shadow_manager=env_wiring.shadow_manager,
        sandbox_manager=env_wiring.sandbox_manager,
    )
    _ = checkpoints_db_path
    return HarnessHostRuntime(
        manifest=resolved_manifest,
        environment=environment,
        env_wiring=env_wiring,
        registry=resolved_registry,
        observability=observability,
        nexus_loop=nexus_loop,
    )
