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
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications._shared.task_memory_wiring import wire_task_memory_from_profile
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.applications._shared.observability_assembly_resolver import (
    assert_observability_assembly_valid,
)
from intergrax.applications._shared.observability_wiring import wire_application_observability
from intergrax.applications._shared.reliability_assembly_resolver import (
    assert_reliability_assembly_valid,
)
from intergrax.applications._shared.reliability_wiring import (
    ApplicationReliabilityWiring,
    wire_application_reliability,
)
from intergrax.applications._shared.security_assembly_resolver import (
    assert_security_assembly_valid,
)
from intergrax.applications._shared.cost_assembly_resolver import (
    assert_cost_assembly_valid,
)
from intergrax.applications._shared.cost_wiring import (
    ApplicationCostWiring,
    wire_application_cost,
)
from intergrax.applications._shared.evaluation_assembly_resolver import (
    assert_evaluation_assembly_valid,
)
from intergrax.applications._shared.evaluation_wiring import (
    ApplicationEvaluationWiring,
    wire_application_evaluation,
)
from intergrax.applications._shared.security_wiring import (
    ApplicationSecurityWiring,
    wire_application_security,
)
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores
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
    reliability: ApplicationReliabilityWiring
    security: ApplicationSecurityWiring
    cost: ApplicationCostWiring
    evaluation: ApplicationEvaluationWiring
    nexus_loop: NexusLoop


def build_harness_host_runtime(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    *,
    settings: Any = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    idempotency_db_path: Path | None = None,
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
    observability_wiring = wire_application_observability(
        environment,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        integration_profile=environment.integration_profile,
    )
    if use_in_memory_trace:
        from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability

        observability = wire_nexus_observability(
            trace_db_path=trace_db_path,
            runtime_events_db_path=runtime_events_db_path,
            integration_profile=environment.integration_profile,
            use_in_memory_trace=True,
            enable_runtime_events=False,
        )
    else:
        assert_observability_assembly_valid(observability_wiring, environment)
        observability = observability_wiring.stores
    reliability_wiring = wire_application_reliability(
        environment,
        idempotency_db_path=idempotency_db_path,
    )
    assert_reliability_assembly_valid(reliability_wiring, environment)
    security_wiring = wire_application_security(environment)
    assert_security_assembly_valid(security_wiring, environment)
    cost_wiring = wire_application_cost(environment)
    assert_cost_assembly_valid(cost_wiring, environment)
    evaluation_wiring = wire_application_evaluation(environment)
    assert_evaluation_assembly_valid(evaluation_wiring, environment)
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
        llm_adapter=resolve_llm_adapter(environment),
        runtime_event_bus=env_wiring.build_context.runtime_event_bus,
        security_wiring=security_wiring,
    )
    assert_security_assembly_valid(security_wiring, environment, nexus=nexus_loop)
    _ = checkpoints_db_path
    return HarnessHostRuntime(
        manifest=resolved_manifest,
        environment=environment,
        env_wiring=env_wiring,
        registry=resolved_registry,
        observability=observability,
        reliability=reliability_wiring,
        security=security_wiring,
        cost=cost_wiring,
        evaluation=evaluation_wiring,
        nexus_loop=nexus_loop,
    )
