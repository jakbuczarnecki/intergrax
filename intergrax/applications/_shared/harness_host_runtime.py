# © Artur Czarnecki. All rights reserved.

"""Shared Tier-3 host runtime assembly (Phase DX-1.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intergrax.harness.application_host import ApplicationHost

from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.applications._shared.acp_checkpoint_host_wiring import (
    resolve_host_agent_checkpoint_store,
    resolve_host_compensation_queue_store,
)
from intergrax.applications._shared.application_host_wiring import (
    apply_application_environment_state_wiring,
    apply_application_host_wiring,
    apply_hook_runtime_guard_wiring,
)
from intergrax.applications._shared.cost_assembly_resolver import (
    assert_cost_assembly_valid,
)
from intergrax.applications._shared.cost_wiring import (
    ApplicationCostWiring,
    wire_application_cost,
)
from intergrax.applications._shared.decision_wiring import (
    application_decision_wiring_spec_from_environment,
    resolve_application_decision_agent_id,
    wire_application_decision,
)
from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.diagnostic_assembly_resolver import DiagnosticWiring
from intergrax.applications._shared.environment_wiring import (
    ApplicationEnvironmentWiring,
    wire_application_environment,
)
from intergrax.applications._shared.evaluation_assembly_resolver import (
    assert_evaluation_assembly_valid,
)
from intergrax.applications._shared.evaluation_wiring import (
    ApplicationEvaluationWiring,
    wire_application_evaluation,
)
from intergrax.applications._shared.guardrail_assembly_resolver import (
    assert_guardrail_assembly_valid,
)
from intergrax.applications._shared.guardrail_wiring import (
    ApplicationGuardrailWiring,
    wire_application_guardrail,
)
from intergrax.applications._shared.llm_resolver import resolve_environment_llm_adapter
from intergrax.applications._shared.nexus_factory import (
    build_nexus_loop_from_environment,
)
from intergrax.applications._shared.observability_assembly_resolver import (
    assert_observability_assembly_valid,
)
from intergrax.applications._shared.observability_wiring import (
    wire_application_observability,
)
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
from intergrax.applications._shared.security_wiring import (
    ApplicationSecurityWiring,
    wire_application_security,
)
from intergrax.applications._shared.task_memory_wiring import (
    wire_task_memory_from_profile,
)
from intergrax.applications._shared.harness_control_plane_governance_wiring import (
    HarnessControlPlaneGovernance,
    build_harness_control_plane_governance,
)
from intergrax.applications._shared.harness_registry_authority import (
    RegistryAssemblyMode,
    resolve_harness_host_registry,
    resolve_registry_assembly_mode,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RegistryProjectionEvidence,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.applications._shared.harness_host_runtime_compat import (
    HarnessHostLegacyComposition,
)
from intergrax.applications._shared.host_task_execution_wiring import (
    build_environment_host_task_execution,
)
from intergrax.applications._shared.profile_resolution import (
    materialize_effective_profile_revision,
    resolve_profile,
)
from intergrax.applications._shared.profile_resolution.activation_service import (
    EffectiveProfileActivationDependencies,
    EffectiveProfileActivationService,
    activate_materialized_revision,
)
from intergrax.applications._shared.profile_resolution.execution_admission import (
    EffectiveProfileExecutionPinningDependencies,
)
from intergrax.applications._shared.profile_resolution.wiring import (
    resolve_effective_profile_persistence_wiring,
)
from intergrax.applications.contracts.profile_resolution import (
    EffectiveProfileRevision,
    EffectiveProfileRevisionScope,
    ProfileLayerInput,
    ProfileResolution,
)
from intergrax.applications.contracts.profile_resolution.activation import (
    ActiveEffectiveProfileRevisionStore,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionPinningStore,
)
from intergrax.applications.contracts.profile_resolution.store import (
    EffectiveProfileRevisionStore,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


__all__ = [
    "HarnessHostRuntime",
    "RegistryAssemblyMode",
    "build_harness_host_runtime",
]


@dataclass(frozen=True)
class HarnessHostRuntime:
    """Resolved Tier-3 runtime artifacts for HTTP/MCP hosts.

    Public execution path: ``execution`` → canonical :class:`HostTaskExecution`
    (strategy-neutral root lifecycle). Nexus orchestration remains an internal
    backend reached through ORCHESTRATION strategy dispatch.
    """

    manifest: ApplicationManifest
    environment: ApplicationEnvironmentProfile
    env_wiring: ApplicationEnvironmentWiring
    registry: AgentRegistryRead
    observability: NexusObservabilityStores
    reliability: ApplicationReliabilityWiring
    security: ApplicationSecurityWiring
    guardrail: ApplicationGuardrailWiring
    cost: ApplicationCostWiring
    evaluation: ApplicationEvaluationWiring
    diagnostic_wiring: DiagnosticWiring
    execution: HostTaskExecution
    _legacy_composition: HarnessHostLegacyComposition
    application_host: ApplicationHost | None
    agent_checkpoint_store: AgentCheckpointStore
    compensation_queue_store: CompensationQueueStore
    registry_projection_evidence: RegistryProjectionEvidence | None = None
    boundary_event_buffer: BoundaryEventBuffer | None = None
    control_plane_governance: HarnessControlPlaneGovernance | None = None
    profile_resolution: ProfileResolution | None = None
    effective_profile_revision: EffectiveProfileRevision | None = None
    effective_profile_revision_store: EffectiveProfileRevisionStore | None = None
    effective_profile_pinning_store: EffectiveProfileExecutionPinningStore | None = None
    effective_profile_active_store: ActiveEffectiveProfileRevisionStore | None = None


def build_harness_host_runtime(
    manifest: ApplicationManifest,
    environment: ApplicationEnvironmentProfile,
    *,
    settings: Any = None,
    tenant_id: str | None = None,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    idempotency_db_path: Path | None = None,
    use_in_memory_trace: bool = False,
    builders: dict[type, Any] | None = None,
    registry: AgentRegistry | None = None,
    registry_projection: MaterializedRegistryProjection | None = None,
    registry_assembly_mode: RegistryAssemblyMode | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    agent_checkpoint_store: AgentCheckpointStore | None = None,
    notification_adapter: NotificationAdapter | None = None,
    application_host: ApplicationHost | None = None,
    document_store: Any | None = None,
    key_value_cache: Any | None = None,
    boundary_event_buffer: Any | None = None,
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    profile_layers: tuple[ProfileLayerInput, ...] = (),
    revision_store: EffectiveProfileRevisionStore | None = None,
    pinning_store: EffectiveProfileExecutionPinningStore | None = None,
    active_store: ActiveEffectiveProfileRevisionStore | None = None,
) -> HarnessHostRuntime:
    """
    Single H-APP path: environment → platform composition → canonical execution.

    Replaces per-host duplicate ``NexusLoop(...)`` construction in scaffold factories.
    Nexus is composed internally as the ORCHESTRATION strategy backend only.
    """
    resolved_manifest = manifest
    if manifest.environment is None:
        resolved_manifest = manifest.model_copy(update={"environment": environment})

    profile_resolution = resolve_profile(environment, layers=profile_layers)
    effective_environment = profile_resolution.effective_profile
    production_mode = effective_environment.execution_mode.value == "strict"
    kv_store = key_value_cache if isinstance(key_value_cache, DistributedKVStore) else None
    doc_store = document_store if isinstance(document_store, DocumentStore) else None
    profile_persistence = resolve_effective_profile_persistence_wiring(
        production_mode=production_mode,
        kv_store=kv_store,
        document_store=doc_store,
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
    )
    revision_scope = EffectiveProfileRevisionScope(
        application_id=resolved_manifest.app_id,
        tenant_id=tenant_id,
    )
    effective_profile_revision = materialize_effective_profile_revision(
        profile_resolution,
        scope=revision_scope,
        store=profile_persistence.revision_store,
    )
    activation_service = EffectiveProfileActivationService(
        EffectiveProfileActivationDependencies(
            revision_store=profile_persistence.revision_store,
            active_store=profile_persistence.active_store,
        ),
    )
    activate_materialized_revision(
        activation_service,
        scope=revision_scope,
        candidate_revision_id=effective_profile_revision.revision_id,
    )

    env_wiring = wire_application_environment(
        resolved_manifest,
        effective_environment,
        settings=settings,
        tenant_id=tenant_id,
        document_store=document_store,
        key_value_cache=key_value_cache,
        boundary_event_buffer=boundary_event_buffer,
    )
    assembly_mode = resolve_registry_assembly_mode(
        effective_environment,
        explicit=registry_assembly_mode,
    )
    resolved_registry, registry_evidence = resolve_harness_host_registry(
        manifest=resolved_manifest,
        build_context=env_wiring.build_context,
        environment=effective_environment,
        assembly_mode=assembly_mode,
        registry_projection=registry_projection,
        registry=registry,
        builders=builders,
    )
    observability_wiring = wire_application_observability(
        effective_environment,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        integration_profile=effective_environment.integration_profile,
    )
    if use_in_memory_trace:
        from intergrax.runtime.nexus.observability_wiring import (
            wire_nexus_observability,
        )

        observability = wire_nexus_observability(
            trace_db_path=trace_db_path,
            runtime_events_db_path=runtime_events_db_path,
            integration_profile=effective_environment.integration_profile,
            use_in_memory_trace=True,
            enable_runtime_events=False,
        )
    else:
        assert_observability_assembly_valid(observability_wiring, effective_environment)
        observability = observability_wiring.stores
    reliability_wiring = wire_application_reliability(
        effective_environment,
        idempotency_db_path=idempotency_db_path,
    )
    assert_reliability_assembly_valid(reliability_wiring, effective_environment)
    security_wiring = wire_application_security(effective_environment)
    assert_security_assembly_valid(security_wiring, effective_environment)
    guardrail_wiring = wire_application_guardrail(effective_environment)
    cost_wiring = wire_application_cost(effective_environment)
    assert_cost_assembly_valid(cost_wiring, effective_environment)
    evaluation_wiring = wire_application_evaluation(effective_environment)
    assert_evaluation_assembly_valid(evaluation_wiring, effective_environment)
    decision_spec = application_decision_wiring_spec_from_environment(effective_environment)
    decision_wiring = wire_application_decision(
        registry=resolved_registry,
        agent_id=resolve_application_decision_agent_id(resolved_registry, effective_environment),
        spec=decision_spec,
    )
    task_memory = wire_task_memory_from_profile(effective_environment)
    declarative_tool_invoker = build_declarative_invoker_from_tool_wiring(env_wiring.tool_wiring)
    resolved_agent_checkpoint_store = resolve_host_agent_checkpoint_store(
        agent_checkpoint_store=agent_checkpoint_store,
        checkpoints_db_path=checkpoints_db_path,
    )
    resolved_compensation_queue_store = resolve_host_compensation_queue_store(
        checkpoints_db_path=checkpoints_db_path,
    )
    nexus_loop = build_nexus_loop_from_environment(
        resolved_registry,
        env=effective_environment,
        trace_store=observability.trace_store,
        checkpoint_store=checkpoint_store,
        agent_checkpoint_store=resolved_agent_checkpoint_store,
        compensation_queue_store=resolved_compensation_queue_store,
        idempotency_store=reliability_wiring.idempotency_store,
        declarative_tool_invoker=declarative_tool_invoker,
        notification_adapter=notification_adapter,
        runtime_events_db_path=observability.runtime_events_db_path,
        task_memory_store=task_memory.store,
        task_memory_db_path=task_memory.db_path,
        shadow_manager=env_wiring.shadow_manager,
        sandbox_manager=env_wiring.sandbox_manager,
        llm_adapter=resolve_environment_llm_adapter(effective_environment, tenant_id="default"),
        runtime_event_bus=env_wiring.build_context.runtime_event_bus,
        security_wiring=security_wiring,
        guardrail_wiring=guardrail_wiring,
        decision_wiring=decision_wiring,
        run_budget=cost_wiring.run_budget,
        key_value_cache=key_value_cache,
        document_store=document_store,
    )
    assert_security_assembly_valid(security_wiring, effective_environment, nexus=nexus_loop)
    assert_guardrail_assembly_valid(guardrail_wiring, effective_environment, nexus=nexus_loop)
    from intergrax.applications._shared.capability_alias_intake_wiring import (
        apply_capability_alias_wiring,
    )
    from intergrax.applications._shared.environment_snapshot_wiring import (
        apply_environment_snapshot_wiring,
        cache_deploy_environment_snapshot,
    )

    apply_capability_alias_wiring(nexus_loop, environment=effective_environment)
    cache_deploy_environment_snapshot(
        resolved_manifest,
        effective_environment,
        registry_snapshot=env_wiring.registry_snapshot,
    )
    apply_environment_snapshot_wiring(
        nexus_loop,
        manifest=resolved_manifest,
        environment=effective_environment,
        registry_snapshot=env_wiring.registry_snapshot,
    )
    apply_application_environment_state_wiring(
        nexus_loop,
        manifest=resolved_manifest,
        environment=effective_environment,
        run_budget=cost_wiring.run_budget,
    )
    apply_application_host_wiring(nexus_loop, application_host)
    apply_hook_runtime_guard_wiring(nexus_loop, effective_environment)
    from intergrax.applications._shared.observability_wiring import (
        wire_observability_event_subscriptions,
    )

    wire_observability_event_subscriptions(
        nexus_loop.event_bus,
        effective_environment.observability_profile,
    )
    from intergrax.applications._shared.reliability_wiring import (
        apply_reliability_governance_wiring,
    )

    apply_reliability_governance_wiring(nexus_loop, effective_environment)
    from intergrax.applications._shared.diagnostic_runtime_wiring import (
        wire_terminal_execution_diagnostics,
    )

    diagnostic_wiring = wire_terminal_execution_diagnostics(
        env=effective_environment,
        env_wiring=env_wiring,
        observability=observability,
        nexus_loop=nexus_loop,
    )
    control_plane_governance = build_harness_control_plane_governance(
        effective_environment,
        mutation_authorization_boundary=mutation_authorization_boundary,
    )
    execution = build_environment_host_task_execution(
        nexus_loop,
        effective_environment,
        pinning_dependencies=EffectiveProfileExecutionPinningDependencies(
            revision_store=profile_persistence.revision_store,
            pinning_store=profile_persistence.pinning_store,
            active_store=profile_persistence.active_store,
            scope=revision_scope,
        ),
    )
    return HarnessHostRuntime(
        manifest=resolved_manifest,
        environment=effective_environment,
        env_wiring=env_wiring,
        registry=resolved_registry,
        registry_projection_evidence=registry_evidence,
        observability=observability,
        reliability=reliability_wiring,
        security=security_wiring,
        guardrail=guardrail_wiring,
        cost=cost_wiring,
        evaluation=evaluation_wiring,
        diagnostic_wiring=diagnostic_wiring,
        execution=execution,
        _legacy_composition=HarnessHostLegacyComposition(nexus_loop=nexus_loop),
        application_host=application_host,
        agent_checkpoint_store=resolved_agent_checkpoint_store,
        compensation_queue_store=resolved_compensation_queue_store,
        boundary_event_buffer=boundary_event_buffer,
        control_plane_governance=control_plane_governance,
        profile_resolution=profile_resolution,
        effective_profile_revision=effective_profile_revision,
        effective_profile_revision_store=profile_persistence.revision_store,
        effective_profile_pinning_store=profile_persistence.pinning_store,
        effective_profile_active_store=profile_persistence.active_store,
    )
