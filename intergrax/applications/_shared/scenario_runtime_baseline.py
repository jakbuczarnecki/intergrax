# © Artur Czarnecki. All rights reserved.

"""Shared Nexus-backed scenario runtime baseline (SCENARIO-PLATFORM-3A)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from intergrax.runtime.execution.host_validation_composition import NexusValidationEngine

from intergrax.applications._shared.cost_assembly_resolver import assert_cost_assembly_valid
from intergrax.applications._shared.cost_wiring import wire_application_cost
from intergrax.applications._shared.decision_wiring import (
    application_decision_wiring_spec_from_environment,
    resolve_application_decision_agent_id,
    wire_application_decision,
)
from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_for_application_host,
)
from intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring import (
    resolve_harness_host_meaningful_side_effect_authorization_wiring,
)
from intergrax.collaborative_work.persistence import (
    CollaborativeWorkMaterializedRepositories,
    open_sqlite_collaborative_work_repositories,
)
from intergrax.applications._shared.diagnostic_assembly_resolver import (
    DiagnosticAssemblyError,
    DiagnosticWiring,
)
from intergrax.applications._shared.environment_wiring import (
    ApplicationEnvironmentWiring,
    wire_application_environment,
)
from intergrax.applications._shared.evaluation_assembly_resolver import (
    assert_evaluation_assembly_valid,
)
from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications._shared.guardrail_wiring import (
    ApplicationGuardrailWiring,
    wire_application_guardrail,
)
from intergrax.applications._shared.llm_resolver import resolve_environment_llm_adapter
from intergrax.applications._shared.host_orchestration_backend_spec_builder import (
    build_host_orchestration_loop_init_spec_from_environment,
)
from intergrax.runtime.execution.application_host_orchestration_composition import (
    compose_application_host_orchestration_session,
)
from intergrax.runtime.execution.application_host_orchestration_session import (
    ApplicationHostOrchestrationSession,
)
from intergrax.runtime.execution.host_orchestration_assembly_validation import (
    apply_host_orchestration_application_runtime_wiring,
    assert_host_orchestration_application_assembly,
)
from intergrax.runtime.execution.scenario_host_diagnostic_wiring import (
    wire_scenario_terminal_execution_diagnostics,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
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
    wire_application_reliability,
)
from intergrax.applications._shared.security_assembly_resolver import (
    assert_security_assembly_valid,
)
from intergrax.applications._shared.security_wiring import (
    ApplicationSecurityWiring,
    wire_application_security,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications._shared.task_memory_wiring import wire_task_memory_from_profile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.decision_flow import DecisionFlowGate
from intergrax.applications._shared.diagnostic_composition import (
    DiagnosticCompositionOverrides,
)
from intergrax.applications._shared.harness_host_task_execution_wiring import (
    build_harness_host_task_execution_governance,
)
from intergrax.applications._shared.profile_resolution.execution_effective_profile_provenance_reader import (
    PinningStoreExecutionEffectiveProfileProvenanceReader,
)
from intergrax.applications._shared.profile_resolution.host_effective_profile_execution_wiring import (
    wire_host_effective_profile_execution,
)
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)
from intergrax.runtime.observability.qualification_runtime_trace import (
    DeferredPersistedTraceFinalize,
)
from intergrax.contracts.host_observability_stores import HostObservabilityStores
from intergrax.runtime.execution.host_observability_composition import wire_host_observability
from intergrax.tools.registry import ToolRegistry
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult

from intergrax.applications._shared.scenario_runtime_profiles import (
    ScenarioRuntimeMode,
    ScenarioRuntimeWorkspace,
)


__all__ = [
    "ScenarioExecutionRequest",
    "ScenarioRuntimeBuildError",
    "ScenarioRuntimeComposition",
    "ScenarioRuntimeExecutionResult",
    "ScenarioRuntimeMode",
    "ScenarioRuntimeWorkspace",
    "build_scenario_runtime_from_environment",
    "ScenarioLabAgentRegistration",
    "build_scenario_lab_agent_registry",
    "execute_scenario_task",
    "rebuild_scenario_runtime_from_composition",
    "rewire_scenario_decision_wiring",
    "validate_scenario_tenant_id",
]


class ScenarioRuntimeBuildError(RuntimeError):
    """Raised when scenario runtime composition cannot satisfy platform invariants."""


@dataclass(frozen=True, slots=True)
class ScenarioLabAgentRegistration:
    """Typed LAB roster entry assembled before scenario runtime composition."""

    agent: Agent
    contract: AgentContract | None = None


def build_scenario_lab_agent_registry(
    *registrations: ScenarioLabAgentRegistration,
) -> AgentRegistry:
    """Tier-1 baseline roster construction for lab scenario runtime composition."""
    registry = AgentRegistry()
    for registration in registrations:
        registry.register(
            registration.agent,
            contract=registration.contract,
        )
    return registry


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeComposition:
    """Immutable execution-semantic scenario runtime artifacts."""

    environment: ApplicationEnvironmentProfile
    env_wiring: ApplicationEnvironmentWiring
    observability: HostObservabilityStores
    registry: AgentRegistry
    host_execution: HostTaskExecutionPort
    orchestration_session: ApplicationHostOrchestrationSession
    tenant_id: str
    security_wiring: ApplicationSecurityWiring
    guardrail_wiring: ApplicationGuardrailWiring
    diagnostic_wiring: DiagnosticWiring
    decision_flow_gate: DecisionFlowGate[AgentExecutionResult] | None = None
    workspace: ScenarioRuntimeWorkspace | None = None
    runtime_mode: ScenarioRuntimeMode | None = None

    @property
    def terminal_diagnostic_trigger_attached(self) -> bool:
        return self.diagnostic_wiring.attached

    @property
    def has_runtime_event_store(self) -> bool:
        return self.observability.runtime_event_store is not None

    @property
    def has_terminal_diagnostic_trigger(self) -> bool:
        return self.diagnostic_wiring.attached


@dataclass(frozen=True, slots=True)
class ScenarioExecutionRequest:
    """Typed scenario task intake at the platform boundary."""

    tenant_id: str
    message: str
    user_id: str = "scenario-user"
    capability: str | None = None
    task_id: TaskId | None = None
    hold_persisted_trace_finalize: bool = False


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeExecutionResult:
    """Minimal platform execution envelope for scenario proofs."""

    task_result: TaskResult
    task_id: TaskId
    run_id: RunId
    tenant_id: str
    deferred_persisted_trace_finalize: DeferredPersistedTraceFinalize | None = None

    @property
    def authoritative_decision_exposure(self):
        return self.task_result.authoritative_decision_exposure


def validate_scenario_tenant_id(tenant_id: str) -> str:
    """Validate explicit tenant id before scenario execution or wiring."""
    if not isinstance(tenant_id, str):
        raise ValueError("tenant_id must be a string")
    if tenant_id != tenant_id.strip():
        raise ValueError("tenant_id must not have leading or trailing whitespace")
    if not tenant_id:
        raise ValueError("tenant_id must be non-empty")
    return tenant_id


def _scenario_allows_lab_manifest_fallback(
    environment: ApplicationEnvironmentProfile,
) -> bool:
    """LAB-only posture: balanced lab hosts may synthesize a manifest; strict/product may not."""
    return (
        environment.application_profile is ApplicationProfile.LAB
        and environment.execution_mode is not ExecutionMode.STRICT
    )


def _resolve_scenario_manifest(
    environment: ApplicationEnvironmentProfile,
    manifest: ApplicationManifest | None,
) -> ApplicationManifest:
    if manifest is not None:
        resolved = manifest
    elif _scenario_allows_lab_manifest_fallback(environment):
        resolved = _scenario_lab_manifest(environment)
    else:
        raise ScenarioRuntimeBuildError(
            "explicit ApplicationManifest is required for strict or production-attached "
            "scenario environments"
        )
    if resolved.environment is None:
        resolved = resolved.model_copy(update={"environment": environment})
    return resolved


def _scenario_lab_manifest(environment: ApplicationEnvironmentProfile) -> ApplicationManifest:
    safe_id = environment.profile_id.replace(".", "_").replace("-", "_")[:48]
    return ApplicationManifest.lab(
        app_id=f"scenario_{safe_id}",
        name=f"Scenario Runtime {environment.profile_id}",
        route_prefix=f"/v1/scenario/{safe_id}",
        env_prefix=f"SCENARIO_{safe_id.upper()}_",
        agents=[],
        environment=environment,
    )


def _scenario_execution_continuation_state_store(
    environment: ApplicationEnvironmentProfile,
) -> object | None:
    """Restart-qualified durable continuation store for strict production-attached scenarios."""
    if environment.execution_mode.value != "strict":
        return None
    from intergrax.runtime.execution.continuation.persistence import (
        ExecutionContinuationDurableBacking,
        backing_execution_continuation_state_store,
        execution_continuation_state_store_from_durable_export,
        export_durable_continuation_state,
    )

    backing = ExecutionContinuationDurableBacking()
    backing_execution_continuation_state_store(backing)
    export = export_durable_continuation_state(backing)
    return execution_continuation_state_store_from_durable_export(export)


def _scenario_collaborative_work_repositories(
    environment: ApplicationEnvironmentProfile,
    *,
    runtime_events_db_path: Path | None,
) -> CollaborativeWorkMaterializedRepositories | None:
    """Isolate collaborative-work SQLite from other scenario DB files under the same root."""
    if environment.execution_mode.value != "strict":
        return None
    if runtime_events_db_path is None:
        return None
    cw_db_path = runtime_events_db_path.parent / "collaborative_work.db"
    return open_sqlite_collaborative_work_repositories(str(cw_db_path))


def _resolve_observability_stores(
    environment: ApplicationEnvironmentProfile,
    *,
    trace_db_path: Path | None,
    runtime_events_db_path: Path | None,
    use_in_memory_trace: bool,
) -> HostObservabilityStores:
    if use_in_memory_trace:
        return wire_host_observability(
            trace_db_path=trace_db_path,
            runtime_events_db_path=runtime_events_db_path,
            integration_profile=environment.integration_profile,
            use_in_memory_trace=True,
            enable_runtime_events=runtime_events_db_path is not None,
        )
    wiring = wire_application_observability(
        environment,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        integration_profile=environment.integration_profile,
    )
    assert_observability_assembly_valid(wiring, environment)
    return wiring.stores


def rewire_scenario_decision_wiring(
    composition: ScenarioRuntimeComposition,
    *,
    validation_engine: NexusValidationEngine | None = None,
) -> ScenarioRuntimeComposition:
    """Rebuild scenario runtime with Decision flow wiring from the current environment profile."""
    return rebuild_scenario_runtime_from_composition(
        composition,
        environment=composition.environment,
        validation_engine=validation_engine,
    )


def rebuild_scenario_runtime_from_composition(
    composition: ScenarioRuntimeComposition,
    *,
    environment: ApplicationEnvironmentProfile,
    validation_engine: NexusValidationEngine | None = None,
    manifest: ApplicationManifest | None = None,
    conformance_check: bool = True,
) -> ScenarioRuntimeComposition:
    """Rebuild Nexus-backed scenario runtime while preserving registry and storage paths."""
    return build_scenario_runtime_from_environment(
        environment=environment,
        registry=composition.registry,
        tenant_id=composition.tenant_id,
        runtime_events_db_path=composition.observability.runtime_events_db_path,
        trace_db_path=composition.observability.trace_db_path,
        manifest=manifest,
        use_in_memory_trace=False,
        require_runtime_event_persistence=True,
        workspace=composition.workspace,
        runtime_mode=composition.runtime_mode,
        conformance_check=conformance_check,
        validation_engine=validation_engine,
        application_tool_registry=composition.env_wiring.tool_wiring.registry,
    )


def build_scenario_runtime_from_environment(
    *,
    environment: ApplicationEnvironmentProfile,
    registry: AgentRegistry,
    tenant_id: str,
    runtime_events_db_path: Path | None = None,
    trace_db_path: Path | None = None,
    document_store: Any | None = None,
    settings: Any = None,
    manifest: ApplicationManifest | None = None,
    use_in_memory_trace: bool = False,
    require_runtime_event_persistence: bool = True,
    workspace: ScenarioRuntimeWorkspace | None = None,
    runtime_mode: ScenarioRuntimeMode | None = None,
    conformance_check: bool = True,
    validation_engine: NexusValidationEngine | None = None,
    application_tool_registry: ToolRegistry | None = None,
) -> ScenarioRuntimeComposition:
    """
    Compose a lighter Nexus-backed scenario runtime from platform primitives.

    Reuses environment, observability, reliability, security, guardrail, and Nexus
    factory wiring without HarnessHostRuntime hosting/control-plane surfaces.
    """
    resolved_tenant_id = validate_scenario_tenant_id(tenant_id)
    resolved_manifest = _resolve_scenario_manifest(environment, manifest)
    resolved_document_store = document_store if isinstance(document_store, DocumentStore) else None
    host_profile = wire_host_effective_profile_execution(
        environment,
        application_id=resolved_manifest.app_id,
        tenant_id=resolved_tenant_id,
        document_store=resolved_document_store,
    )
    effective_environment = host_profile.effective_environment

    env_wiring = wire_application_environment(
        resolved_manifest,
        effective_environment,
        settings=settings,
        tenant_id=resolved_tenant_id,
        document_store=document_store,
        conformance_check=conformance_check,
        application_tool_registry=application_tool_registry,
    )
    profile_provenance_reader = PinningStoreExecutionEffectiveProfileProvenanceReader(
        host_profile.persistence.pinning_store,
    )
    env_wiring = replace(
        env_wiring,
        composition=replace(
            env_wiring.composition,
            diagnostic_composition_overrides=DiagnosticCompositionOverrides(
                execution_effective_profile_provenance_reader=profile_provenance_reader,
            ),
        ),
    )
    observability = _resolve_observability_stores(
        effective_environment,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        use_in_memory_trace=use_in_memory_trace,
    )
    if require_runtime_event_persistence and observability.runtime_event_store is None:
        raise ScenarioRuntimeBuildError(
            "RuntimeEvent persistence is required but no runtime event store was created. "
            "Provide runtime_events_db_path or enable observability runtime events."
        )

    reliability_wiring = wire_application_reliability(effective_environment)
    assert_reliability_assembly_valid(reliability_wiring, effective_environment)
    cost_wiring = wire_application_cost(effective_environment)
    assert_cost_assembly_valid(cost_wiring, effective_environment)
    security_wiring = wire_application_security(effective_environment)
    assert_security_assembly_valid(security_wiring, effective_environment)
    guardrail_wiring = wire_application_guardrail(effective_environment)
    evaluation_wiring = wire_application_evaluation(effective_environment)
    assert_evaluation_assembly_valid(evaluation_wiring, effective_environment)
    decision_spec = application_decision_wiring_spec_from_environment(effective_environment)
    decision_wiring = wire_application_decision(
        registry=registry,
        agent_id=resolve_application_decision_agent_id(registry, effective_environment),
        spec=decision_spec,
        environment=effective_environment,
    )
    task_memory = wire_task_memory_from_profile(effective_environment)
    scenario_collaborative_work = _scenario_collaborative_work_repositories(
        effective_environment,
        runtime_events_db_path=runtime_events_db_path,
    )
    meaningful_side_effect_wiring = (
        resolve_harness_host_meaningful_side_effect_authorization_wiring(
            effective_environment,
            collaborative_work_repositories=scenario_collaborative_work,
            decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
            runtime_event_persistence=observability.runtime_event_store,
        )
    )
    declarative_tool_invoker = build_declarative_invoker_for_application_host(
        env_wiring.tool_wiring,
        effective_environment,
        manifest=resolved_manifest,
        agent_registry=registry,
        tenant_id=resolved_tenant_id,
        idempotency_store=reliability_wiring.idempotency_store,
        meaningful_side_effect_authorization=(
            meaningful_side_effect_wiring.authorization_port
        ),
    )

    orchestration_spec = build_host_orchestration_loop_init_spec_from_environment(
        registry,
        env=effective_environment,
        child_context_inheritance=host_profile.child_context_inheritance,
        trace_store=observability.trace_store,
        idempotency_store=reliability_wiring.idempotency_store,
        declarative_tool_invoker=declarative_tool_invoker,
        runtime_events_db_path=observability.runtime_events_db_path,
        task_memory_store=task_memory.store,
        task_memory_db_path=task_memory.db_path,
        shadow_manager=env_wiring.shadow_manager,
        sandbox_manager=env_wiring.sandbox_manager,
        llm_adapter=resolve_environment_llm_adapter(
            effective_environment,
            tenant_id=resolved_tenant_id,
        ),
        runtime_event_bus=env_wiring.composition.runtime_event_bus,
        security_wiring=security_wiring,
        guardrail_wiring=guardrail_wiring,
        decision_wiring=decision_wiring,
        run_budget=cost_wiring.run_budget,
        validation_engine=validation_engine,
        document_store=document_store,
        execution_continuation_state_store=_scenario_execution_continuation_state_store(
            effective_environment,
        ),
    )
    harness_execution_governance = build_harness_host_task_execution_governance()
    orchestration_session, materialization = compose_application_host_orchestration_session(
        registry,
        orchestration_spec,
        effective_environment,
        revision_admission=host_profile.revision_admission,
        root_authority_admission=harness_execution_governance.root_authority_admission,
        admit_root_governance_identity=harness_execution_governance.admit_root_governance_identity,
    )
    assert_host_orchestration_application_assembly(
        materialization,
        env=effective_environment,
        security_wiring=security_wiring,
        guardrail_wiring=guardrail_wiring,
    )
    apply_host_orchestration_application_runtime_wiring(
        materialization,
        env=effective_environment,
    )
    host_execution = orchestration_session.host_execution

    try:
        diagnostic_wiring = wire_scenario_terminal_execution_diagnostics(
            materialization=materialization,
            env=effective_environment,
            env_wiring=env_wiring,
            observability=observability,
            scenario_runtime_mode=runtime_mode,
        )
    except DiagnosticAssemblyError as exc:
        raise ScenarioRuntimeBuildError(str(exc)) from exc

    return ScenarioRuntimeComposition(
        environment=effective_environment,
        env_wiring=env_wiring,
        observability=observability,
        registry=registry,
        host_execution=host_execution,
        orchestration_session=orchestration_session,
        tenant_id=resolved_tenant_id,
        security_wiring=security_wiring,
        guardrail_wiring=guardrail_wiring,
        diagnostic_wiring=diagnostic_wiring,
        decision_flow_gate=decision_wiring.gate,
        workspace=workspace,
        runtime_mode=runtime_mode,
    )


async def execute_scenario_task(
    composition: ScenarioRuntimeComposition,
    request: ScenarioExecutionRequest,
) -> ScenarioRuntimeExecutionResult:
    """Execute one scenario task through the composed Nexus loop."""
    tenant_id = validate_scenario_tenant_id(request.tenant_id)
    if tenant_id != composition.tenant_id:
        raise ValueError("request tenant_id must match scenario runtime tenant_id")

    task_kwargs: dict[str, Any] = {
        "tenant_id": tenant_id,
        "user_id": request.user_id,
        "message": request.message,
    }
    if request.task_id is not None:
        task_kwargs["task_id"] = request.task_id
    if request.capability is not None:
        task_kwargs["context"] = TaskContext(capability=request.capability)

    task = Task(**task_kwargs)
    composition.orchestration_session.trace_lifecycle.set_hold_persisted_trace_finalize(
        request.hold_persisted_trace_finalize,
    )
    try:
        task_result = await composition.host_execution.execute(task)
    finally:
        composition.orchestration_session.trace_lifecycle.set_hold_persisted_trace_finalize(False)
    deferred_finalize = (
        composition.orchestration_session.trace_lifecycle.take_deferred_persisted_trace_finalize()
    )
    return ScenarioRuntimeExecutionResult(
        task_result=task_result,
        task_id=task.task_id,
        run_id=task_result.run_id,
        tenant_id=tenant_id,
        deferred_persisted_trace_finalize=deferred_finalize,
    )
