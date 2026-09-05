# © Artur Czarnecki. All rights reserved.

"""Shared Nexus-backed scenario runtime baseline (SCENARIO-PLATFORM-3A)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.applications._shared.cost_assembly_resolver import assert_cost_assembly_valid
from intergrax.applications._shared.cost_wiring import wire_application_cost
from intergrax.applications._shared.decision_wiring import (
    application_decision_wiring_spec_from_environment,
    apply_application_decision_wiring,
    resolve_application_decision_agent_id,
    wire_application_decision,
)
from intergrax.applications._shared.declarative_tool_wiring import (
    build_declarative_invoker_from_tool_wiring,
)
from intergrax.applications._shared.diagnostic_assembly_resolver import (
    DiagnosticAssemblyError,
    DiagnosticWiring,
)
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    wire_terminal_execution_diagnostics,
)
from intergrax.applications._shared.environment_wiring import (
    ApplicationEnvironmentWiring,
    wire_application_environment,
)
from intergrax.applications._shared.evaluation_assembly_resolver import (
    assert_evaluation_assembly_valid,
)
from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications._shared.guardrail_assembly_resolver import (
    assert_guardrail_assembly_valid,
)
from intergrax.applications._shared.guardrail_wiring import (
    ApplicationGuardrailWiring,
    wire_application_guardrail,
)
from intergrax.applications._shared.llm_resolver import resolve_environment_llm_adapter
from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications._shared.observability_assembly_resolver import (
    assert_observability_assembly_valid,
)
from intergrax.applications._shared.observability_wiring import (
    wire_application_observability,
    wire_observability_event_subscriptions,
)
from intergrax.applications._shared.reliability_assembly_resolver import (
    assert_reliability_assembly_valid,
)
from intergrax.applications._shared.reliability_wiring import (
    apply_reliability_governance_wiring,
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
from intergrax.contracts.execution_identity import RunId, TaskId, mint_run_id
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import (
    NexusObservabilityStores,
    wire_nexus_observability,
)
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
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
    "build_scenario_lab_agent_registry",
    "execute_scenario_task",
    "rebuild_scenario_runtime_from_composition",
    "rewire_scenario_decision_wiring",
    "validate_scenario_tenant_id",
]


class ScenarioRuntimeBuildError(RuntimeError):
    """Raised when scenario runtime composition cannot satisfy platform invariants."""


def build_scenario_lab_agent_registry() -> AgentRegistry:
    """Tier-1 baseline roster construction for lab scenario runtime composition."""
    return AgentRegistry()


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeComposition:
    """Immutable Nexus-backed scenario runtime artifacts."""

    environment: ApplicationEnvironmentProfile
    env_wiring: ApplicationEnvironmentWiring
    observability: NexusObservabilityStores
    registry: AgentRegistry
    nexus_loop: NexusLoop
    tenant_id: str
    security_wiring: ApplicationSecurityWiring
    guardrail_wiring: ApplicationGuardrailWiring
    diagnostic_wiring: DiagnosticWiring
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


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeExecutionResult:
    """Minimal platform execution envelope for scenario proofs."""

    task_result: TaskResult
    task_id: TaskId
    run_id: RunId
    tenant_id: str


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


def _resolve_observability_stores(
    environment: ApplicationEnvironmentProfile,
    *,
    trace_db_path: Path | None,
    runtime_events_db_path: Path | None,
    use_in_memory_trace: bool,
) -> NexusObservabilityStores:
    if use_in_memory_trace:
        return wire_nexus_observability(
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
) -> None:
    """Reapply Decision flow wiring and validation engine from the current environment profile."""
    environment = composition.environment
    decision_spec = application_decision_wiring_spec_from_environment(environment)
    decision_wiring = wire_application_decision(
        registry=composition.registry,
        agent_id=resolve_application_decision_agent_id(composition.registry, environment),
        spec=decision_spec,
    )
    if validation_engine is not None:
        composition.nexus_loop.apply_validation_engine(validation_engine)
    apply_application_decision_wiring(composition.nexus_loop, decision_wiring)


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

    env_wiring = wire_application_environment(
        resolved_manifest,
        environment,
        settings=settings,
        tenant_id=resolved_tenant_id,
        document_store=document_store,
        conformance_check=conformance_check,
        application_tool_registry=application_tool_registry,
    )
    observability = _resolve_observability_stores(
        environment,
        trace_db_path=trace_db_path,
        runtime_events_db_path=runtime_events_db_path,
        use_in_memory_trace=use_in_memory_trace,
    )
    if require_runtime_event_persistence and observability.runtime_event_store is None:
        raise ScenarioRuntimeBuildError(
            "RuntimeEvent persistence is required but no runtime event store was created. "
            "Provide runtime_events_db_path or enable observability runtime events."
        )

    reliability_wiring = wire_application_reliability(environment)
    assert_reliability_assembly_valid(reliability_wiring, environment)
    cost_wiring = wire_application_cost(environment)
    assert_cost_assembly_valid(cost_wiring, environment)
    security_wiring = wire_application_security(environment)
    assert_security_assembly_valid(security_wiring, environment)
    guardrail_wiring = wire_application_guardrail(environment)
    evaluation_wiring = wire_application_evaluation(environment)
    assert_evaluation_assembly_valid(evaluation_wiring, environment)
    decision_spec = application_decision_wiring_spec_from_environment(environment)
    decision_wiring = wire_application_decision(
        registry=registry,
        agent_id=resolve_application_decision_agent_id(registry, environment),
        spec=decision_spec,
    )
    task_memory = wire_task_memory_from_profile(environment)
    declarative_tool_invoker = build_declarative_invoker_from_tool_wiring(env_wiring.tool_wiring)

    nexus_loop = build_nexus_loop_from_environment(
        registry,
        env=environment,
        trace_store=observability.trace_store,
        idempotency_store=reliability_wiring.idempotency_store,
        declarative_tool_invoker=declarative_tool_invoker,
        runtime_events_db_path=observability.runtime_events_db_path,
        task_memory_store=task_memory.store,
        task_memory_db_path=task_memory.db_path,
        shadow_manager=env_wiring.shadow_manager,
        sandbox_manager=env_wiring.sandbox_manager,
        llm_adapter=resolve_environment_llm_adapter(environment, tenant_id=resolved_tenant_id),
        runtime_event_bus=env_wiring.build_context.runtime_event_bus,
        security_wiring=security_wiring,
        guardrail_wiring=guardrail_wiring,
        decision_wiring=decision_wiring,
        run_budget=cost_wiring.run_budget,
        validation_engine=validation_engine,
        document_store=document_store,
    )
    assert_security_assembly_valid(security_wiring, environment, nexus=nexus_loop)
    assert_guardrail_assembly_valid(guardrail_wiring, environment, nexus=nexus_loop)

    wire_observability_event_subscriptions(
        nexus_loop.event_bus,
        environment.observability_profile,
    )
    apply_reliability_governance_wiring(nexus_loop, environment)

    try:
        diagnostic_wiring = wire_terminal_execution_diagnostics(
            env=environment,
            env_wiring=env_wiring,
            observability=observability,
            nexus_loop=nexus_loop,
            scenario_runtime_mode=runtime_mode,
        )
    except DiagnosticAssemblyError as exc:
        raise ScenarioRuntimeBuildError(str(exc)) from exc

    return ScenarioRuntimeComposition(
        environment=environment,
        env_wiring=env_wiring,
        observability=observability,
        registry=registry,
        nexus_loop=nexus_loop,
        tenant_id=resolved_tenant_id,
        security_wiring=security_wiring,
        guardrail_wiring=guardrail_wiring,
        diagnostic_wiring=diagnostic_wiring,
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
    run_id = mint_run_id()
    task_result = await UnifiedTaskRunner(composition.nexus_loop).run_task(task, run_id=run_id)
    return ScenarioRuntimeExecutionResult(
        task_result=task_result,
        task_id=task.task_id,
        run_id=run_id,
        tenant_id=tenant_id,
    )
