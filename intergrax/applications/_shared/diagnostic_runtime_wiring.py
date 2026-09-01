# © Artur Czarnecki. All rights reserved.

"""Shared platform diagnostic runtime composition for Tier-3 harness hosts (ONE-SPINE-3)."""

from __future__ import annotations

from intergrax.applications._shared.diagnostic_read_wiring import (
    HostDiagnosticReadDependencies,
    resolve_host_diagnostic_read_dependencies,
)
from intergrax.applications._shared.diagnostic_assembly_resolver import (
    DiagnosticWiring,
    assert_diagnostic_assembly_valid,
    resolve_central_diagnostics_required,
)
from intergrax.applications._shared.diagnostic_cursor_secret import (
    resolve_problem_list_cursor_secret,
)
from intergrax.applications._shared.environment_wiring import ApplicationEnvironmentWiring
from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_problem_grouping_feature_projector import (
    DiagnosticProblemGroupingFeatureProjector,
)
from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
    wire_problem_occurrence_persistence,
)
from intergrax.runtime.diagnostics.document_store_problem_persistence import wire_problem_persistence
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingEngine,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTrigger,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores
from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)


def resolve_host_diagnostic_runtime_dependencies(
    *,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
) -> HostDiagnosticReadDependencies | None:
    """
    Resolve shared diagnostic persistence for runtime write orchestration.

    Returns ``None`` when the host lacks required platform document-store capabilities.
    """
    build_context = env_wiring.build_context
    wiring_context = build_context.tool_wiring_context
    if wiring_context is None or wiring_context.document_store is None:
        return None
    document_store = assert_conditional_document_store(wiring_context.document_store)

    runtime_events = observability.runtime_event_store
    if runtime_events is None:
        return None

    return HostDiagnosticReadDependencies(
        problem_persistence=wire_problem_persistence(
            document_store=document_store,
            list_cursor_secret=resolve_problem_list_cursor_secret(),
        ),
        occurrence_persistence=wire_problem_occurrence_persistence(
            document_store=document_store,
            occurrence_cursor_secret=resolve_problem_list_cursor_secret(),
        ),
        runtime_event_persistence=runtime_events,
        causal_evidence_persistence=wire_causal_evidence_persistence(
            document_store=document_store,
        ),
    )


def build_diagnostic_orchestrator(
    dependencies: HostDiagnosticReadDependencies,
) -> DiagnosticOrchestrator:
    """Construct canonical ``DiagnosticOrchestrator`` over shared platform persistence."""
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return DiagnosticOrchestrator(
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=dependencies.runtime_event_persistence,
            causal_evidence=dependencies.causal_evidence_persistence,
        ),
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=ProblemGroupingEngine(
            registry,
            feature_projector=DiagnosticProblemGroupingFeatureProjector(),
        ),
        problem_lifecycle_engine=ProblemLifecycleEngine(
            dependencies.problem_persistence,
            dependencies.occurrence_persistence,
        ),
    )


def build_terminal_execution_diagnostic_trigger(
    dependencies: HostDiagnosticReadDependencies,
) -> TerminalExecutionDiagnosticTrigger:
    """Construct production terminal diagnostic trigger over shared orchestrator."""
    return TerminalExecutionDiagnosticTrigger(build_diagnostic_orchestrator(dependencies))


def try_build_terminal_execution_diagnostic_trigger(
    *,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
) -> TerminalExecutionDiagnosticTrigger | None:
    """Best-effort runtime trigger when required platform storage is available."""
    dependencies = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=env_wiring,
        observability=observability,
    )
    if dependencies is None:
        return None
    return build_terminal_execution_diagnostic_trigger(dependencies)


def _diagnostic_prerequisite_gaps(
    *,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
) -> tuple[bool, bool]:
    wiring_context = env_wiring.build_context.tool_wiring_context
    missing_document_store = wiring_context is None or wiring_context.document_store is None
    missing_runtime_events = observability.runtime_event_store is None
    return missing_document_store, missing_runtime_events


def wire_terminal_execution_diagnostics(
    *,
    env: ApplicationEnvironmentProfile,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
    nexus_loop: NexusLoop,
    scenario_runtime_mode: object | None = None,
) -> DiagnosticWiring:
    """
    Policy-aware terminal diagnostic composition over the canonical orchestrator spine.

    When diagnostics are required, missing prerequisites fail closed.
    """
    required = resolve_central_diagnostics_required(
        env,
        scenario_runtime_mode=scenario_runtime_mode,  # type: ignore[arg-type]
    )
    missing_document_store, missing_runtime_events = _diagnostic_prerequisite_gaps(
        env_wiring=env_wiring,
        observability=observability,
    )
    terminal_diagnostic_trigger = try_build_terminal_execution_diagnostic_trigger(
        env_wiring=env_wiring,
        observability=observability,
    )
    attached = terminal_diagnostic_trigger is not None
    assert_diagnostic_assembly_valid(
        required=required,
        attached=attached,
        missing_document_store=missing_document_store,
        missing_runtime_events=missing_runtime_events,
    )
    if attached:
        nexus_loop.attach_terminal_diagnostic_trigger(terminal_diagnostic_trigger)
    return DiagnosticWiring(required=required, attached=attached)


def resolve_host_terminal_execution_diagnostic_trigger(
    runtime: HarnessHostRuntime,
) -> TerminalExecutionDiagnosticTrigger:
    """Resolve production terminal diagnostic trigger from harness host runtime wiring."""
    return build_terminal_execution_diagnostic_trigger(
        resolve_host_diagnostic_read_dependencies(runtime),
    )


__all__ = [
    "build_diagnostic_orchestrator",
    "build_terminal_execution_diagnostic_trigger",
    "resolve_host_diagnostic_runtime_dependencies",
    "resolve_host_terminal_execution_diagnostic_trigger",
    "try_build_terminal_execution_diagnostic_trigger",
    "wire_terminal_execution_diagnostics",
]
