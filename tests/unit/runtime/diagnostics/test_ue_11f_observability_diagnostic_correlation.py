# © Artur Czarnecki. All rights reserved.

"""UE-11F — observability evidence → terminal diagnostic correlation qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.contracts.execution_identity import EventId, RunId, TaskId, mint_run_id
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticOrchestrationRequest,
    DiagnosticOrchestrationResult,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_problem_grouping_feature_projector import (
    DiagnosticProblemGroupingFeatureProjector,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.in_memory_problem_persistence import InMemoryProblemPersistence
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingEngine,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTrigger,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "ue-11f-tenant"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIAGNOSTICS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"

_FORBIDDEN_LIFECYCLE_OWNERSHIP_CALLS = frozenset(
    {
        "mint_execution_id",
        "mint_root_execution_identity",
        "bind_root_execution_budget",
        "resolve_root_execution_context",
        "ExecutionRuntime",
        "StrategyExecutionRouter",
    }
)
_BIND_ACTIVE_EXECUTION_IDENTITY_ALLOWLIST = frozenset(
    {"intergrax/runtime/diagnostics/terminal_execution_diagnostic_bridge.py"},
)


class RecordingExecutionReconstructor(ExecutionReconstructor):
    """Records production reconstruction scopes while delegating to canonical persistence."""

    def __init__(
        self,
        *,
        runtime_events: InMemoryRuntimeEventStore,
        causal_evidence: InMemoryCausalEvidencePersistence,
    ) -> None:
        super().__init__(runtime_events=runtime_events, causal_evidence=causal_evidence)
        self.invocations: list[tuple[str, TaskId, RunId]] = []

    def reconstruct_execution(
        self,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        *,
        initial_limit: int = 1000,
        max_limit: int = 1_000_000,
    ):
        self.invocations.append((tenant_id, task_id, run_id))
        return super().reconstruct_execution(
            tenant_id,
            task_id,
            run_id,
            initial_limit=initial_limit,
            max_limit=max_limit,
        )


class RecordingDiagnosticOrchestrator(DiagnosticOrchestrator):
    """Records production orchestration requests at the public orchestrator boundary."""

    def __init__(
        self,
        *,
        execution_reconstructor: ExecutionReconstructor,
        lifecycle_analyzer: LifecycleAnomalyAnalyzer,
        assessment_builder: DiagnosticAssessmentBuilder,
        grouping_engine: ProblemGroupingEngine,
        problem_lifecycle_engine: ProblemLifecycleEngine,
    ) -> None:
        super().__init__(
            execution_reconstructor=execution_reconstructor,
            lifecycle_analyzer=lifecycle_analyzer,
            assessment_builder=assessment_builder,
            grouping_engine=grouping_engine,
            problem_lifecycle_engine=problem_lifecycle_engine,
        )
        self.recorded_requests: list[DiagnosticOrchestrationRequest] = []
        self.recorded_results: list[DiagnosticOrchestrationResult] = []

    def run(self, request: DiagnosticOrchestrationRequest) -> DiagnosticOrchestrationResult:
        self.recorded_requests.append(request)
        result = super().run(request)
        self.recorded_results.append(result)
        return result


def _build_grouping_engine() -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return ProblemGroupingEngine(
        registry,
        feature_projector=DiagnosticProblemGroupingFeatureProjector(),
    )


def _inject_identity_preserving_violation(
    runtime_store: InMemoryRuntimeEventStore,
    *,
    violating_event_type: RuntimeEventType,
):
    def _handler(event: RuntimeEvent) -> None:
        if event.event_type is not RuntimeEventType.TASK_COMPLETED:
            return
        runtime_store.append(
            sample_runtime_event(
                tenant_id=event.tenant_id,
                task_id=event.task_id,
                run_id=event.run_id,
                attempt_id=event.attempt_id,
                execution_id=event.execution_id,
            ).model_copy(update={"event_type": violating_event_type}),
            tenant_id=event.tenant_id,
        )

    return _handler


def _build_ue_11f_nexus_stack() -> tuple[
    NexusLoop,
    InMemoryRuntimeEventStore,
    RecordingDiagnosticOrchestrator,
    RecordingExecutionReconstructor,
]:
    runtime_store = InMemoryRuntimeEventStore()
    stores = wire_nexus_observability(
        use_in_memory_trace=True,
        runtime_event_store=runtime_store,
    )
    causal_store = InMemoryCausalEvidencePersistence()
    reconstructor = RecordingExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
    )
    persistence = InMemoryProblemPersistence()
    orchestrator = RecordingDiagnosticOrchestrator(
        execution_reconstructor=reconstructor,
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=_build_grouping_engine(),
        problem_lifecycle_engine=ProblemLifecycleEngine(persistence),
    )
    trigger = TerminalExecutionDiagnosticTrigger(orchestrator)

    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(
        registry,
        trace_store=stores.trace_store,
        runtime_event_store=runtime_store,
    )
    loop.attach_terminal_diagnostic_trigger(trigger)
    loop.event_bus.subscribe(
        _inject_identity_preserving_violation(
            runtime_store,
            violating_event_type=RuntimeEventType.RETRY_SCHEDULED,
        ),
        event_types={RuntimeEventType.TASK_COMPLETED},
        priority=10,
    )
    return loop, runtime_store, orchestrator, reconstructor


def _event_index(events: tuple[RuntimeEvent, ...]) -> dict[EventId, RuntimeEvent]:
    return {event.event_id: event for event in events}


def _terminal_completed_event(events: tuple[RuntimeEvent, ...]) -> RuntimeEvent:
    completed = tuple(
        event for event in events if event.event_type is RuntimeEventType.TASK_COMPLETED
    )
    assert completed
    return completed[-1]


@pytest.mark.asyncio
async def test_ue_11f_terminal_failure_correlates_execution_identity_through_obs_evidence() -> None:
    loop, runtime_store, orchestrator, reconstructor = _build_ue_11f_nexus_stack()
    runner = UnifiedTaskRunner(loop)
    run_id = mint_run_id()

    result = await runner.run_task(
        Task(
            tenant_id=_TENANT,
            user_id="ue-11f-user",
            message="ue-11f terminal failure correlation",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=run_id,
    )

    assert result.state is TaskState.COMPLETED
    assert len(orchestrator.recorded_requests) == 1
    assert len(orchestrator.recorded_results) == 1

    persisted_events = tuple(runtime_store.list_for_task(result.task_id, tenant_id=_TENANT))
    terminal_event = _terminal_completed_event(persisted_events)
    violation_events = tuple(
        event
        for event in persisted_events
        if event.event_type is RuntimeEventType.RETRY_SCHEDULED
    )
    assert violation_events

    source_execution_id = violation_events[0].execution_id
    source_attempt_id = violation_events[0].attempt_id
    source_run_id = violation_events[0].run_id
    source_tenant_id = violation_events[0].tenant_id
    assert source_run_id == run_id
    for violation_event in violation_events:
        assert violation_event.execution_id == source_execution_id
        assert violation_event.attempt_id == source_attempt_id
        assert violation_event.run_id == source_run_id
        assert violation_event.tenant_id == source_tenant_id

    assert terminal_event.run_id == source_run_id
    assert terminal_event.attempt_id == source_attempt_id
    assert terminal_event.tenant_id == source_tenant_id

    request = orchestrator.recorded_requests[0]
    assert request.tenant_id == source_tenant_id
    assert len(request.executions) == 1
    scope = request.executions[0]
    assert scope.tenant_id == source_tenant_id
    assert scope.task_id == result.task_id
    assert scope.run_id == source_run_id

    assert len(reconstructor.invocations) == 1
    reconstructed_tenant, reconstructed_task_id, reconstructed_run_id = reconstructor.invocations[0]
    assert reconstructed_tenant == source_tenant_id
    assert reconstructed_task_id == result.task_id
    assert reconstructed_run_id == source_run_id

    reconstruction = reconstructor.reconstruct_execution(
        source_tenant_id,
        result.task_id,
        source_run_id,
    )
    assert reconstruction.positioned_events

    orchestration_result = orchestrator.recorded_results[0]
    execution_analysis = orchestration_result.execution_results[0]
    assert execution_analysis.tenant_id == source_tenant_id
    assert execution_analysis.task_id == result.task_id
    assert execution_analysis.run_id == source_run_id
    assert execution_analysis.assessment.has_findings

    events_by_id = _event_index(persisted_events)
    referenced_event_ids: set[EventId] = set()
    for finding in execution_analysis.assessment.findings:
        assert finding.attempt_id == source_attempt_id
        referenced_event_ids.update(finding.supporting_event_ids)
        for event_id in finding.supporting_event_ids:
            supporting_event = events_by_id[event_id]
            assert supporting_event.execution_id == source_execution_id
            assert supporting_event.attempt_id == source_attempt_id
            assert supporting_event.run_id == source_run_id
            assert supporting_event.tenant_id == source_tenant_id

    assert referenced_event_ids
    reconstruction_by_id = {
        positioned.event.event_id: positioned.event
        for positioned in reconstruction.positioned_events
    }
    for event_id in referenced_event_ids:
        reconstructed_event = reconstruction_by_id[event_id]
        assert reconstructed_event.execution_id == source_execution_id


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_forbidden_calls(
    path: Path,
    forbidden: frozenset[str],
) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in forbidden:
            violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def test_ue_11f_diagnostics_has_no_execution_lifecycle_ownership() -> None:
    violations: list[str] = []
    for path in sorted(_DIAGNOSTICS_ROOT.rglob("*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        violations.extend(_collect_forbidden_calls(path, _FORBIDDEN_LIFECYCLE_OWNERSHIP_CALLS))
        bind_violations = _collect_forbidden_calls(path, frozenset({"bind_active_execution_identity"}))
        if rel not in _BIND_ACTIVE_EXECUTION_IDENTITY_ALLOWLIST:
            violations.extend(bind_violations)
    assert violations == [], (
        "Diagnostics must not own execution lifecycle: " + ", ".join(violations)
    )
