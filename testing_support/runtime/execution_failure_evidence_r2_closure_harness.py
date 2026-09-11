# © Artur Czarnecki. All rights reserved.

"""Canonical ExecutionRuntime + ChildExecutionRunner harness for DIAG R2 closure (A12–A25)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import TaskId, mint_task_id
from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingEngine,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTrigger,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.contracts.execution_identity import RunId
from intergrax.runtime.execution.failure_evidence.runtime_event_recorder import (
    RuntimeEventExecutionFailureEvidenceRecorder,
)
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
    RootExecutionOptions,
    resolve_root_execution_context,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
    lifecycle_engine_for_tests,
    read_service_for_tests,
)

_OBSERVED_AT = datetime(2026, 9, 11, 10, 0, tzinfo=UTC)


@dataclass(slots=True)
class ExecutionFailureEvidenceClosureHarness:
    tenant_id: str
    task_id: TaskId
    runtime_store: InMemoryRuntimeEventStore
    event_bus: RuntimeEventBus
    failure_recorder: RuntimeEventExecutionFailureEvidenceRecorder
    lineage: InMemoryExecutionLineagePersistence
    runtime: ExecutionRuntime[object, object]
    child_runner: ChildExecutionRunner[object, object]
    execution_reconstructor: ExecutionReconstructor
    orchestrator: DiagnosticOrchestrator
    trigger: TerminalExecutionDiagnosticTrigger
    read_service: DiagnosticReadService
    problem_persistence: ProblemPersistence
    occurrence_persistence: ProblemOccurrencePersistence

    def bind_root_delegate(self, delegate: object) -> None:
        self.runtime = ExecutionRuntime[object, object](
            delegate,
            execution_lineage_persistence=self.lineage,
            failure_evidence_recorder=self.failure_recorder,
            run_budget=RunBudget(),
        )

    def reconstruct_for_run(self, run_id: RunId) -> object:
        return self.execution_reconstructor.reconstruct_execution(
            self.tenant_id,
            self.task_id,
            run_id,
        )

    def resolve_root_context(
        self,
        *,
        run_id: object | None = None,
        attempt_id: object | None = None,
        execution_id: object | None = None,
    ) -> RootExecutionContext:
        return resolve_root_execution_context(
            RootExecutionOptions(
                authority=ParentExecutionAuthority.unrestricted_root(),
                tenant_id=self.tenant_id,
                task_id=self.task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            ),
        )

    def list_execution_failed_events(
        self,
        *,
        attempt_id: object | None = None,
        execution_id: object | None = None,
    ) -> tuple[object, ...]:
        events = self.runtime_store.list_for_task(
            str(self.task_id),
            tenant_id=self.tenant_id,
            limit=500,
        )
        filtered = [
            event
            for event in events
            if event.event_type is RuntimeEventType.EXECUTION_FAILED
        ]
        if attempt_id is not None:
            filtered = [event for event in filtered if event.attempt_id == attempt_id]
        if execution_id is not None:
            filtered = [
                event for event in filtered if event.execution_id == execution_id
            ]
        return tuple(filtered)

    def seed_terminal_lifecycle_events(
        self,
        context: RootExecutionContext,
        *,
        failed: bool,
        cancelled: bool = False,
    ) -> None:
        if cancelled:
            terminal = RuntimeEventType.CANCELLED
        elif failed:
            terminal = RuntimeEventType.TASK_FAILED
        else:
            terminal = RuntimeEventType.TASK_COMPLETED
        for event_type in (
            RuntimeEventType.TASK_CREATED,
            terminal,
        ):
            event = sample_runtime_event(
                tenant_id=self.tenant_id,
                task_id=self.task_id,
                run_id=context.run_id,
                attempt_id=context.attempt_id,
                execution_id=context.execution_id,
            ).model_copy(update={"event_type": event_type})
            self.runtime_store.append(event, tenant_id=self.tenant_id)

    def run_terminal_diagnostics(self, context: RootExecutionContext) -> None:
        self.trigger.trigger_for_terminal_execution(
            tenant_id=self.tenant_id,
            task_id=self.task_id,
            run_id=context.run_id,
            observed_at=_OBSERVED_AT,
        )


def build_execution_failure_evidence_r2_closure_harness(
    *,
    tenant_id: str | None = None,
    lineage: ExecutionLineagePersistence | None = None,
    attach_lineage_to_reconstructor: bool = True,
    root_delegate: object | None = None,
) -> ExecutionFailureEvidenceClosureHarness:
    resolved_tenant = tenant_id or "tenant-efe-r2-closure"
    task_id = mint_task_id()
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    recorder = RuntimeEventExecutionFailureEvidenceRecorder(event_bus)
    resolved_lineage = (
        lineage
        if isinstance(lineage, InMemoryExecutionLineagePersistence)
        else InMemoryExecutionLineagePersistence()
    )
    runtime = ExecutionRuntime[object, object](
        delegate=root_delegate or _NoOpRootDelegate(),
        execution_lineage_persistence=resolved_lineage,
        failure_evidence_recorder=recorder,
        run_budget=RunBudget(),
    )
    child_runner = ChildExecutionRunner[object, object](
        ledger=create_execution_budget_ledger(RunBudget()),
    )
    lineage_reader = (
        resolved_lineage if attach_lineage_to_reconstructor else None
    )
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
        execution_lineage=lineage_reader,
    )
    from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
        InMemoryProblemPersistence,
    )

    persistence = InMemoryProblemPersistence()
    occurrence_store = in_memory_document_store_for_problem_tests()
    occurrence_persistence = document_store_occurrence_persistence_for_tests(
        occurrence_store,
    )
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=reconstructor,
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=ProblemGroupingEngine(registry),
        problem_lifecycle_engine=lifecycle_engine_for_tests(
            persistence,
            occurrence_persistence,
            document_store=occurrence_store,
        ),
    )
    read_service = read_service_for_tests(
        persistence,
        reconstructor,
        occurrence_persistence=occurrence_persistence,
        document_store=occurrence_store,
    )
    trigger = TerminalExecutionDiagnosticTrigger(orchestrator)
    return ExecutionFailureEvidenceClosureHarness(
        tenant_id=resolved_tenant,
        task_id=task_id,
        runtime_store=runtime_store,
        event_bus=event_bus,
        failure_recorder=recorder,
        lineage=resolved_lineage,
        runtime=runtime,
        child_runner=child_runner,
        execution_reconstructor=reconstructor,
        orchestrator=orchestrator,
        trigger=trigger,
        read_service=read_service,
        problem_persistence=persistence,
        occurrence_persistence=occurrence_persistence,
    )


class _NoOpRootDelegate:
    async def execute(self, request: object) -> object:
        return request


__all__ = [
    "ExecutionFailureEvidenceClosureHarness",
    "build_execution_failure_evidence_r2_closure_harness",
]
