# © Artur Czarnecki. All rights reserved.

"""Worker role implementations for OBS-DG005 (import after qualification bootstrap)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import intergrax
from intergrax.contracts.execution_event_position import AsOfBoundary
from intergrax.contracts.execution_identity import EventId, mint_event_id, mint_execution_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticExecutionScope,
    DiagnosticOrchestrationRequest,
)
from intergrax.runtime.diagnostics.diagnostic_orchestrator import DiagnosticOrchestrator
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_problem_grouping_feature_projector import (
    DiagnosticProblemGroupingFeatureProjector,
)
from intergrax.runtime.diagnostics.in_memory_problem_persistence import (
    InMemoryProblemPersistence,
)
from tests.unit.runtime.diagnostics.problem_persistence_test_support import (
    document_store_occurrence_persistence_for_tests,
    in_memory_document_store_for_problem_tests,
    lifecycle_engine_for_tests,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingEngine,
    ProblemGroupingStrategyRegistry,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.runtime_event_history import RuntimeEventHistoryPolicy
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.contracts.event_severity import EventSeverity

from testing_support.obs_distributed_topology.models import (
    DiagnosticsWorkerResult,
    Dg005Scenario,
    IdempotentRetryWorkerResult,
    PlannedRuntimeEvent,
    PositionedEventSummary,
    ReaderWorkerResult,
    WriterWorkerResult,
)
from testing_support.obs_distributed_topology.provider_factory import (
    DEFAULT_DG005_EVIDENCE_PROVIDER_FACTORY,
    EvidenceProviderFactory,
)


def _import_root() -> str:
    return str(Path(intergrax.__file__).resolve().parent.parent)


def _planned_to_runtime_event(planned: PlannedRuntimeEvent) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=EventId(planned.event_id),
        tenant_id=planned.tenant_id,
        task_id=planned.task_id,
        run_id=planned.run_id,
        attempt_id=planned.attempt_id,
        execution_id=planned.execution_id,
        event_type=RuntimeEventType(planned.event_type),
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=EventSeverity.INFO,
        timestamp=datetime.fromisoformat(planned.timestamp_iso),
        correlation_id=str(planned.task_id),
    )


def run_writer_role(
    scenario: Dg005Scenario,
    *,
    provider_factory: EvidenceProviderFactory = DEFAULT_DG005_EVIDENCE_PROVIDER_FACTORY,
) -> WriterWorkerResult:
    port = provider_factory(scenario.provider)
    bus = RuntimeEventBus(
        persistence=port,
        record_history=False,
        history_policy=RuntimeEventHistoryPolicy.disabled(),
    )
    for planned in (
        *scenario.primary_events,
        *scenario.isolated_run_events,
        *scenario.foreign_tenant_events,
        scenario.idempotent_event,
    ):
        bus.record(_planned_to_runtime_event(planned), tenant_id=planned.tenant_id)

    summaries: list[PositionedEventSummary] = []
    for positioned in port.list_positioned_for_run(
        str(scenario.primary_run_id),
        tenant_id=scenario.primary_tenant,
    ):
        event = positioned.event
        summaries.append(
            PositionedEventSummary(
                event_id=str(event.event_id),
                position=positioned.position.value,
                tenant_id=event.tenant_id,
                task_id=str(event.task_id),
                run_id=str(event.run_id),
                attempt_id=str(event.attempt_id),
                execution_id=str(event.execution_id),
            )
        )

    return WriterWorkerResult(
        role="writer",
        qualification_sha=scenario.qualification_sha,
        import_root=_import_root(),
        intergrax_file=str(Path(intergrax.__file__).resolve()),
        history_len_after_writes=len(bus.history),
        primary_summaries=tuple(summaries),
        writer_provider_object_id=id(port),
    )


def run_reader_role(
    scenario: Dg005Scenario,
    *,
    provider_factory: EvidenceProviderFactory = DEFAULT_DG005_EVIDENCE_PROVIDER_FACTORY,
) -> ReaderWorkerResult:
    port = provider_factory(scenario.provider)
    reconstructor = ExecutionReconstructor(
        runtime_events=port,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    reconstruction = reconstructor.reconstruct_execution(
        scenario.primary_tenant,
        scenario.primary_task_id,
        scenario.primary_run_id,
        initial_limit=scenario.reconstruction_initial_limit,
    )
    actual_ids = [str(row.event.event_id) for row in reconstruction.positioned_events]
    positions = [row.position.value for row in reconstruction.positioned_events]

    as_of_position = scenario.primary_events[scenario.as_of_position_index - 1]
    as_of_boundary = AsOfBoundary(
        run_id=scenario.primary_run_id,
        position=next(
            positioned.position
            for positioned in port.list_positioned_for_run(
                str(scenario.primary_run_id),
                tenant_id=scenario.primary_tenant,
            )
            if str(positioned.event.event_id) == str(as_of_position.event_id)
        ),
    )
    as_of_reconstruction = reconstructor.reconstruct_execution(
        scenario.primary_tenant,
        scenario.primary_task_id,
        scenario.primary_run_id,
        execution_as_of=as_of_boundary,
        initial_limit=scenario.reconstruction_initial_limit,
    )
    as_of_ids = [
        str(row.event.event_id) for row in as_of_reconstruction.positioned_events
    ]

    isolated_ids = [
        str(row.event.event_id)
        for row in port.list_positioned_for_run(
            str(scenario.isolated_run_id),
            tenant_id=scenario.primary_tenant,
        )
    ]
    foreign_visible = len(
        port.list_positioned_for_run(
            str(scenario.foreign_run_id),
            tenant_id=scenario.primary_tenant,
        )
    )
    grouped = port.list_positioned_for_task_grouped_by_run(
        str(scenario.primary_task_id),
        tenant_id=scenario.primary_tenant,
    )
    grouped_run_ids = tuple(sorted(str(run_id) for run_id, _ in grouped.runs))

    idempotent_count = len(
        port.list_positioned_for_run(
            str(scenario.primary_run_id),
            tenant_id=scenario.primary_tenant,
        )
    )

    return ReaderWorkerResult(
        role="reader",
        qualification_sha=scenario.qualification_sha,
        import_root=_import_root(),
        intergrax_file=str(Path(intergrax.__file__).resolve()),
        reader_provider_object_id=id(port),
        runtime_history_completeness=reconstruction.runtime_history_completeness.value,
        event_ids_in_order=tuple(actual_ids),
        positions_in_order=tuple(positions),
        as_of_event_ids=tuple(as_of_ids),
        isolated_run_event_ids=tuple(isolated_ids),
        foreign_tenant_visible_count=foreign_visible,
        task_grouped_run_ids=grouped_run_ids,
        idempotent_run_count=idempotent_count,
    )


def run_idempotent_retry_role(
    scenario: Dg005Scenario,
    *,
    provider_factory: EvidenceProviderFactory = DEFAULT_DG005_EVIDENCE_PROVIDER_FACTORY,
) -> IdempotentRetryWorkerResult:
    port = provider_factory(scenario.provider)
    port.append(
        _planned_to_runtime_event(scenario.idempotent_event),
        tenant_id=scenario.idempotent_event.tenant_id,
    )
    listed = port.list_positioned_for_run(
        str(scenario.primary_run_id),
        tenant_id=scenario.primary_tenant,
    )
    return IdempotentRetryWorkerResult(
        role="idempotent_retry",
        qualification_sha=scenario.qualification_sha,
        import_root=_import_root(),
        intergrax_file=str(Path(intergrax.__file__).resolve()),
        listed_count=len(listed),
    )


def run_diagnostics_role(
    scenario: Dg005Scenario,
    *,
    provider_factory: EvidenceProviderFactory = DEFAULT_DG005_EVIDENCE_PROVIDER_FACTORY,
) -> DiagnosticsWorkerResult:
    port = provider_factory(scenario.provider)
    reconstructor = ExecutionReconstructor(
        runtime_events=port,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    grouping_engine = ProblemGroupingEngine(
        registry,
        feature_projector=DiagnosticProblemGroupingFeatureProjector(),
    )
    persistence = InMemoryProblemPersistence()
    orchestrator = DiagnosticOrchestrator(
        execution_reconstructor=reconstructor,
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
        assessment_builder=DiagnosticAssessmentBuilder(),
        grouping_engine=grouping_engine,
        problem_lifecycle_engine=lifecycle_engine_for_tests(
            persistence,
            document_store_occurrence_persistence_for_tests(
                in_memory_document_store_for_problem_tests(),
            ),
        ),
    )

    diag_time = datetime(2026, 6, 9, 8, 0, 0, tzinfo=UTC)
    for index, event_type in enumerate(
        (
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        )
    ):
        planned = PlannedRuntimeEvent(
            event_id=mint_event_id(),
            tenant_id=scenario.primary_tenant,
            task_id=scenario.diagnostics_task_id,
            run_id=scenario.diagnostics_run_id,
            attempt_id=scenario.diagnostics_attempt_id,
            execution_id=mint_execution_id(),
            event_type=event_type.value,
            timestamp_iso=(diag_time + timedelta(minutes=index)).isoformat(),
        )
        port.append(
            _planned_to_runtime_event(planned),
            tenant_id=planned.tenant_id,
        )

    request = DiagnosticOrchestrationRequest(
        tenant_id=scenario.primary_tenant,
        executions=(
            DiagnosticExecutionScope(
                tenant_id=scenario.primary_tenant,
                task_id=scenario.diagnostics_task_id,
                run_id=scenario.diagnostics_run_id,
            ),
        ),
    )
    result = orchestrator.run(request)
    return DiagnosticsWorkerResult(
        role="diagnostics",
        qualification_sha=scenario.qualification_sha,
        import_root=_import_root(),
        intergrax_file=str(Path(intergrax.__file__).resolve()),
        execution_analyses=len(result.execution_results),
        grouping_candidates=len(result.grouping_result.candidates),
    )


def worker_result_to_json_dict(result: object) -> dict[str, object]:
    return asdict(result)
