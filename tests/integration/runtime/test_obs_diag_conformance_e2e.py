# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-CONFORMANCE — canonical evidence → reconstruction → diagnostics E2E."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticExecutionScope,
    DiagnosticOrchestrationIntegrityError,
    DiagnosticOrchestrationRequest,
    DiagnosticSignalSubjectScope,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingCandidate,
    ProblemGroupingEngine,
    ProblemGroupingMethod,
    ProblemGroupingProvenance,
    ProblemGroupingStrategy,
    ProblemGroupingStrategyCharacteristics,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyRegistry,
    ProblemGroupingStrategyResult,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_SOURCE_LAYER_APPLICATION,
    PlatformProblemSignal,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from testing_support.runtime.diagnostics.problem_persistence_test_support import read_service_for_tests
from tests.unit.runtime.diagnostics.test_diagnostic_orchestrator import (
    _build_orchestrator,
    _request,
    _scope,
    _seed_retry_violation_sequence,
)
from testing_support.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt
from intergrax.contracts.execution_lineage import build_execution_lineage_attempt_scope

pytestmark = [pytest.mark.integration, pytest.mark.obs_diag_conformance]

_TENANT = "tenant-a"
_OBSERVED_AT = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def _append_sequence(
    store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str,
    task_id,
    run_id,
    attempt_id,
    event_types: tuple[RuntimeEventType, ...],
) -> None:
    for event_type in event_types:
        store.append(
            sample_runtime_event(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ).model_copy(update={"event_type": event_type}),
            tenant_id=tenant_id,
        )


def test_successful_execution_produces_no_problem() -> None:
    orchestrator, runtime_store, _, persistence = _build_orchestrator()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    _append_sequence(
        runtime_store,
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=(RuntimeEventType.TASK_CREATED, RuntimeEventType.TASK_COMPLETED),
    )

    result = orchestrator.run(_request(_scope(task_id, run_id)))

    assert not result.execution_results[0].assessment.has_findings
    assert result.lifecycle_result.created == ()
    assert persistence.query_problems(tenant_id=_TENANT, limit=10).problems == ()


def test_terminal_failure_creates_problem_and_read_model() -> None:
    orchestrator, runtime_store, _, persistence = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    read_service = read_service_for_tests(persistence, reconstructor)

    result = orchestrator.run(_request(_scope(task_id, run_id), observed_at=_OBSERVED_AT))

    assert result.execution_results[0].assessment.has_findings
    problem = result.lifecycle_result.created[0]
    detail = read_service.get_problem(tenant_id=_TENANT, problem_id=problem.problem_id)
    assert detail is not None
    assert detail.tenant_id == _TENANT
    assert detail.occurrence_count == 1


def test_retry_a1_fail_a2_success_visible_in_reconstruction() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    _append_sequence(
        runtime_store,
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        event_types=(
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ),
    )
    _append_sequence(
        runtime_store,
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a2,
        event_types=(RuntimeEventType.TASK_CREATED, RuntimeEventType.TASK_COMPLETED),
    )

    orchestrator.run(_request(_scope(task_id, run_id)))
    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT, task_id, run_id)

    assert len(reconstruction.attempts) == 2
    assert reconstruction.attempts[0].attempt_id == attempt_a1
    assert reconstruction.attempts[1].attempt_id == attempt_a2


def test_child_lineage_comes_from_shared_reconstruction() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    lineage = InMemoryExecutionLineagePersistence()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    scope = build_execution_lineage_attempt_scope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    root_exec = mint_execution_id()
    child_exec = mint_execution_id()
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, root_exec)
    lineage.admit_root(scope, root_exec, root_exec)
    lineage.admit_child(scope, root_exec, child_exec, root_exec)

    _append_sequence(
        runtime_store,
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=(RuntimeEventType.TASK_CREATED, RuntimeEventType.TASK_COMPLETED),
    )
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
        execution_lineage=lineage,
    )
    orchestrator, _, _, _ = _build_orchestrator(
        runtime_store=runtime_store,
        execution_reconstructor=reconstructor,
    )

    result = orchestrator.run(_request(_scope(task_id, run_id)))

    reconstruction = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert reconstruction.has_lineage_evidence
    assert result.execution_results[0].has_runtime_events


def test_multi_execution_grouping_after_per_execution_reconstruction() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_a, run_a = _seed_retry_violation_sequence(runtime_store)
    task_b, run_b = _seed_retry_violation_sequence(runtime_store)

    result = orchestrator.run(_request(_scope(task_a, run_a), _scope(task_b, run_b)))

    assert len(result.execution_results) == 2
    assert len(result.grouping_result.candidates) >= 1
    assert len(result.lifecycle_result.created) >= 1


def test_tenant_isolation_fails_closed() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)

    with pytest.raises(DiagnosticOrchestrationIntegrityError):
        orchestrator.run(
            DiagnosticOrchestrationRequest(
                tenant_id=_TENANT,
                executions=(
                    DiagnosticExecutionScope(
                        tenant_id="tenant-other",
                        task_id=task_id,
                        run_id=run_id,
                    ),
                ),
                grouping_strategy_id=STRATEGY_ID,
                observed_at=_OBSERVED_AT,
            ),
        )


def test_signal_subject_path_skips_execution_reconstruction() -> None:
    orchestrator, _, _, persistence = _build_orchestrator()
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE,
        severity=PROBLEM_SEVERITY_ERROR,
        source_layer=PROBLEM_SOURCE_LAYER_APPLICATION,
        source_component="startup",
        safe_message="startup failed",
    )
    request = DiagnosticOrchestrationRequest(
        tenant_id=_TENANT,
        signal_subjects=(
            DiagnosticSignalSubjectScope(
                tenant_id=_TENANT,
                application_id="app-1",
                instance_id="inst-1",
                problem_signals=(signal,),
            ),
        ),
        grouping_strategy_id=STRATEGY_ID,
        observed_at=_OBSERVED_AT,
    )

    result = orchestrator.run(request)

    assert result.execution_results == ()
    assert len(result.signal_subject_results) == 1
    assert result.signal_subject_results[0].assessment.has_findings
    assert len(result.lifecycle_result.created) == 1
    assert persistence.query_problems(tenant_id=_TENANT, limit=10).problems


def test_diagnostic_persistence_does_not_mutate_execution_reconstruction() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    before = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    orchestrator.run(_request(_scope(task_id, run_id)))
    after = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert after == before


def test_reconstruction_input_immutable_through_diagnostic_spine() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )
    reconstruction = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    snapshot = replace(reconstruction)

    orchestrator.run(_request(_scope(task_id, run_id)))

    again = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert again == snapshot


_CUSTOM_STRATEGY_ID = ProblemGroupingStrategyId("obs.diag.conformance.custom")


class _SingletonPerInputGroupingStrategy(ProblemGroupingStrategy):
    @property
    def strategy_id(self) -> ProblemGroupingStrategyId:
        return _CUSTOM_STRATEGY_ID

    @property
    def strategy_version(self) -> str:
        return "1"

    @property
    def characteristics(self) -> ProblemGroupingStrategyCharacteristics:
        return ProblemGroupingStrategyCharacteristics(
            method=ProblemGroupingMethod.DETERMINISTIC,
            deterministic=True,
        )

    def group(self, inputs):
        candidates = tuple(
            ProblemGroupingCandidate(
                members=(inp.subject.ref,),
                provenance=ProblemGroupingProvenance(
                    strategy_id=self.strategy_id,
                    strategy_version=self.strategy_version,
                    method=ProblemGroupingMethod.DETERMINISTIC,
                    supporting_subject_refs=(inp.subject.ref,),
                ),
            )
            for inp in inputs
        )
        return ProblemGroupingStrategyResult(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            candidates=candidates,
        )


def test_custom_grouping_strategy_delegates_through_contract() -> None:
    from tests.unit.runtime.diagnostics.test_problem_grouping import (
        _assessment,
        _assessment_input,
    )

    registry = ProblemGroupingStrategyRegistry()
    registry.register(_SingletonPerInputGroupingStrategy())
    engine = ProblemGroupingEngine(registry)

    grouped = engine.group(
        (_assessment_input(_assessment()),),
        strategy_id=_CUSTOM_STRATEGY_ID,
    )

    assert grouped.strategy_id == _CUSTOM_STRATEGY_ID
    assert len(grouped.candidates) == 1


def test_injected_custom_execution_reconstruction_reader() -> None:
    from intergrax.runtime.observability.reconstruction import ExecutionReconstruction

    runtime_store = InMemoryRuntimeEventStore()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    calls: list[tuple[str, object, object]] = []
    default = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    )

    class _CustomExecutionReconstructionReader:
        def reconstruct_execution(self, tenant_id, task_id, run_id, *, execution_as_of=None):
            calls.append((tenant_id, task_id, run_id))
            return default.reconstruct_execution(
                tenant_id,
                task_id,
                run_id,
                execution_as_of=execution_as_of,
            )

    reader = _CustomExecutionReconstructionReader()
    orchestrator, _, _, _ = _build_orchestrator(
        runtime_store=runtime_store,
        execution_reconstructor=reader,
    )
    result = orchestrator.run(_request(_scope(task_id, run_id)))
    assert calls == [(_TENANT, task_id, run_id)]
    assert result.execution_results[0].assessment.has_findings
    assert isinstance(
        reader.reconstruct_execution(_TENANT, task_id, run_id),
        ExecutionReconstruction,
    )
