# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import (
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticCertainty
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceConflictError,
    FunctionalEvidencePersistenceIntegrityError,
    FunctionalEvidenceQueryRequest,
)
from intergrax.runtime.diagnostics.functional_evidence_reconstruction import (
    FunctionalEvidenceReconstructor,
)
from intergrax.runtime.diagnostics.functional_validation import (
    DiagnosticExecutionCorrelation,
    ExpectedActualRelation,
    FunctionalValidationEvidence,
    FunctionalValidationIntegrityError,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
    build_functional_outcome_invalid_signal,
    functional_validation_evidence_id,
    validate_functional_validation_correlation,
    validate_problem_signal_correlation_alignment,
)
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID,
    PlatformProblemSignal,
)

pytestmark = pytest.mark.unit


def _correlation(*, tenant_id: str = "tenant-a") -> DiagnosticExecutionCorrelation:
    return DiagnosticExecutionCorrelation(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )


def _validation(
  correlation: DiagnosticExecutionCorrelation,
  *,
  outcome: FunctionalValidationOutcome = FunctionalValidationOutcome.FAILED,
) -> FunctionalValidationEvidence:
    return FunctionalValidationEvidence(
        validation_id=functional_validation_evidence_id(
            validator_id="oracle.v1",
            validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
            correlation=correlation,
            idempotency_key="attempt-1",
        ),
        validator=FunctionalValidatorRef(validator_id="oracle.v1", validator_version="1"),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=outcome,
        correlation=correlation,
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
        relation_summary="expected fact missing from answer",
    )


def _artifact_ref(ref: str) -> ObservabilityArtifactReference:
    return ObservabilityArtifactReference(artifact_ref=ref)


def _terminal_completed_event(correlation: DiagnosticExecutionCorrelation) -> RuntimeEvent:
    return sample_runtime_event(
        tenant_id=correlation.tenant_id,
        task_id=correlation.task_id,
        run_id=correlation.run_id,
    ).model_copy(update={"event_type": RuntimeEventType.TASK_COMPLETED})


def test_t1_completed_execution_and_functional_fail_are_independent_facts() -> None:
    correlation = _correlation()
    validation = _validation(correlation)
    _, signal = build_functional_outcome_invalid_signal(validation=validation)

    store = InMemoryRuntimeEventStore()
    terminal_event = _terminal_completed_event(correlation)
    store.append(terminal_event, tenant_id=correlation.tenant_id)

    assert terminal_event.event_type is RuntimeEventType.TASK_COMPLETED
    assert signal.problem_kind == PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID
    assert signal.functional_validation is validation
    assert validation.outcome is FunctionalValidationOutcome.FAILED


def test_t2_functional_fail_does_not_change_execution_terminal_state() -> None:
    correlation = _correlation()
    validation = _validation(correlation)
    build_functional_outcome_invalid_signal(validation=validation)

    store = InMemoryRuntimeEventStore()
    before = store.list_for_task(
        str(correlation.task_id),
        tenant_id=correlation.tenant_id,
    )
    terminal_event = _terminal_completed_event(correlation)
    store.append(terminal_event, tenant_id=correlation.tenant_id)
    after = store.list_for_task(
        str(correlation.task_id),
        tenant_id=correlation.tenant_id,
    )

    assert len(after) == len(before) + 1
    assert terminal_event.event_type is RuntimeEventType.TASK_COMPLETED
    assert after[-1].event_type is RuntimeEventType.TASK_COMPLETED


def test_t3_missing_execution_correlation_fails_closed() -> None:
    correlation = _correlation()

    with pytest.raises(FunctionalValidationIntegrityError):
        validate_functional_validation_correlation(tenant_id="", correlation=correlation)

    with pytest.raises(FunctionalValidationIntegrityError):
        validate_problem_signal_correlation_alignment(
            signal_task_id=str(correlation.task_id),
            signal_run_id=str(mint_run_id()),
            correlation=correlation,
        )


def test_t4_tenant_mismatch_fails_closed() -> None:
    correlation = _correlation(tenant_id="tenant-a")
    validation = _validation(correlation)

    with pytest.raises(FunctionalValidationIntegrityError):
        validate_functional_validation_correlation(tenant_id="tenant-b", correlation=correlation)

    persistence = InMemoryFunctionalEvidencePersistence()
    evidence = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        scope=PipelineEvidenceScope(
            tenant_id=correlation.tenant_id,
            task_id=correlation.task_id,
            run_id=correlation.run_id,
        ),
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.test",
            operation_id="op-1",
        ),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name="embed",
            status=PipelineOperationStatus.SUCCEEDED,
        ),
    )
    persistence.append(evidence)

    with pytest.raises(FunctionalEvidencePersistenceIntegrityError):
        persistence.query_evidence(
            FunctionalEvidenceQueryRequest(
                tenant_id="tenant-b",
                task_id=correlation.task_id,
                run_id=correlation.run_id,
                cursor=f"tenant-a|{correlation.task_id}|{correlation.run_id}|0",
            )
        )


def test_t5_duplicate_functional_evidence_is_idempotent() -> None:
    persistence = InMemoryFunctionalEvidencePersistence()
    correlation = _correlation()
    evidence = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        scope=PipelineEvidenceScope(
            tenant_id=correlation.tenant_id,
            task_id=correlation.task_id,
            run_id=correlation.run_id,
        ),
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.test",
            operation_id="op-1",
        ),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name="embed",
            status=PipelineOperationStatus.SUCCEEDED,
        ),
    )

    first = persistence.append(evidence)
    second = persistence.append(evidence)
    assert first == second

    conflicting = evidence.model_copy(
        update={
            "operation_outcome": PipelineOperationOutcomeFact(
                operation_name="index",
                status=PipelineOperationStatus.FAILED,
            ),
        }
    )
    with pytest.raises(FunctionalEvidencePersistenceConflictError):
        persistence.append(conflicting)


def test_t6_out_of_order_evidence_has_deterministic_reconstruction_order() -> None:
    persistence = InMemoryFunctionalEvidencePersistence()
    correlation = _correlation()
    later = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    earlier = datetime(2026, 8, 31, 11, 0, tzinfo=timezone.utc)

    def _operation(operation_name: str, recorded_at: datetime) -> PlatformFunctionalEvidence:
        return PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
            scope=PipelineEvidenceScope(
                tenant_id=correlation.tenant_id,
                task_id=correlation.task_id,
                run_id=correlation.run_id,
            ),
            provenance=PipelineEvidenceProvenance(
                producer_component="diag.test",
                operation_id=operation_name,
                recorded_at=recorded_at,
            ),
            operation_outcome=PipelineOperationOutcomeFact(
                operation_name=operation_name,
                status=PipelineOperationStatus.SUCCEEDED,
            ),
        )

    persistence.append(_operation("index", later))
    persistence.append(_operation("embed", earlier))

    reconstruction = FunctionalEvidenceReconstructor(persistence).reconstruct(
        tenant_id=correlation.tenant_id,
        task_id=correlation.task_id,
        run_id=correlation.run_id,
    )
    operation_names = tuple(
        item.operation_outcome.operation_name
        for item in reconstruction.evidence
        if item.operation_outcome is not None
    )
    assert operation_names == ("embed", "index")


def test_t7_missing_evidence_yields_insufficient_evidence() -> None:
    persistence = InMemoryFunctionalEvidencePersistence()
    correlation = _correlation()
    reconstruction = FunctionalEvidenceReconstructor(persistence).reconstruct(
        tenant_id=correlation.tenant_id,
        task_id=correlation.task_id,
        run_id=correlation.run_id,
        required_kinds=frozenset({PipelineEvidenceKind.CANDIDATE_RANK}),
    )

    assert reconstruction.certainty is DiagnosticCertainty.INSUFFICIENT_EVIDENCE
    assert PipelineEvidenceKind.CANDIDATE_RANK in reconstruction.completeness.missing_kinds


def test_functional_signal_requires_failed_validation_outcome() -> None:
    correlation = _correlation()
    validation = _validation(correlation, outcome=FunctionalValidationOutcome.PASSED)

    with pytest.raises(FunctionalValidationIntegrityError):
        build_functional_outcome_invalid_signal(validation=validation)


def test_problem_signal_accepts_functional_validation_payload() -> None:
    correlation = _correlation()
    validation = _validation(correlation)
    signal = PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_FUNCTIONAL_OUTCOME_INVALID,
        functional_validation=validation,
        task_id=str(correlation.task_id),
        run_id=str(correlation.run_id),
    )

    assert signal.functional_validation == validation
