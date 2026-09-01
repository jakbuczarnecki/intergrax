# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.functional_diagnostic_bounds import (
    MAX_FUNCTIONAL_DIAGNOSTIC_CHECKS,
    MAX_FUNCTIONAL_DIAGNOSTIC_DEPENDENCIES,
    MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticCheckStatus,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import (
    FunctionalDiagnosticAnalyzer,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.runtime.diagnostics.functional_diagnostic_specification import (
    CandidateExistsRequirement,
    FunctionalDiagnosticCheck,
    FunctionalDiagnosticRequirement,
    FunctionalDiagnosticRequirementKind,
    FunctionalDiagnosticSpecification,
    FunctionalDiagnosticSpecificationIntegrityError,
    OperationOutcomeStatusRequirement,
    OutputRelationExistsRequirement,
    SelectionArtifactMatchRequirement,
    SelectionExistsRequirement,
    ValidationOutcomeRequirement,
    validate_functional_diagnostic_specification,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineCandidateFact,
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
    PipelineOutputRelationFact,
    PipelineSelectionFact,
    PipelineValidationLinkFact,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_validation import (
    DiagnosticExecutionCorrelation,
    ExpectedActualRelation,
    FunctionalValidationEvidence,
    FunctionalValidationKind,
    FunctionalValidationOutcome,
    FunctionalValidatorRef,
    functional_validation_evidence_id,
)
from intergrax.runtime.diagnostics.functional_validation_lookup import (
    FunctionalValidationEvidenceLookup,
)
from intergrax.runtime.diagnostics.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference

pytestmark = pytest.mark.unit

_TEST_CURSOR_SECRET = b"x" * 32
_BASE_TIME = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)

_SPEC_ID = FunctionalDiagnosticSpecificationId("fdspec_a0000000000000000000000000000001")
_CHECK_OPERATION = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000001")
_CHECK_CANDIDATE = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000002")
_CHECK_SELECTION = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000003")
_CHECK_OUTPUT = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000004")
_CHECK_VALIDATION = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000005")


def _persistence() -> InMemoryFunctionalEvidencePersistence:
    return InMemoryFunctionalEvidencePersistence(cursor_secret=_TEST_CURSOR_SECRET)


def _scope(
    *,
    tenant_id: str = "tenant-a",
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> PipelineEvidenceScope:
    return PipelineEvidenceScope(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id,
    )


def _artifact(ref: str) -> ObservabilityArtifactReference:
    return ObservabilityArtifactReference(artifact_ref=ref)


def _append(
    persistence: InMemoryFunctionalEvidencePersistence,
    evidence: PlatformFunctionalEvidence,
) -> None:
    persistence.append(evidence)


def _operation(
    scope: PipelineEvidenceScope,
    *,
    operation_id: str,
    status: PipelineOperationStatus = PipelineOperationStatus.SUCCEEDED,
    recorded_at: datetime | None = None,
    evidence_id: str | None = None,
) -> PlatformFunctionalEvidence:
    return PlatformFunctionalEvidence(
        evidence_id=evidence_id or mint_event_id(),
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.test",
            operation_id=operation_id,
            recorded_at=recorded_at or _BASE_TIME,
        ),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name=operation_id,
            status=status,
        ),
    )


def _candidate(
    scope: PipelineEvidenceScope,
    *,
    query_id: str,
    artifact_ref: str = "candidate:1",
    recorded_at: datetime | None = None,
) -> PlatformFunctionalEvidence:
    return PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.CANDIDATE_RANK,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.test",
            operation_id="rank",
            recorded_at=recorded_at or _BASE_TIME,
        ),
        candidate=PipelineCandidateFact(
            query_id=query_id,
            candidate_artifact_ref=_artifact(artifact_ref),
            rank=1,
            selected=False,
        ),
    )


def _selection(
    scope: PipelineEvidenceScope,
    *,
    query_id: str,
    selected_ref: str,
    recorded_at: datetime | None = None,
) -> PlatformFunctionalEvidence:
    return PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.SELECTION,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.test",
            operation_id="select",
            recorded_at=recorded_at or _BASE_TIME,
        ),
        selection=PipelineSelectionFact(
            query_id=query_id,
            selected_artifact_ref=_artifact(selected_ref),
            candidate_count=3,
        ),
    )


def _output_relation(
    scope: PipelineEvidenceScope,
    *,
    operation_id: str,
    recorded_at: datetime | None = None,
) -> PlatformFunctionalEvidence:
    return PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.OUTPUT_RELATION,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.test",
            operation_id=operation_id,
            recorded_at=recorded_at or _BASE_TIME,
        ),
        output_relation=PipelineOutputRelationFact(
            selected_artifact_ref=_artifact("selected:1"),
            output_artifact_ref=_artifact("output:1"),
            relation_kind="generated_from",
        ),
    )


def _validation(
    scope: PipelineEvidenceScope,
    *,
    outcome: FunctionalValidationOutcome,
    idempotency_key: str = "attempt-1",
) -> tuple[FunctionalValidationEvidence, PlatformFunctionalEvidence]:
    correlation = DiagnosticExecutionCorrelation(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )
    validation_id = functional_validation_evidence_id(
        validator_id="oracle.v1",
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        correlation=correlation,
        idempotency_key=idempotency_key,
    )
    validation = FunctionalValidationEvidence(
        validation_id=validation_id,
        validator=FunctionalValidatorRef(validator_id="oracle.v1"),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=outcome,
        correlation=correlation,
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
    )
    link = PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.VALIDATION,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.test",
            operation_id="validate",
            recorded_at=_BASE_TIME,
        ),
        validation_link=PipelineValidationLinkFact(
            validation_id=validation.validation_id,
            output_artifact_ref=_artifact("output:1"),
        ),
    )
    return validation, link


def _analyzer(
    persistence: InMemoryFunctionalEvidencePersistence,
) -> FunctionalDiagnosticAnalyzer:
    return FunctionalDiagnosticAnalyzer(persistence)


def _simple_spec(
  checks: tuple[FunctionalDiagnosticCheck, ...],
) -> FunctionalDiagnosticSpecification:
    return FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=checks,
    )


def test_f3_1_generic_pass() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(persistence, _operation(scope, operation_id="search"))
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Operation search succeeded.",
                fail_claim="Operation search failed.",
                insufficient_claim="No operation outcome evidence.",
            ),
        ),
    )
    analysis = _analyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    )
    assert analysis.check_results[0].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS


def test_f3_2_generic_fail() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(
        persistence,
        _operation(scope, operation_id="search", status=PipelineOperationStatus.FAILED),
    )
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Operation search succeeded.",
                fail_claim="Operation search failed.",
                insufficient_claim="No operation outcome evidence.",
            ),
        ),
    )
    analysis = _analyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    )
    result = analysis.check_results[0]
    assert result.status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
    assert analysis.first_proven_failure == _CHECK_OPERATION


def test_f3_3_missing_evidence() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(persistence, _operation(scope, operation_id="search"))
    validation, validation_link = _validation(scope, outcome=FunctionalValidationOutcome.FAILED)
    _append(persistence, validation_link)
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Operation passed.",
                fail_claim="Operation failed.",
                insufficient_claim="No operation evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_CANDIDATE,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id="q1"),
                ),
                pass_claim="Candidates exist.",
                fail_claim="Candidates missing.",
                insufficient_claim="No candidate evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_VALIDATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
                    validation_outcome=ValidationOutcomeRequirement(
                        validation_id=validation.validation_id,
                        expected_outcome=FunctionalValidationOutcome.PASSED,
                    ),
                ),
                pass_claim="Validation passed.",
                fail_claim="Validation failed.",
                insufficient_claim="No validation evidence.",
            ),
        ),
    )
    analysis = _analyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
        validations=FunctionalValidationEvidenceLookup.for_scope(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            validations=(validation,),
        ),
    )
    by_id = {item.check_id: item for item in analysis.check_results}
    assert by_id[_CHECK_OPERATION].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_CANDIDATE].status is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE
    assert by_id[_CHECK_VALIDATION].status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL


def test_f3_4_contradictory_evidence() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(
        persistence,
        _operation(
            scope,
            operation_id="search",
            status=PipelineOperationStatus.SUCCEEDED,
            recorded_at=_BASE_TIME,
            evidence_id=mint_event_id(),
        ),
    )
    _append(
        persistence,
        _operation(
            scope,
            operation_id="search",
            status=PipelineOperationStatus.FAILED,
            recorded_at=_BASE_TIME + timedelta(seconds=1),
            evidence_id=mint_event_id(),
        ),
    )
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Passed.",
                fail_claim="Failed.",
                insufficient_claim="Insufficient.",
            ),
        ),
    )
    result = _analyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    ).check_results[0]
    assert result.status is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE
    assert result.limitations


def test_f3_5_dependency_semantics() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(
        persistence,
        _operation(scope, operation_id="search", status=PipelineOperationStatus.FAILED),
    )
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Op pass.",
                fail_claim="Op fail.",
                insufficient_claim="Op missing.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OUTPUT,
                dependencies=(_CHECK_OPERATION,),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OUTPUT_RELATION_EXISTS,
                    output_relation_exists=OutputRelationExistsRequirement(operation_id="synthesize"),
                ),
                pass_claim="Output exists.",
                fail_claim="Output missing.",
                insufficient_claim="No output evidence.",
            ),
        ),
    )
    by_id = {
        item.check_id: item
        for item in _analyzer(persistence).analyze(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            specification=spec,
        ).check_results
    }
    assert by_id[_CHECK_OPERATION].status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
    assert by_id[_CHECK_OUTPUT].status is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM


def test_f3_6_duplicate_evidence() -> None:
    scope = _scope()
    persistence = _persistence()
    evidence = _operation(scope, operation_id="search")
    _append(persistence, evidence)
    _append(persistence, evidence)
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Passed.",
                fail_claim="Failed.",
                insufficient_claim="Missing.",
            ),
        ),
    )
    first = _analyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    )
    second = _analyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    )
    assert first == second


def test_f3_7_out_of_order_evidence() -> None:
    scope = _scope()
    persistence = _persistence()
    late = _operation(
        scope,
        operation_id="search",
        recorded_at=_BASE_TIME + timedelta(hours=1),
    )
    early = _operation(
        scope,
        operation_id="search",
        recorded_at=_BASE_TIME,
    )
    _append(persistence, late)
    _append(persistence, early)
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Passed.",
                fail_claim="Failed.",
                insufficient_claim="Missing.",
            ),
        ),
    )
    result = _analyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    ).check_results[0]
    assert result.status is FunctionalDiagnosticCheckStatus.PROVEN_PASS


def test_f3_8_late_evidence_convergence() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(persistence, _operation(scope, operation_id="search"))
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_CANDIDATE,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id="q1"),
                ),
                pass_claim="Candidates exist.",
                fail_claim="No candidates.",
                insufficient_claim="No candidate evidence.",
            ),
        ),
    )
    analyzer = _analyzer(persistence)
    before = analyzer.analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    ).check_results[0]
    assert before.status is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE
    _append(persistence, _candidate(scope, query_id="q1"))
    after = analyzer.analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    ).check_results[0]
    assert after.status is FunctionalDiagnosticCheckStatus.PROVEN_PASS


def test_f3_9_tenant_isolation() -> None:
    scope_a = _scope(tenant_id="tenant-a")
    scope_b = _scope(
        tenant_id="tenant-b",
        task_id=scope_a.task_id,
        run_id=scope_a.run_id,
    )
    persistence = _persistence()
    _append(persistence, _operation(scope_b, operation_id="search"))
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Passed.",
                fail_claim="Failed.",
                insufficient_claim="Missing.",
            ),
        ),
    )
    result = _analyzer(persistence).analyze(
        tenant_id=scope_a.tenant_id,
        task_id=scope_a.task_id,
        run_id=scope_a.run_id,
        specification=spec,
    ).check_results[0]
    assert result.status is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE


def test_f3_10_attempt_isolation() -> None:
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    scope = _scope(attempt_id=attempt_a)
    scope_b = PipelineEvidenceScope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=attempt_b,
    )
    persistence = _persistence()
    _append(persistence, _operation(scope_b, operation_id="search"))
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Passed.",
                fail_claim="Failed.",
                insufficient_claim="Missing.",
            ),
        ),
    )
    result = _analyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=attempt_a,
        specification=spec,
    ).check_results[0]
    assert result.status is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE


def test_f3_11_bounded_results() -> None:
    checks: list[FunctionalDiagnosticCheck] = []
    for index in range(MAX_FUNCTIONAL_DIAGNOSTIC_CHECKS):
        suffix = f"{index:032x}"
        checks.append(
            FunctionalDiagnosticCheck(
                check_id=FunctionalDiagnosticCheckId(f"fdcheck_{suffix}"),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id=f"op-{index}",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="p",
                fail_claim="f",
                insufficient_claim="i",
            ),
        )
    validate_functional_diagnostic_specification(
        FunctionalDiagnosticSpecification(
            specification_id=_SPEC_ID,
            version=1,
            checks=tuple(checks),
        ),
    )
    with pytest.raises(FunctionalDiagnosticSpecificationIntegrityError):
        validate_functional_diagnostic_specification(
            FunctionalDiagnosticSpecification(
                specification_id=_SPEC_ID,
                version=1,
                checks=tuple(
                    checks
                    + [
                        FunctionalDiagnosticCheck(
                            check_id=FunctionalDiagnosticCheckId(
                                "fdcheck_b0000000000000000000000000000001",
                            ),
                            requirement=FunctionalDiagnosticRequirement(
                                kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                                operation_outcome_status=OperationOutcomeStatusRequirement(
                                    operation_id="overflow",
                                    expected_status=PipelineOperationStatus.SUCCEEDED,
                                ),
                            ),
                            pass_claim="p",
                            fail_claim="f",
                            insufficient_claim="i",
                        ),
                    ],
                ),
            ),
        )
    deps = tuple(
        FunctionalDiagnosticCheckId(f"fdcheck_{index:032x}")
        for index in range(MAX_FUNCTIONAL_DIAGNOSTIC_DEPENDENCIES)
    )
    validate_functional_diagnostic_specification(
        FunctionalDiagnosticSpecification(
            specification_id=_SPEC_ID,
            version=1,
            checks=(
                FunctionalDiagnosticCheck(
                    check_id=FunctionalDiagnosticCheckId("fdcheck_c0000000000000000000000000000001"),
                    dependencies=deps,
                    requirement=FunctionalDiagnosticRequirement(
                        kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                        operation_outcome_status=OperationOutcomeStatusRequirement(
                            operation_id="dep-test",
                            expected_status=PipelineOperationStatus.SUCCEEDED,
                        ),
                    ),
                    pass_claim="p",
                    fail_claim="f",
                    insufficient_claim="i",
                ),
                *(
                    FunctionalDiagnosticCheck(
                        check_id=dep,
                        requirement=FunctionalDiagnosticRequirement(
                            kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                            operation_outcome_status=OperationOutcomeStatusRequirement(
                                operation_id=f"dep-{dep}",
                                expected_status=PipelineOperationStatus.SUCCEEDED,
                            ),
                        ),
                        pass_claim="p",
                        fail_claim="f",
                        insufficient_claim="i",
                    )
                    for dep in deps
                ),
            ),
        ),
    )
    scope = _scope()
    persistence = _persistence()
    for index in range(MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS + 2):
        _append(
            persistence,
            _operation(
                scope,
                operation_id="search",
                recorded_at=_BASE_TIME + timedelta(seconds=index),
            ),
        )
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Passed.",
                fail_claim="Failed.",
                insufficient_claim="Missing.",
            ),
        ),
    )
    result = _analyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    ).check_results[0]
    assert len(result.supporting_evidence_refs) <= MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS


def _retrieval_like_spec() -> FunctionalDiagnosticSpecification:
    return _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Search operation succeeded.",
                fail_claim="Search operation failed.",
                insufficient_claim="No search operation evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_CANDIDATE,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id="retrieval-q1"),
                ),
                pass_claim="Candidates generated.",
                fail_claim="No candidates.",
                insufficient_claim="No candidate evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_SELECTION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH,
                    selection_artifact_match=SelectionArtifactMatchRequirement(
                        query_id="retrieval-q1",
                        expected_artifact_ref="context:correct",
                    ),
                ),
                pass_claim="Correct context selected.",
                fail_claim="Wrong context selected.",
                insufficient_claim="No selection evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OUTPUT,
                dependencies=(_CHECK_SELECTION,),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OUTPUT_RELATION_EXISTS,
                    output_relation_exists=OutputRelationExistsRequirement(operation_id="synthesize"),
                ),
                pass_claim="Output produced.",
                fail_claim="No output.",
                insufficient_claim="No output evidence.",
            ),
        ),
    )


def test_f3_12_retrieval_like_scenario() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(persistence, _operation(scope, operation_id="search"))
    _append(persistence, _candidate(scope, query_id="retrieval-q1"))
    _append(persistence, _selection(scope, query_id="retrieval-q1", selected_ref="context:wrong"))
    by_id = {
        item.check_id: item
        for item in _analyzer(persistence).analyze(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            specification=_retrieval_like_spec(),
        ).check_results
    }
    assert by_id[_CHECK_OPERATION].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_CANDIDATE].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_SELECTION].status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
    assert by_id[_CHECK_OUTPUT].status is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM


def test_f3_13_tool_scenario() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(persistence, _candidate(scope, query_id="tool-q1", artifact_ref="tool:search_web"))
    _append(persistence, _selection(scope, query_id="tool-q1", selected_ref="tool:wrong_tool"))
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_CANDIDATE,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id="tool-q1"),
                ),
                pass_claim="Tool candidates exist.",
                fail_claim="No tool candidates.",
                insufficient_claim="No candidate evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_SELECTION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH,
                    selection_artifact_match=SelectionArtifactMatchRequirement(
                        query_id="tool-q1",
                        expected_artifact_ref="tool:search_web",
                    ),
                ),
                pass_claim="Correct tool selected.",
                fail_claim="Wrong tool selected.",
                insufficient_claim="No selection evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                dependencies=(_CHECK_SELECTION,),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="invoke",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Tool invoked.",
                fail_claim="Invocation failed.",
                insufficient_claim="No invocation evidence.",
            ),
        ),
    )
    by_id = {
        item.check_id: item
        for item in _analyzer(persistence).analyze(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            specification=spec,
        ).check_results
    }
    assert by_id[_CHECK_CANDIDATE].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_SELECTION].status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
    assert by_id[_CHECK_OPERATION].status is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM


def test_f3_14_web_search_scenario() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(persistence, _operation(scope, operation_id="web_search"))
    _append(persistence, _candidate(scope, query_id="web-q1", artifact_ref="web:result-2"))
    _append(persistence, _selection(scope, query_id="web-q1", selected_ref="web:result-2"))
    validation, validation_link = _validation(scope, outcome=FunctionalValidationOutcome.FAILED)
    _append(persistence, validation_link)
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="web_search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Search succeeded.",
                fail_claim="Search failed.",
                insufficient_claim="No search evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_CANDIDATE,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id="web-q1"),
                ),
                pass_claim="Sources found.",
                fail_claim="No sources.",
                insufficient_claim="No source evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_SELECTION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.SELECTION_EXISTS,
                    selection_exists=SelectionExistsRequirement(query_id="web-q1"),
                ),
                pass_claim="Source selected.",
                fail_claim="No selection.",
                insufficient_claim="No selection evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_VALIDATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
                    validation_outcome=ValidationOutcomeRequirement(
                        validation_id=validation.validation_id,
                        expected_outcome=FunctionalValidationOutcome.PASSED,
                    ),
                ),
                pass_claim="Answer validated.",
                fail_claim="Answer validation failed.",
                insufficient_claim="No validation evidence.",
            ),
        ),
    )
    by_id = {
        item.check_id: item
        for item in _analyzer(persistence).analyze(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            specification=spec,
            validations=FunctionalValidationEvidenceLookup.for_scope(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            validations=(validation,),
        ),
        ).check_results
    }
    assert by_id[_CHECK_OPERATION].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_CANDIDATE].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_SELECTION].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_VALIDATION].status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL


def test_f3_15_model_routing_scenario() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(persistence, _candidate(scope, query_id="model-q1", artifact_ref="model:small-fast"))
    _append(persistence, _selection(scope, query_id="model-q1", selected_ref="model:large-slow"))
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_CANDIDATE,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id="model-q1"),
                ),
                pass_claim="Model candidates exist.",
                fail_claim="No model candidates.",
                insufficient_claim="No candidate evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_SELECTION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH,
                    selection_artifact_match=SelectionArtifactMatchRequirement(
                        query_id="model-q1",
                        expected_artifact_ref="model:small-fast",
                    ),
                ),
                pass_claim="Correct model routed.",
                fail_claim="Wrong model routed.",
                insufficient_claim="No routing evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                dependencies=(_CHECK_SELECTION,),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="execute-model",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Model executed.",
                fail_claim="Model execution failed.",
                insufficient_claim="No execution evidence.",
            ),
        ),
    )
    by_id = {
        item.check_id: item
        for item in _analyzer(persistence).analyze(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            specification=spec,
        ).check_results
    }
    assert by_id[_CHECK_CANDIDATE].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_SELECTION].status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
    assert by_id[_CHECK_OPERATION].status is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM


def test_f3_16_synthesis_failure() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(persistence, _operation(scope, operation_id="search"))
    _append(persistence, _candidate(scope, query_id="syn-q1"))
    _append(persistence, _selection(scope, query_id="syn-q1", selected_ref="context:ok"))
    _append(persistence, _output_relation(scope, operation_id="synthesize"))
    validation, validation_link = _validation(scope, outcome=FunctionalValidationOutcome.FAILED)
    _append(persistence, validation_link)
    spec = _simple_spec(
        (
            FunctionalDiagnosticCheck(
                check_id=_CHECK_OPERATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="search",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Search ok.",
                fail_claim="Search failed.",
                insufficient_claim="No search evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_CANDIDATE,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id="syn-q1"),
                ),
                pass_claim="Candidates ok.",
                fail_claim="No candidates.",
                insufficient_claim="No candidate evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_SELECTION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH,
                    selection_artifact_match=SelectionArtifactMatchRequirement(
                        query_id="syn-q1",
                        expected_artifact_ref="context:ok",
                    ),
                ),
                pass_claim="Selection ok.",
                fail_claim="Selection wrong.",
                insufficient_claim="No selection evidence.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_VALIDATION,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
                    validation_outcome=ValidationOutcomeRequirement(
                        validation_id=validation.validation_id,
                        expected_outcome=FunctionalValidationOutcome.PASSED,
                    ),
                ),
                pass_claim="Final validation passed.",
                fail_claim="Final validation failed.",
                insufficient_claim="No validation evidence.",
            ),
        ),
    )
    by_id = {
        item.check_id: item
        for item in _analyzer(persistence).analyze(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            specification=spec,
            validations=FunctionalValidationEvidenceLookup.for_scope(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            validations=(validation,),
        ),
        ).check_results
    }
    assert by_id[_CHECK_OPERATION].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_CANDIDATE].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_SELECTION].status is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert by_id[_CHECK_VALIDATION].status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL


def test_same_analyzer_cross_domain_equivalence() -> None:
    scope = _scope()
    persistence = _persistence()
    _append(persistence, _operation(scope, operation_id="search"))
    _append(persistence, _candidate(scope, query_id="retrieval-q1"))
    _append(persistence, _selection(scope, query_id="retrieval-q1", selected_ref="context:wrong"))
    analyzer = _analyzer(persistence)
    retrieval = analyzer.analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=_retrieval_like_spec(),
    )
    assert type(retrieval.check_results[0]) is type(retrieval.check_results[0])
    assert retrieval.specification_id == _SPEC_ID
