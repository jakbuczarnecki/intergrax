# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysisIntegrityError,
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
    SelectionExistsRequirement,
    ValidationOutcomeRequirement,
    validate_functional_diagnostic_specification,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceProvenance,
    PipelineEvidenceScope,
    PipelineOperationOutcomeFact,
    PipelineOperationStatus,
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

pytestmark = pytest.mark.unit

_TEST_CURSOR_SECRET = b"x" * 32
_BASE_TIME = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
_SPEC_ID = FunctionalDiagnosticSpecificationId("fdspec_a0000000000000000000000000000001")
_CHECK_A = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000001")
_CHECK_B = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000002")
_CHECK_C = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000003")
_CHECK_D = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000004")
_CHECK_INDEPENDENT = FunctionalDiagnosticCheckId("fdcheck_a0000000000000000000000000000005")


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


def _operation(scope: PipelineEvidenceScope, *, operation_id: str) -> PlatformFunctionalEvidence:
    return PlatformFunctionalEvidence(
        evidence_id=mint_event_id(),
        kind=PipelineEvidenceKind.OPERATION_OUTCOME,
        scope=scope,
        provenance=PipelineEvidenceProvenance(
            producer_component="diag.test",
            operation_id=operation_id,
            recorded_at=_BASE_TIME,
        ),
        operation_outcome=PipelineOperationOutcomeFact(
            operation_name=operation_id,
            status=PipelineOperationStatus.SUCCEEDED,
        ),
    )


_UNSET_ATTEMPT: object = object()


def _validation(
    scope: PipelineEvidenceScope,
    *,
    outcome: FunctionalValidationOutcome = FunctionalValidationOutcome.PASSED,
    tenant_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None | object = _UNSET_ATTEMPT,
    idempotency_key: str = "attempt-1",
) -> FunctionalValidationEvidence:
    resolved_attempt = scope.attempt_id if attempt_id is _UNSET_ATTEMPT else attempt_id
    correlation = DiagnosticExecutionCorrelation(
        tenant_id=tenant_id if tenant_id is not None else scope.tenant_id,
        task_id=task_id if task_id is not None else scope.task_id,
        run_id=run_id if run_id is not None else scope.run_id,
        attempt_id=resolved_attempt,
    )
    return FunctionalValidationEvidence(
        validation_id=functional_validation_evidence_id(
            validator_id="oracle.v1",
            validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
            correlation=correlation,
            idempotency_key=idempotency_key,
        ),
        validator=FunctionalValidatorRef(validator_id="oracle.v1"),
        validation_kind=FunctionalValidationKind.ORACLE_ASSERTION,
        outcome=outcome,
        correlation=correlation,
        expected_actual_relation=ExpectedActualRelation.CONTAINS,
    )


def _validation_spec(
    validation_id: str,
    *,
    check_id: FunctionalDiagnosticCheckId = _CHECK_A,
) -> FunctionalDiagnosticSpecification:
    return FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=(
            FunctionalDiagnosticCheck(
                check_id=check_id,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
                    validation_outcome=ValidationOutcomeRequirement(
                        validation_id=validation_id,
                        expected_outcome=FunctionalValidationOutcome.PASSED,
                    ),
                ),
                pass_claim="Validation passed.",
                fail_claim="Validation failed.",
                insufficient_claim="No validation evidence.",
            ),
        ),
    )


def _lookup(
    scope: PipelineEvidenceScope,
    *validations: FunctionalValidationEvidence,
) -> FunctionalValidationEvidenceLookup:
    return FunctionalValidationEvidenceLookup.for_scope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        validations=validations,
    )


def _statuses(
    scope: PipelineEvidenceScope,
    spec: FunctionalDiagnosticSpecification,
    *,
    validations: FunctionalValidationEvidenceLookup | None = None,
    attempt_id: str | None = None,
) -> dict[FunctionalDiagnosticCheckId, FunctionalDiagnosticCheckStatus]:
    analysis = FunctionalDiagnosticAnalyzer(_persistence()).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=attempt_id if attempt_id is not None else scope.attempt_id,
        specification=spec,
        validations=validations,
    )
    return {item.check_id: item.status for item in analysis.check_results}


def test_v1_correct_scope_validation_passes() -> None:
    scope = _scope()
    validation = _validation(scope, outcome=FunctionalValidationOutcome.PASSED)
    statuses = _statuses(
        scope,
        _validation_spec(validation.validation_id),
        validations=_lookup(scope, validation),
    )
    assert statuses[_CHECK_A] is FunctionalDiagnosticCheckStatus.PROVEN_PASS


def test_v2_wrong_run_fails_closed() -> None:
    scope = _scope()
    wrong_run = mint_run_id()
    validation = _validation(scope, run_id=wrong_run)
    with pytest.raises(FunctionalDiagnosticAnalysisIntegrityError):
        _lookup(scope, validation)


def test_v3_wrong_task_fails_closed() -> None:
    scope = _scope()
    validation = _validation(scope, task_id=mint_task_id())
    with pytest.raises(FunctionalDiagnosticAnalysisIntegrityError):
        _lookup(scope, validation)


def test_v4_wrong_tenant_fails_closed() -> None:
    scope = _scope()
    validation = _validation(scope, tenant_id="tenant-b")
    with pytest.raises(FunctionalDiagnosticAnalysisIntegrityError):
        _lookup(scope, validation)


def test_v5_wrong_attempt_fails_closed() -> None:
    scope = _scope(attempt_id=mint_attempt_id())
    validation = _validation(scope, attempt_id=mint_attempt_id())
    with pytest.raises(FunctionalDiagnosticAnalysisIntegrityError):
        _lookup(scope, validation)


def test_v6_missing_validation_is_insufficient() -> None:
    scope = _scope()
    validation = _validation(scope)
    statuses = _statuses(
        scope,
        _validation_spec(validation.validation_id),
        validations=_lookup(scope),
    )
    assert statuses[_CHECK_A] is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE


def test_v7_duplicate_identical_validation_is_idempotent() -> None:
    scope = _scope()
    validation = _validation(scope)
    lookup = _lookup(scope, validation, validation)
    statuses = _statuses(
        scope,
        _validation_spec(validation.validation_id),
        validations=lookup,
    )
    assert statuses[_CHECK_A] is FunctionalDiagnosticCheckStatus.PROVEN_PASS


def test_v8_duplicate_conflicting_validation_fails_closed() -> None:
    scope = _scope()
    validation_pass = _validation(scope, outcome=FunctionalValidationOutcome.PASSED)
    validation_fail = validation_pass.model_copy(
        update={"outcome": FunctionalValidationOutcome.FAILED},
    )
    with pytest.raises(FunctionalDiagnosticAnalysisIntegrityError):
        _lookup(scope, validation_pass, validation_fail)


def _ab_dependency_spec(*, reverse: bool) -> FunctionalDiagnosticSpecification:
    check_a = FunctionalDiagnosticCheck(
        check_id=_CHECK_A,
        requirement=FunctionalDiagnosticRequirement(
            kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
            operation_outcome_status=OperationOutcomeStatusRequirement(
                operation_id="op-a",
                expected_status=PipelineOperationStatus.SUCCEEDED,
            ),
        ),
        pass_claim="A pass.",
        fail_claim="A fail.",
        insufficient_claim="A missing.",
    )
    check_b = FunctionalDiagnosticCheck(
        check_id=_CHECK_B,
        dependencies=(_CHECK_A,),
        requirement=FunctionalDiagnosticRequirement(
            kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
            candidate_exists=CandidateExistsRequirement(query_id="q-b"),
        ),
        pass_claim="B pass.",
        fail_claim="B fail.",
        insufficient_claim="B missing.",
    )
    checks = (check_b, check_a) if reverse else (check_a, check_b)
    return FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=checks,
    )


def test_d1_forward_dependency_order() -> None:
    scope = _scope()
    persistence = _persistence()
    persistence.append(_operation(scope, operation_id="op-a"))
    analysis = FunctionalDiagnosticAnalyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=_ab_dependency_spec(reverse=False),
    )
    statuses = {item.check_id: item.status for item in analysis.check_results}
    assert statuses[_CHECK_A] is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert statuses[_CHECK_B] is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE


def test_d2_reverse_tuple_order_matches_forward_semantics() -> None:
    scope = _scope()
    persistence = _persistence()
    persistence.append(_operation(scope, operation_id="op-a"))
    analyzer = FunctionalDiagnosticAnalyzer(persistence)
    forward = {
        item.check_id: item.status
        for item in analyzer.analyze(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            specification=_ab_dependency_spec(reverse=False),
        ).check_results
    }
    reverse = {
        item.check_id: item.status
        for item in analyzer.analyze(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            specification=_ab_dependency_spec(reverse=True),
        ).check_results
    }
    assert forward == reverse


def test_d3_diamond_dependency_graph() -> None:
    scope = _scope()
    persistence = _persistence()
    persistence.append(_operation(scope, operation_id="op-a"))
    spec = FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=(
            FunctionalDiagnosticCheck(
                check_id=_CHECK_A,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="op-a",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="A pass.",
                fail_claim="A fail.",
                insufficient_claim="A missing.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_B,
                dependencies=(_CHECK_A,),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.SELECTION_EXISTS,
                    selection_exists=SelectionExistsRequirement(query_id="q-b"),
                ),
                pass_claim="B pass.",
                fail_claim="B fail.",
                insufficient_claim="B missing.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_C,
                dependencies=(_CHECK_A,),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.SELECTION_EXISTS,
                    selection_exists=SelectionExistsRequirement(query_id="q-c"),
                ),
                pass_claim="C pass.",
                fail_claim="C fail.",
                insufficient_claim="C missing.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_D,
                dependencies=(_CHECK_B, _CHECK_C),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OUTPUT_RELATION_EXISTS,
                    output_relation_exists=OutputRelationExistsRequirement(operation_id="op-d"),
                ),
                pass_claim="D pass.",
                fail_claim="D fail.",
                insufficient_claim="D missing.",
            ),
        ),
    )
    analysis = FunctionalDiagnosticAnalyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    )
    statuses = {item.check_id: item.status for item in analysis.check_results}
    assert statuses[_CHECK_A] is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert statuses[_CHECK_B] is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE
    assert statuses[_CHECK_C] is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE
    assert statuses[_CHECK_D] is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM


def test_d4_independent_branch_continues_after_failure() -> None:
    scope = _scope()
    persistence = _persistence()
    persistence.append(
        PlatformFunctionalEvidence(
            evidence_id=mint_event_id(),
            kind=PipelineEvidenceKind.OPERATION_OUTCOME,
            scope=scope,
            provenance=PipelineEvidenceProvenance(
                producer_component="diag.test",
                operation_id="op-a",
                recorded_at=_BASE_TIME,
            ),
            operation_outcome=PipelineOperationOutcomeFact(
                operation_name="op-a",
                status=PipelineOperationStatus.FAILED,
            ),
        ),
    )
    persistence.append(_operation(scope, operation_id="op-independent"))
    spec = FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=(
            FunctionalDiagnosticCheck(
                check_id=_CHECK_A,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="op-a",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="A pass.",
                fail_claim="A fail.",
                insufficient_claim="A missing.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_INDEPENDENT,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="op-independent",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="Independent pass.",
                fail_claim="Independent fail.",
                insufficient_claim="Independent missing.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_C,
                dependencies=(_CHECK_A,),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id="q-c"),
                ),
                pass_claim="C pass.",
                fail_claim="C fail.",
                insufficient_claim="C missing.",
            ),
        ),
    )
    analysis = FunctionalDiagnosticAnalyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec,
    )
    statuses = {item.check_id: item.status for item in analysis.check_results}
    assert statuses[_CHECK_A] is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
    assert statuses[_CHECK_INDEPENDENT] is FunctionalDiagnosticCheckStatus.PROVEN_PASS
    assert statuses[_CHECK_C] is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM


def test_d5_upstream_insufficient_blocks_downstream() -> None:
    scope = _scope()
    spec = FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=(
            FunctionalDiagnosticCheck(
                check_id=_CHECK_A,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
                    candidate_exists=CandidateExistsRequirement(query_id="q-a"),
                ),
                pass_claim="A pass.",
                fail_claim="A fail.",
                insufficient_claim="A missing.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_B,
                dependencies=(_CHECK_A,),
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.SELECTION_EXISTS,
                    selection_exists=SelectionExistsRequirement(query_id="q-b"),
                ),
                pass_claim="B pass.",
                fail_claim="B fail.",
                insufficient_claim="B missing.",
            ),
        ),
    )
    statuses = _statuses(scope, spec)
    assert statuses[_CHECK_A] is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE
    assert statuses[_CHECK_B] is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM


def test_d6_deterministic_across_equivalent_orderings() -> None:
    scope = _scope()
    persistence = _persistence()
    persistence.append(_operation(scope, operation_id="op-a"))
    persistence.append(_operation(scope, operation_id="op-b"))
    spec_a_first = FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=(
            FunctionalDiagnosticCheck(
                check_id=_CHECK_A,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="op-a",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="A pass.",
                fail_claim="A fail.",
                insufficient_claim="A missing.",
            ),
            FunctionalDiagnosticCheck(
                check_id=_CHECK_B,
                requirement=FunctionalDiagnosticRequirement(
                    kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                    operation_outcome_status=OperationOutcomeStatusRequirement(
                        operation_id="op-b",
                        expected_status=PipelineOperationStatus.SUCCEEDED,
                    ),
                ),
                pass_claim="B pass.",
                fail_claim="B fail.",
                insufficient_claim="B missing.",
            ),
        ),
    )
    spec_b_first = FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=(spec_a_first.checks[1], spec_a_first.checks[0]),
    )
    first = FunctionalDiagnosticAnalyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec_a_first,
    )
    second = FunctionalDiagnosticAnalyzer(persistence).analyze(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        specification=spec_b_first,
    )
    assert {item.check_id: item.status for item in first.check_results} == {
        item.check_id: item.status for item in second.check_results
    }


@pytest.mark.parametrize(
    ("kind", "valid_payload", "extra_field", "extra_value"),
    [
        (
            FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
            "operation_outcome_status",
            "candidate_exists",
            CandidateExistsRequirement(query_id="q1"),
        ),
        (
            FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS,
            "candidate_exists",
            "selection_exists",
            SelectionExistsRequirement(query_id="q1"),
        ),
        (
            FunctionalDiagnosticRequirementKind.SELECTION_EXISTS,
            "selection_exists",
            "candidate_exists",
            CandidateExistsRequirement(query_id="q1"),
        ),
        (
            FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME,
            "validation_outcome",
            "candidate_exists",
            CandidateExistsRequirement(query_id="q1"),
        ),
    ],
)
def test_requirement_exact_one_payload_rejects_extra(
    kind: FunctionalDiagnosticRequirementKind,
    valid_payload: str,
    extra_field: str,
    extra_value: object,
) -> None:
    base_kwargs: dict[str, object] = {"kind": kind}
    if valid_payload == "operation_outcome_status":
        base_kwargs["operation_outcome_status"] = OperationOutcomeStatusRequirement(
            operation_id="op",
            expected_status=PipelineOperationStatus.SUCCEEDED,
        )
    elif valid_payload == "candidate_exists":
        base_kwargs["candidate_exists"] = CandidateExistsRequirement(query_id="q1")
    elif valid_payload == "selection_exists":
        base_kwargs["selection_exists"] = SelectionExistsRequirement(query_id="q1")
    elif valid_payload == "validation_outcome":
        base_kwargs["validation_outcome"] = ValidationOutcomeRequirement(
            validation_id=mint_event_id(),
            expected_outcome=FunctionalValidationOutcome.PASSED,
        )
    base_kwargs[extra_field] = extra_value
    requirement = FunctionalDiagnosticRequirement(**base_kwargs)
    spec = FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=(
            FunctionalDiagnosticCheck(
                check_id=_CHECK_A,
                requirement=requirement,
                pass_claim="p",
                fail_claim="f",
                insufficient_claim="i",
            ),
        ),
    )
    with pytest.raises(FunctionalDiagnosticSpecificationIntegrityError):
        validate_functional_diagnostic_specification(spec)


@pytest.mark.parametrize(
    "kind",
    list(FunctionalDiagnosticRequirementKind),
)
def test_requirement_exact_one_payload_rejects_missing(kind: FunctionalDiagnosticRequirementKind) -> None:
    requirement = FunctionalDiagnosticRequirement(kind=kind)
    spec = FunctionalDiagnosticSpecification(
        specification_id=_SPEC_ID,
        version=1,
        checks=(
            FunctionalDiagnosticCheck(
                check_id=_CHECK_A,
                requirement=requirement,
                pass_claim="p",
                fail_claim="f",
                insufficient_claim="i",
            ),
        ),
    )
    with pytest.raises(FunctionalDiagnosticSpecificationIntegrityError):
        validate_functional_diagnostic_specification(spec)


def test_run_level_validation_insufficient_for_attempt_scoped_analysis() -> None:
    attempt = mint_attempt_id()
    scope = _scope(attempt_id=attempt)
    validation = _validation(scope, attempt_id=None)
    statuses = _statuses(
        scope,
        _validation_spec(validation.validation_id),
        validations=_lookup(scope, validation),
        attempt_id=attempt,
    )
    assert statuses[_CHECK_A] is FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE
