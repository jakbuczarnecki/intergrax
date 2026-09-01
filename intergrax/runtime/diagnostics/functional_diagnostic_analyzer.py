# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic generic functional diagnostic analyzer (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId
from intergrax.contracts.functional_diagnostic_bounds import (
    MAX_FUNCTIONAL_DIAGNOSTIC_ANALYSIS_LIMITATIONS,
    MAX_FUNCTIONAL_DIAGNOSTIC_LIMITATIONS_PER_RESULT,
    MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticAnalysis,
    FunctionalDiagnosticAnalysisIntegrityError,
    FunctionalDiagnosticCheckResult,
    FunctionalDiagnosticCheckStatus,
    _CONTRADICTION_LIMITATION,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
)
from intergrax.runtime.diagnostics.functional_diagnostic_specification import (
    FunctionalDiagnosticCheck,
    FunctionalDiagnosticRequirement,
    FunctionalDiagnosticRequirementKind,
    FunctionalDiagnosticSpecification,
    validate_functional_diagnostic_specification,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidencePersistenceIntegrityError,
    FunctionalEvidenceQueryRequest,
)
from intergrax.runtime.diagnostics.functional_validation_lookup import (
    FunctionalValidationEvidenceLookup,
)
from intergrax.runtime.observability.functional_validation_evidence import (
    FunctionalValidationOutcome,
)

_DEFAULT_PAGE_SIZE = 100


class _Signal(StrEnum):
    PASS = "pass"
    FAIL = "fail"


@dataclass(slots=True)
class _EvaluationAccumulator:
    pass_refs: list[EventId] = field(default_factory=list)
    fail_refs: list[EventId] = field(default_factory=list)
    saw_pass: bool = False
    saw_fail: bool = False
    contradictory: bool = False

    def observe_pass(self, evidence_id: EventId) -> None:
        if self.saw_fail:
            self.contradictory = True
        self.saw_pass = True
        self._append_ref(self.pass_refs, evidence_id)

    def observe_fail(self, evidence_id: EventId) -> None:
        if self.saw_pass:
            self.contradictory = True
        self.saw_fail = True
        self._append_ref(self.fail_refs, evidence_id)

    @staticmethod
    def _append_ref(refs: list[EventId], evidence_id: EventId) -> None:
        if len(refs) < MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS:
            refs.append(evidence_id)

    def resolve(
        self,
        *,
        pass_claim: str,
        fail_claim: str,
        insufficient_claim: str,
    ) -> tuple[FunctionalDiagnosticCheckStatus, str, tuple[EventId, ...], tuple[str, ...]]:
        if self.contradictory:
            refs = _merge_refs(self.pass_refs, self.fail_refs)
            return (
                FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE,
                insufficient_claim or "Contradictory evidence prevents a proven conclusion.",
                refs,
                (_CONTRADICTION_LIMITATION,),
            )
        if self.saw_pass and not self.saw_fail:
            return (
                FunctionalDiagnosticCheckStatus.PROVEN_PASS,
                pass_claim,
                tuple(self.pass_refs),
                (),
            )
        if self.saw_fail and not self.saw_pass:
            return (
                FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
                fail_claim,
                tuple(self.fail_refs),
                (),
            )
        return (
            FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE,
            insufficient_claim,
            (),
            (),
        )


class FunctionalDiagnosticAnalyzer:
    """
    One generic analyzer for all functional diagnostic profiles.

    Uses bounded paginated evidence queries — never materializes full history.
    """

    def __init__(self, persistence: FunctionalEvidencePersistence) -> None:
        self._persistence = persistence

    def analyze(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        specification: FunctionalDiagnosticSpecification,
        attempt_id: AttemptId | None = None,
        validations: FunctionalValidationEvidenceLookup | None = None,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> FunctionalDiagnosticAnalysis:
        normalized_tenant = _require_tenant_id(tenant_id)
        validated_spec = validate_functional_diagnostic_specification(specification)
        validation_lookup = _resolve_validation_lookup(
            tenant_id=normalized_tenant,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            validations=validations,
        )
        statuses: dict[FunctionalDiagnosticCheckId, FunctionalDiagnosticCheckStatus] = {}
        results_by_id: dict[FunctionalDiagnosticCheckId, FunctionalDiagnosticCheckResult] = {}
        analysis_limitations: list[str] = []

        for check in _topological_evaluation_order(validated_spec.checks):
            gate_status = _dependency_gate_status(check, statuses)
            if gate_status is not None:
                statuses[check.check_id] = gate_status
                results_by_id[check.check_id] = _blocked_result(
                    check=check,
                    status=gate_status,
                )
                continue

            status, claim, refs, limitations = self._evaluate_check(
                tenant_id=normalized_tenant,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                check=check,
                validation_lookup=validation_lookup,
                page_size=page_size,
            )
            statuses[check.check_id] = status
            bounded_limitations = limitations[:MAX_FUNCTIONAL_DIAGNOSTIC_LIMITATIONS_PER_RESULT]
            results_by_id[check.check_id] = FunctionalDiagnosticCheckResult(
                check_id=check.check_id,
                status=status,
                factual_claim=claim,
                supporting_evidence_refs=refs[:MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS],
                limitations=bounded_limitations,
            )

        results = tuple(results_by_id[check.check_id] for check in validated_spec.checks)

        first_failure = _resolve_first_proven_failure(validated_spec.checks, statuses)
        return FunctionalDiagnosticAnalysis(
            tenant_id=normalized_tenant,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            specification_id=validated_spec.specification_id,
            specification_version=validated_spec.version,
            check_results=tuple(results),
            first_proven_failure=first_failure,
            limitations=tuple(analysis_limitations[:MAX_FUNCTIONAL_DIAGNOSTIC_ANALYSIS_LIMITATIONS]),
        )

    def _evaluate_check(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        attempt_id: AttemptId | None,
        check: FunctionalDiagnosticCheck,
        validation_lookup: FunctionalValidationEvidenceLookup,
        page_size: int,
    ) -> tuple[FunctionalDiagnosticCheckStatus, str, tuple[EventId, ...], tuple[str, ...]]:
        requirement = check.requirement
        if requirement.kind is FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME:
            return self._evaluate_validation_requirement(
                check=check,
                validation_lookup=validation_lookup,
            )
        accumulator = _EvaluationAccumulator()
        evidence_kind = _evidence_kind_for_requirement(requirement.kind)
        cursor: str | None = None
        while True:
            page = self._persistence.query_evidence(
                FunctionalEvidenceQueryRequest(
                    tenant_id=tenant_id,
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    kind=evidence_kind,
                    page_size=page_size,
                    cursor=cursor,
                ),
            )
            _assert_page_scope(
                page_tenant=page.tenant_id,
                page_task=page.task_id,
                page_run=page.run_id,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
            )
            for item in page.items:
                self._observe_evidence(
                    accumulator=accumulator,
                    requirement=requirement,
                    evidence=item,
                )
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return accumulator.resolve(
            pass_claim=check.pass_claim,
            fail_claim=check.fail_claim,
            insufficient_claim=check.insufficient_claim,
        )

    def _evaluate_validation_requirement(
        self,
        *,
        check: FunctionalDiagnosticCheck,
        validation_lookup: FunctionalValidationEvidenceLookup,
    ) -> tuple[FunctionalDiagnosticCheckStatus, str, tuple[EventId, ...], tuple[str, ...]]:
        requirement = check.requirement.validation_outcome
        if requirement is None:
            raise FunctionalDiagnosticAnalysisIntegrityError(
                "validation_outcome requirement missing payload",
            )
        validation = validation_lookup.get(requirement.validation_id)
        if validation is None:
            return (
                FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE,
                check.insufficient_claim,
                (),
                (),
            )
        if validation.outcome is requirement.expected_outcome:
            return (
                FunctionalDiagnosticCheckStatus.PROVEN_PASS,
                check.pass_claim,
                (validation.validation_id,),
                (),
            )
        if validation.outcome is FunctionalValidationOutcome.INCONCLUSIVE:
            return (
                FunctionalDiagnosticCheckStatus.INSUFFICIENT_EVIDENCE,
                check.insufficient_claim,
                (validation.validation_id,),
                (),
            )
        return (
            FunctionalDiagnosticCheckStatus.PROVEN_FAIL,
            check.fail_claim,
            (validation.validation_id,),
            (),
        )

    def _observe_evidence(
        self,
        *,
        accumulator: _EvaluationAccumulator,
        requirement: FunctionalDiagnosticRequirement,
        evidence: PlatformFunctionalEvidence,
    ) -> None:
        match requirement.kind:
            case FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS:
                payload = requirement.operation_outcome_status
                if payload is None or evidence.operation_outcome is None:
                    return
                if evidence.provenance.operation_id != payload.operation_id:
                    return
                if evidence.operation_outcome.status is payload.expected_status:
                    accumulator.observe_pass(evidence.evidence_id)
                else:
                    accumulator.observe_fail(evidence.evidence_id)
            case FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS:
                payload = requirement.candidate_exists
                if payload is None or evidence.candidate is None:
                    return
                if evidence.candidate.query_id == payload.query_id:
                    accumulator.observe_pass(evidence.evidence_id)
            case FunctionalDiagnosticRequirementKind.SELECTION_EXISTS:
                payload = requirement.selection_exists
                if payload is None or evidence.selection is None:
                    return
                if evidence.selection.query_id == payload.query_id:
                    accumulator.observe_pass(evidence.evidence_id)
            case FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH:
                payload = requirement.selection_artifact_match
                if payload is None or evidence.selection is None:
                    return
                if evidence.selection.query_id != payload.query_id:
                    return
                selected_ref = evidence.selection.selected_artifact_ref.artifact_ref
                if selected_ref == payload.expected_artifact_ref:
                    accumulator.observe_pass(evidence.evidence_id)
                else:
                    accumulator.observe_fail(evidence.evidence_id)
            case FunctionalDiagnosticRequirementKind.OUTPUT_RELATION_EXISTS:
                payload = requirement.output_relation_exists
                if payload is None or evidence.output_relation is None:
                    return
                if evidence.provenance.operation_id == payload.operation_id:
                    accumulator.observe_pass(evidence.evidence_id)
            case FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME:
                return


def _evidence_kind_for_requirement(
    kind: FunctionalDiagnosticRequirementKind,
) -> PipelineEvidenceKind:
    match kind:
        case FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS:
            return PipelineEvidenceKind.OPERATION_OUTCOME
        case FunctionalDiagnosticRequirementKind.CANDIDATE_EXISTS:
            return PipelineEvidenceKind.CANDIDATE_RANK
        case FunctionalDiagnosticRequirementKind.SELECTION_EXISTS:
            return PipelineEvidenceKind.SELECTION
        case FunctionalDiagnosticRequirementKind.SELECTION_ARTIFACT_MATCH:
            return PipelineEvidenceKind.SELECTION
        case FunctionalDiagnosticRequirementKind.OUTPUT_RELATION_EXISTS:
            return PipelineEvidenceKind.OUTPUT_RELATION
        case FunctionalDiagnosticRequirementKind.VALIDATION_OUTCOME:
            raise FunctionalDiagnosticAnalysisIntegrityError(
                "validation_outcome uses lookup, not evidence pagination",
            )


def _topological_evaluation_order(
    checks: tuple[FunctionalDiagnosticCheck, ...],
) -> tuple[FunctionalDiagnosticCheck, ...]:
    """
    Deterministic topological order for dependency evaluation.

    Tie-breaker: original specification tuple index (stable, operator-visible order).
    """
    specification_index = {check.check_id: index for index, check in enumerate(checks)}
    check_by_id = {check.check_id: check for check in checks}
    in_degree = {check.check_id: len(check.dependencies) for check in checks}
    successors: dict[FunctionalDiagnosticCheckId, list[FunctionalDiagnosticCheckId]] = {
        check.check_id: [] for check in checks
    }
    for check in checks:
        for dependency in check.dependencies:
            successors[dependency].append(check.check_id)

    ready = sorted(
        (check_id for check_id, degree in in_degree.items() if degree == 0),
        key=lambda check_id: specification_index[check_id],
    )
    ordered_ids: list[FunctionalDiagnosticCheckId] = []
    while ready:
        current = ready.pop(0)
        ordered_ids.append(current)
        newly_ready: list[FunctionalDiagnosticCheckId] = []
        for successor in successors[current]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                newly_ready.append(successor)
        if newly_ready:
            ready.extend(newly_ready)
            ready.sort(key=lambda check_id: specification_index[check_id])

    if len(ordered_ids) != len(checks):
        raise FunctionalDiagnosticAnalysisIntegrityError(
            "dependency graph could not be topologically ordered",
        )
    return tuple(check_by_id[check_id] for check_id in ordered_ids)


def _resolve_validation_lookup(
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None,
    validations: FunctionalValidationEvidenceLookup | None,
) -> FunctionalValidationEvidenceLookup:
    if validations is None:
        return FunctionalValidationEvidenceLookup.for_scope(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            validations=(),
        )
    return FunctionalValidationEvidenceLookup.for_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        validations=validations.validations,
    )


def _dependency_gate_status(
    check: FunctionalDiagnosticCheck,
    statuses: dict[FunctionalDiagnosticCheckId, FunctionalDiagnosticCheckStatus],
) -> FunctionalDiagnosticCheckStatus | None:
    if not check.dependencies:
        return None
    for dependency in check.dependencies:
        dependency_status = statuses.get(dependency)
        if dependency_status is None:
            return FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM
        if dependency_status is not FunctionalDiagnosticCheckStatus.PROVEN_PASS:
            return FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM
    return None


def _blocked_result(
    *,
    check: FunctionalDiagnosticCheck,
    status: FunctionalDiagnosticCheckStatus,
) -> FunctionalDiagnosticCheckResult:
    claim = (
        "Evaluation blocked because an upstream dependency did not reach PROVEN_PASS."
        if status is FunctionalDiagnosticCheckStatus.BLOCKED_BY_UPSTREAM
        else check.insufficient_claim
    )
    return FunctionalDiagnosticCheckResult(
        check_id=check.check_id,
        status=status,
        factual_claim=claim,
        supporting_evidence_refs=(),
        limitations=(),
    )


def _resolve_first_proven_failure(
    checks: tuple[FunctionalDiagnosticCheck, ...],
    statuses: dict[FunctionalDiagnosticCheckId, FunctionalDiagnosticCheckStatus],
) -> FunctionalDiagnosticCheckId | None:
    for check in checks:
        if statuses.get(check.check_id) is FunctionalDiagnosticCheckStatus.PROVEN_FAIL:
            return check.check_id
    return None


def _merge_refs(left: list[EventId], right: list[EventId]) -> tuple[EventId, ...]:
    merged: list[EventId] = []
    for evidence_id in (*left, *right):
        if evidence_id not in merged:
            merged.append(evidence_id)
        if len(merged) >= MAX_FUNCTIONAL_DIAGNOSTIC_SUPPORTING_REFS:
            break
    return tuple(merged)


def _assert_page_scope(
    *,
    page_tenant: str,
    page_task: TaskId,
    page_run: RunId,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
) -> None:
    if page_tenant != tenant_id:
        raise FunctionalDiagnosticAnalysisIntegrityError("tenant scope mismatch in evidence page")
    if page_task != task_id or page_run != run_id:
        raise FunctionalDiagnosticAnalysisIntegrityError("execution scope mismatch in evidence page")


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise FunctionalDiagnosticAnalysisIntegrityError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized:
        raise FunctionalDiagnosticAnalysisIntegrityError("tenant_id must be non-empty")
    if tenant_id != normalized:
        raise FunctionalDiagnosticAnalysisIntegrityError(
            "tenant_id must not contain leading or trailing whitespace",
        )
    return normalized


__all__ = ["FunctionalDiagnosticAnalyzer"]
