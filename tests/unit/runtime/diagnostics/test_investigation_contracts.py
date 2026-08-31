# © Artur Czarnecki. All rights reserved.

"""DIAG-8B — platform-to-investigator boundary contract tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.evidence_claims import (
    ClaimKind,
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
    mint_evidence_claim_id,
    validate_evidence_reference_id,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticCertainty,
    DiagnosticFinding,
    DiagnosticFindingKind,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticGroupingProvenance,
    DiagnosticOccurrenceReadStatus,
    DiagnosticProblemOccurrenceView,
    DiagnosticProblemSummary,
    DiagnosticReadUnavailableReason,
)
from intergrax.runtime.diagnostics.investigation_contracts import (
    IncidentInvestigationInput,
    IncidentInvestigationIntegrityError,
    IncidentInvestigationProblemContext,
    InvestigationConclusion,
    InvestigationConclusionIntegrityError,
    InvestigationConclusionStatus,
    incident_investigation_input_from_problem_details,
    validate_incident_investigation_input,
    validate_investigation_conclusion,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyKind, LifecycleAnomalyScope
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingMethod,
    problem_grouping_subject_ref_for_application_instance,
    problem_grouping_subject_ref_for_execution,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    ProblemOccurrenceAggregateHealth,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    ProblemId,
    ProblemReconciliationKeyKind,
    ProblemStatus,
    mint_problem_id,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_OBSERVED_AT = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)


def _grouping_provenance() -> DiagnosticGroupingProvenance:
    return DiagnosticGroupingProvenance(
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        reconciliation_key_kind=ProblemReconciliationKeyKind.DETERMINISTIC,
        deterministic_signature=None,
    )


def _problem_summary(
    *,
    tenant_id: str = _TENANT_A,
    problem_id: ProblemId | None = None,
) -> DiagnosticProblemSummary:
    resolved_problem_id = problem_id or mint_problem_id()
    return DiagnosticProblemSummary(
        problem_id=resolved_problem_id,
        tenant_id=tenant_id,
        status=ProblemStatus.OPEN,
        first_seen_at=_OBSERVED_AT,
        last_seen_at=_OBSERVED_AT,
        occurrence_count=1,
        grouping_provenance=_grouping_provenance(),
        occurrence_aggregate_health=ProblemOccurrenceAggregateHealth.CONSISTENT,
    )


def _execution_occurrence(*, tenant_id: str = _TENANT_A) -> DiagnosticProblemOccurrenceView:
    task_id = mint_task_id()
    run_id = mint_run_id()
    assessment = DiagnosticAssessment(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        findings=(),
        limitations=(),
    )
    return DiagnosticProblemOccurrenceView(
        subject_ref=problem_grouping_subject_ref_for_execution(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        ),
        observed_at=_OBSERVED_AT,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        read_status=DiagnosticOccurrenceReadStatus.AVAILABLE,
        assessment=assessment,
        unavailable_reason=None,
    )


def _application_occurrence(*, tenant_id: str = _TENANT_A) -> DiagnosticProblemOccurrenceView:
    return DiagnosticProblemOccurrenceView(
        subject_ref=problem_grouping_subject_ref_for_application_instance(
            tenant_id=tenant_id,
            application_id="app-demo",
            instance_id="instance-1",
        ),
        observed_at=_OBSERVED_AT,
        strategy_id=STRATEGY_ID,
        strategy_version=STRATEGY_VERSION,
        method=ProblemGroupingMethod.DETERMINISTIC,
        read_status=DiagnosticOccurrenceReadStatus.UNAVAILABLE,
        assessment=None,
        unavailable_reason=DiagnosticReadUnavailableReason.NON_EXECUTION_SUBJECT,
    )


def _problem_context(
    *,
    tenant_id: str = _TENANT_A,
    problem_id: ProblemId | None = None,
    occurrences: tuple[DiagnosticProblemOccurrenceView, ...] | None = None,
) -> IncidentInvestigationProblemContext:
    return IncidentInvestigationProblemContext(
        problem=_problem_summary(tenant_id=tenant_id, problem_id=problem_id),
        occurrences=occurrences or (_execution_occurrence(tenant_id=tenant_id),),
    )


def _investigation_input(
    *,
    tenant_id: str = _TENANT_A,
    contexts: tuple[IncidentInvestigationProblemContext, ...] | None = None,
) -> IncidentInvestigationInput:
    return IncidentInvestigationInput(
        tenant_id=tenant_id,
        problem_contexts=contexts
        or (
            _problem_context(tenant_id=tenant_id),
        ),
    )


def test_input_retains_canonical_problem_id_and_summary_types() -> None:
    problem_id = mint_problem_id()
    validated = validate_incident_investigation_input(
        _investigation_input(
            contexts=(
                _problem_context(problem_id=problem_id),
            ),
        )
    )

    assert type(validated.problem_contexts[0].problem.problem_id) is str
    assert validated.problem_contexts[0].problem.problem_id == problem_id
    assert isinstance(validated.problem_contexts[0].problem, DiagnosticProblemSummary)


def test_execution_occurrence_accepted_without_top_level_task_or_run_ids() -> None:
    validated = validate_incident_investigation_input(_investigation_input())

    assert not hasattr(validated, "task_id")
    assert not hasattr(validated, "run_id")
    occurrence = validated.problem_contexts[0].occurrences[0]
    assert occurrence.read_status is DiagnosticOccurrenceReadStatus.AVAILABLE
    assert occurrence.assessment is not None


def test_application_instance_occurrence_with_non_execution_subject_accepted() -> None:
    validated = validate_incident_investigation_input(
        _investigation_input(
            contexts=(
                _problem_context(
                    occurrences=(_application_occurrence(),),
                ),
            ),
        )
    )

    occurrence = validated.problem_contexts[0].occurrences[0]
    assert occurrence.read_status is DiagnosticOccurrenceReadStatus.UNAVAILABLE
    assert occurrence.unavailable_reason is DiagnosticReadUnavailableReason.NON_EXECUTION_SUBJECT
    assert occurrence.assessment is None


def test_mixed_tenant_rejected() -> None:
    with pytest.raises(IncidentInvestigationIntegrityError, match="tenant_id"):
        validate_incident_investigation_input(
            _investigation_input(
                contexts=(
                    _problem_context(
                        tenant_id=_TENANT_B,
                        occurrences=(_execution_occurrence(tenant_id=_TENANT_A),),
                    ),
                ),
            )
        )


def test_duplicate_problem_id_rejected() -> None:
    shared_problem_id = mint_problem_id()
    with pytest.raises(IncidentInvestigationIntegrityError, match="duplicate ProblemId"):
        validate_incident_investigation_input(
            _investigation_input(
                contexts=(
                    _problem_context(problem_id=shared_problem_id),
                    _problem_context(problem_id=shared_problem_id),
                ),
            )
        )


def test_empty_input_rejected() -> None:
    with pytest.raises(IncidentInvestigationIntegrityError, match="at least one problem"):
        validate_incident_investigation_input(
            IncidentInvestigationInput(
                tenant_id=_TENANT_A,
                problem_contexts=(),
            )
        )


def test_investigation_conclusion_uses_separate_status_not_problem_lifecycle() -> None:
    problem_id = mint_problem_id()
    claim = EvidenceBackedClaim(
        claim_id=mint_evidence_claim_id(),
        statement="domain diagnosis supported in scenario scope",
        claim_kind=ClaimKind("incident.domain_diagnosis"),
        supporting_evidence_ids=(validate_evidence_reference_id("evidence.demo.1"),),
        resolution=ClaimResolution.SUPPORTED,
    )
    conclusion = validate_investigation_conclusion(
        InvestigationConclusion(
            status=InvestigationConclusionStatus.SUPPORTED,
            investigated_problem_ids=(problem_id,),
            claim_set=EvidenceClaimSet(claims=(claim,)),
            summary="supported domain diagnosis",
        )
    )

    assert conclusion.status is InvestigationConclusionStatus.SUPPORTED
    assert conclusion.status is not ProblemStatus.RESOLVED
    assert DiagnosticCertainty.PROVEN.value not in {member.value for member in InvestigationConclusionStatus}
    assert not hasattr(conclusion, "root_cause_proven")


def test_investigation_conclusion_unresolved_and_not_accepted_states_exist() -> None:
    problem_id = mint_problem_id()
    unresolved = validate_investigation_conclusion(
        InvestigationConclusion(
            status=InvestigationConclusionStatus.UNRESOLVED,
            investigated_problem_ids=(problem_id,),
        )
    )
    not_accepted = validate_investigation_conclusion(
        InvestigationConclusion(
            status=InvestigationConclusionStatus.NOT_ACCEPTED,
            investigated_problem_ids=(problem_id,),
        )
    )

    assert unresolved.status is InvestigationConclusionStatus.UNRESOLVED
    assert not_accepted.status is InvestigationConclusionStatus.NOT_ACCEPTED


def test_investigation_conclusion_requires_problem_ids() -> None:
    with pytest.raises(InvestigationConclusionIntegrityError, match="at least one investigated"):
        validate_investigation_conclusion(
            InvestigationConclusion(
                status=InvestigationConclusionStatus.UNRESOLVED,
                investigated_problem_ids=(),
            )
        )


def test_input_from_problem_details_preserves_occurrence_assessment_and_limitations_shape() -> None:
    from intergrax.runtime.diagnostics.diagnostic_read_models import DiagnosticProblemDetail

    task_id = mint_task_id()
    run_id = mint_run_id()
    problem_id = mint_problem_id()
    detail = DiagnosticProblemDetail(
        problem_id=problem_id,
        tenant_id=_TENANT_A,
        status=ProblemStatus.OPEN,
        first_seen_at=_OBSERVED_AT,
        last_seen_at=_OBSERVED_AT,
        occurrence_count=1,
        record_version=1,
        grouping_provenance=_grouping_provenance(),
        occurrence_aggregate_health=ProblemOccurrenceAggregateHealth.CONSISTENT,
        occurrences=(
            DiagnosticProblemOccurrenceView(
                subject_ref=problem_grouping_subject_ref_for_execution(
                    tenant_id=_TENANT_A,
                    task_id=task_id,
                    run_id=run_id,
                ),
                observed_at=_OBSERVED_AT,
                strategy_id=STRATEGY_ID,
                strategy_version=STRATEGY_VERSION,
                method=ProblemGroupingMethod.DETERMINISTIC,
                read_status=DiagnosticOccurrenceReadStatus.AVAILABLE,
                assessment=DiagnosticAssessment(
                    tenant_id=_TENANT_A,
                    task_id=task_id,
                    run_id=run_id,
                    findings=(
                        DiagnosticFinding(
                            kind=DiagnosticFindingKind.EVENT_AFTER_TERMINAL,
                            scope=LifecycleAnomalyScope.EXECUTION,
                            attempt_id=None,
                            certainty=DiagnosticCertainty.PROVEN,
                            claim="event after terminal",
                            source_anomaly_kind=LifecycleAnomalyKind.EVENT_AFTER_TERMINAL,
                            supporting_event_ids=(),
                            supporting_evidence_ids=(),
                            supporting_positions=(),
                        ),
                    ),
                    limitations=(),
                ),
                unavailable_reason=None,
            ),
        ),
        returned_occurrence_count=1,
        total_occurrence_count=1,
        is_occurrences_truncated=False,
    )

    validated = incident_investigation_input_from_problem_details(
        tenant_id=_TENANT_A,
        details=(detail,),
    )

    assessment = validated.problem_contexts[0].occurrences[0].assessment
    assert assessment is not None
    assert assessment.findings[0].kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    assert assessment.limitations == ()
