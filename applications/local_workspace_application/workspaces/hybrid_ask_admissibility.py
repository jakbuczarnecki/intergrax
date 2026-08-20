# © Artur Czarnecki. All rights reserved.

"""Deterministic Evidence Admissibility evaluation for Workspace Ask V2."""

from __future__ import annotations

from datetime import datetime

from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeQueryExecutionResultV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    EvidenceAdmissibilityResultV1,
    EvidenceAdmissibilityStatusV1,
    EvidenceTemporalMetadataV1,
    IndexedWorkspaceEvidenceV1,
    LiveWorkspaceEvidenceV1,
    RequirementAdmissibilityReasonCodeV1,
    RequiredEvidenceEvaluationV1,
    RequirementEvaluationStatusV1,
)
from local_workspace_application.workspaces.hybrid_ask_policy import (
    IndexedEvidenceRequirementV1,
    LiveEvidenceRequirementV1,
    RequiredEvidenceObligationV1,
    ValidatedEvidencePlanV1,
)
from intergrax.runtime.evidence.obligation_derivation_contracts import (
    MaxAgeTemporalConstraintV1,
    PointInTimeEvidenceTemporalV1,
    TemporalConstraintV1,
    ValidAtTemporalConstraintV1,
    ValidityIntervalEvidenceTemporalV1,
)


def _satisfies_temporal_constraint(
    *,
    constraint: TemporalConstraintV1 | None,
    evidence_temporal: EvidenceTemporalMetadataV1 | None,
    evaluated_at: datetime,
) -> bool:
    if constraint is None:
        return True
    if evidence_temporal is None:
        return False
    if isinstance(constraint, MaxAgeTemporalConstraintV1):
        if not isinstance(evidence_temporal, PointInTimeEvidenceTemporalV1):
            return False
        effective_at = evidence_temporal.effective_at
        if effective_at > evaluated_at:
            return False
        age_seconds = (evaluated_at - effective_at).total_seconds()
        return 0 <= age_seconds <= constraint.max_age_seconds
    if isinstance(constraint, ValidAtTemporalConstraintV1):
        if not isinstance(evidence_temporal, ValidityIntervalEvidenceTemporalV1):
            return False
        return (
            evidence_temporal.valid_from
            <= evaluated_at
            <= evidence_temporal.valid_until
        )
    return False


def _evaluate_indexed_requirement(
    requirement: IndexedEvidenceRequirementV1,
    indexed_evidence: tuple[IndexedWorkspaceEvidenceV1, ...],
    *,
    evaluated_at: datetime,
) -> RequiredEvidenceEvaluationV1:
    matched_ids: list[str] = []
    for item in indexed_evidence:
        if requirement.indexed_source_binding_id is not None:
            if item.indexed_source_binding_id != requirement.indexed_source_binding_id:
                continue
        matched_ids.append(item.evidence_id)
    if not matched_ids:
        reason = (
            RequirementAdmissibilityReasonCodeV1.INDEXED_BINDING_MISMATCH
            if requirement.indexed_source_binding_id is not None
            else RequirementAdmissibilityReasonCodeV1.NO_MATCHING_EVIDENCE
        )
        return RequiredEvidenceEvaluationV1(
            requirement_id=requirement.requirement_id,
            status=RequirementEvaluationStatusV1.UNSATISFIED,
            reason_code=reason,
        )
    temporally_valid_ids = [
        evidence_id
        for evidence_id in matched_ids
        if _satisfies_temporal_constraint(
            constraint=requirement.temporal_constraint,
            evidence_temporal=next(
                item.temporal
                for item in indexed_evidence
                if item.evidence_id == evidence_id
            ),
            evaluated_at=evaluated_at,
        )
    ]
    if temporally_valid_ids:
        return RequiredEvidenceEvaluationV1(
            requirement_id=requirement.requirement_id,
            status=RequirementEvaluationStatusV1.SATISFIED,
            matched_evidence_ids=tuple(temporally_valid_ids),
        )
    if requirement.temporal_constraint is not None:
        return RequiredEvidenceEvaluationV1(
            requirement_id=requirement.requirement_id,
            status=RequirementEvaluationStatusV1.UNSATISFIED,
            reason_code=RequirementAdmissibilityReasonCodeV1.EVIDENCE_TEMPORALLY_INVALID,
        )
    return RequiredEvidenceEvaluationV1(
        requirement_id=requirement.requirement_id,
        status=RequirementEvaluationStatusV1.SATISFIED,
        matched_evidence_ids=tuple(matched_ids),
    )


def _evaluate_live_requirement(
    requirement: LiveEvidenceRequirementV1,
    live_evidence: tuple[LiveWorkspaceEvidenceV1, ...],
    *,
    evaluated_at: datetime,
) -> RequiredEvidenceEvaluationV1:
    matched_items = tuple(
        item for item in live_evidence if item.call_id == requirement.call_id
    )
    if not matched_items:
        has_other_call_evidence = any(
            item.call_id != requirement.call_id for item in live_evidence
        )
        reason = (
            RequirementAdmissibilityReasonCodeV1.LIVE_CALL_MISMATCH
            if has_other_call_evidence
            else RequirementAdmissibilityReasonCodeV1.NO_MATCHING_EVIDENCE
        )
        return RequiredEvidenceEvaluationV1(
            requirement_id=requirement.requirement_id,
            status=RequirementEvaluationStatusV1.UNSATISFIED,
            reason_code=reason,
        )
    temporally_valid_ids = tuple(
        item.evidence_id
        for item in matched_items
        if _satisfies_temporal_constraint(
            constraint=requirement.temporal_constraint,
            evidence_temporal=item.temporal,
            evaluated_at=evaluated_at,
        )
    )
    if temporally_valid_ids:
        return RequiredEvidenceEvaluationV1(
            requirement_id=requirement.requirement_id,
            status=RequirementEvaluationStatusV1.SATISFIED,
            matched_evidence_ids=temporally_valid_ids,
        )
    if requirement.temporal_constraint is not None:
        return RequiredEvidenceEvaluationV1(
            requirement_id=requirement.requirement_id,
            status=RequirementEvaluationStatusV1.UNSATISFIED,
            reason_code=RequirementAdmissibilityReasonCodeV1.EVIDENCE_TEMPORALLY_INVALID,
        )
    return RequiredEvidenceEvaluationV1(
        requirement_id=requirement.requirement_id,
        status=RequirementEvaluationStatusV1.SATISFIED,
        matched_evidence_ids=tuple(item.evidence_id for item in matched_items),
    )


def evaluate_evidence_admissibility(
    *,
    obligations: tuple[RequiredEvidenceObligationV1, ...],
    indexed_evidence: tuple[IndexedWorkspaceEvidenceV1, ...],
    live_evidence: tuple[LiveWorkspaceEvidenceV1, ...],
    evaluated_at: datetime,
) -> EvidenceAdmissibilityResultV1:
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("evaluated_at_must_be_timezone_aware")
    if not obligations:
        return EvidenceAdmissibilityResultV1(
            overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
            requirement_evaluations=(),
            evaluated_at=evaluated_at,
        )
    evaluations: list[RequiredEvidenceEvaluationV1] = []
    for obligation in obligations:
        if isinstance(obligation, IndexedEvidenceRequirementV1):
            evaluations.append(
                _evaluate_indexed_requirement(
                    obligation,
                    indexed_evidence,
                    evaluated_at=evaluated_at,
                )
            )
        else:
            evaluations.append(
                _evaluate_live_requirement(
                    obligation,
                    live_evidence,
                    evaluated_at=evaluated_at,
                )
            )
    overall = (
        EvidenceAdmissibilityStatusV1.SATISFIED
        if all(
            item.status is RequirementEvaluationStatusV1.SATISFIED
            for item in evaluations
        )
        else EvidenceAdmissibilityStatusV1.UNSATISFIED
    )
    return EvidenceAdmissibilityResultV1(
        overall_status=overall,
        requirement_evaluations=tuple(evaluations),
        evaluated_at=evaluated_at,
    )


def evaluate_execution_admissibility(
    *,
    validated_plan: ValidatedEvidencePlanV1,
    execution: KnowledgeQueryExecutionResultV1,
    evaluated_at: datetime,
) -> EvidenceAdmissibilityResultV1:
    return evaluate_evidence_admissibility(
        obligations=validated_plan.plan.required_evidence_obligations,
        indexed_evidence=execution.indexed_evidence,
        live_evidence=execution.live_evidence,
        evaluated_at=evaluated_at,
    )
