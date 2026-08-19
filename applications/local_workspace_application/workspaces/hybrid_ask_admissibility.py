# © Artur Czarnecki. All rights reserved.

"""Deterministic Evidence Admissibility evaluation for Workspace Ask V2."""

from __future__ import annotations

from local_workspace_application.workspaces.hybrid_ask_execution import (
    KnowledgeQueryExecutionResultV1,
)
from local_workspace_application.workspaces.hybrid_ask_models import (
    EvidenceAdmissibilityResultV1,
    EvidenceAdmissibilityStatusV1,
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


def _evaluate_indexed_requirement(
    requirement: IndexedEvidenceRequirementV1,
    indexed_evidence: tuple[IndexedWorkspaceEvidenceV1, ...],
) -> RequiredEvidenceEvaluationV1:
    matched_ids: list[str] = []
    for item in indexed_evidence:
        if requirement.indexed_source_binding_id is not None:
            if item.indexed_source_binding_id != requirement.indexed_source_binding_id:
                continue
        matched_ids.append(item.evidence_id)
    if matched_ids:
        return RequiredEvidenceEvaluationV1(
            requirement_id=requirement.requirement_id,
            status=RequirementEvaluationStatusV1.SATISFIED,
            matched_evidence_ids=tuple(matched_ids),
        )
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


def _evaluate_live_requirement(
    requirement: LiveEvidenceRequirementV1,
    live_evidence: tuple[LiveWorkspaceEvidenceV1, ...],
) -> RequiredEvidenceEvaluationV1:
    matched_ids = tuple(
        item.evidence_id
        for item in live_evidence
        if item.call_id == requirement.call_id
    )
    if matched_ids:
        return RequiredEvidenceEvaluationV1(
            requirement_id=requirement.requirement_id,
            status=RequirementEvaluationStatusV1.SATISFIED,
            matched_evidence_ids=matched_ids,
        )
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


def evaluate_evidence_admissibility(
    *,
    obligations: tuple[RequiredEvidenceObligationV1, ...],
    indexed_evidence: tuple[IndexedWorkspaceEvidenceV1, ...],
    live_evidence: tuple[LiveWorkspaceEvidenceV1, ...],
) -> EvidenceAdmissibilityResultV1:
    if not obligations:
        return EvidenceAdmissibilityResultV1(
            overall_status=EvidenceAdmissibilityStatusV1.SATISFIED,
            requirement_evaluations=(),
        )
    evaluations: list[RequiredEvidenceEvaluationV1] = []
    for obligation in obligations:
        if isinstance(obligation, IndexedEvidenceRequirementV1):
            evaluations.append(
                _evaluate_indexed_requirement(obligation, indexed_evidence)
            )
        else:
            evaluations.append(_evaluate_live_requirement(obligation, live_evidence))
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
    )


def evaluate_execution_admissibility(
    *,
    validated_plan: ValidatedEvidencePlanV1,
    execution: KnowledgeQueryExecutionResultV1,
) -> EvidenceAdmissibilityResultV1:
    return evaluate_evidence_admissibility(
        obligations=validated_plan.plan.required_evidence_obligations,
        indexed_evidence=execution.indexed_evidence,
        live_evidence=execution.live_evidence,
    )
