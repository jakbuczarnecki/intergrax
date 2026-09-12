# © Artur Czarnecki. All rights reserved.

"""Governance evaluation orchestration (DS-E2E-15J-L5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GOVERNANCE_TASK_ID,
    GOVERNANCE_VERSION,
    GovernanceDataSourceKind,
    GovernanceDataSourceRef,
    GovernanceDecision,
    GovernanceDecisionAuditMetadata,
    GovernanceDisposition,
    GovernanceEvaluationRequest,
    GovernancePolicyRef,
    GovernanceReasonCode,
    GovernanceReasonRef,
    PolicyParticipationRecord,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.protocol import (
    PolicyEvaluationResult,
    PolicyEvaluator,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.contracts import (
    ModelSelectionStatus,
)


def _aggregate_disposition(
    results: tuple[PolicyEvaluationResult, ...],
) -> GovernanceDisposition:
    if any(item.contribution is GovernanceDisposition.BLOCK for item in results):
        return GovernanceDisposition.BLOCK
    if any(
        item.contribution is GovernanceDisposition.REQUIRE_APPROVAL for item in results
    ):
        return GovernanceDisposition.REQUIRE_APPROVAL
    return GovernanceDisposition.ALLOW


def _policy_refs_from_results(
    results: tuple[PolicyEvaluationResult, ...],
) -> tuple[GovernancePolicyRef, ...]:
    return tuple(
        GovernancePolicyRef(
            policy_id=item.policy_id,
            policy_version=item.policy_version,
        )
        for item in results
    )


def _reasons_from_results(
    results: tuple[PolicyEvaluationResult, ...],
) -> tuple[GovernanceReasonRef, ...]:
    merged: list[GovernanceReasonRef] = []
    for item in results:
        merged.extend(item.reasons)
    return tuple(merged)


def _participation_summary(result: PolicyEvaluationResult) -> str:
    return f"{result.policy_id}: {result.contribution}"


def _select_evaluators(
    evaluators: tuple[PolicyEvaluator, ...],
    applicable: tuple[GovernancePolicyRef, ...],
) -> tuple[PolicyEvaluator, ...]:
    if not applicable:
        return evaluators
    allowed_ids = {item.policy_id for item in applicable}
    return tuple(item for item in evaluators if item.policy_id in allowed_ids)


def _insufficient_selection_decision(
    request: GovernanceEvaluationRequest,
    *,
    stamp: datetime,
    decision_id: str,
    reason_code: GovernanceReasonCode,
    summary: str,
) -> GovernanceDecision:
    capability_keys = tuple(
        profile.model_identity.profile_key for profile in request.capability_evidence
    )
    data_sources: list[GovernanceDataSourceRef] = []
    selection_refs = ()
    if request.model_recommendation is not None:
        selection_refs = request.model_recommendation.evidence_references
        data_sources.append(
            GovernanceDataSourceRef(
                source_kind=GovernanceDataSourceKind.MODEL_SELECTION_RECOMMENDATION,
                reference_id=request.model_recommendation.decision_metadata.selection_task_id,
            )
        )
    for key in capability_keys:
        data_sources.append(
            GovernanceDataSourceRef(
                source_kind=GovernanceDataSourceKind.MODEL_CAPABILITY_PROFILE,
                reference_id=key,
            )
        )
    reason = GovernanceReasonRef(
        reason_code=reason_code,
        summary=summary,
        policy_id=GOVERNANCE_TASK_ID,
    )
    return GovernanceDecision(
        disposition=GovernanceDisposition.BLOCK,
        reason_references=(reason,),
        policy_references=(),
        audit_metadata=GovernanceDecisionAuditMetadata(
            governance_task_id=GOVERNANCE_TASK_ID,
            governance_version=GOVERNANCE_VERSION,
            decision_id=decision_id,
            evaluated_at=stamp,
            scenario_id=request.task_context.scenario_id,
            recommended_profile_key=None,
            applied_policy_refs=(),
            policy_participation=(),
            data_source_refs=tuple(data_sources),
            selection_evidence_refs=selection_refs,
            capability_profile_keys=capability_keys,
        ),
    )


@dataclass(frozen=True, slots=True)
class GovernanceEvaluationEngine:
    """Evaluates whether a model selection recommendation may proceed; does not execute."""

    evaluators: tuple[PolicyEvaluator, ...]

    def evaluate(
        self,
        request: GovernanceEvaluationRequest,
        *,
        evaluated_at: datetime | None = None,
    ) -> GovernanceDecision:
        stamp = evaluated_at or datetime.now(tz=UTC)
        decision_id = uuid4().hex

        recommendation = request.model_recommendation
        if recommendation is None:
            return _insufficient_selection_decision(
                request,
                stamp=stamp,
                decision_id=decision_id,
                reason_code=GovernanceReasonCode.NO_MODEL_RECOMMENDATION,
                summary="model selection recommendation was not provided",
            )
        if recommendation.status is not ModelSelectionStatus.RECOMMENDED:
            return _insufficient_selection_decision(
                request,
                stamp=stamp,
                decision_id=decision_id,
                reason_code=GovernanceReasonCode.SELECTION_NOT_RECOMMENDED,
                summary=f"selection status is {recommendation.status}, not recommended",
            )
        if recommendation.selected_model_reference is None:
            return _insufficient_selection_decision(
                request,
                stamp=stamp,
                decision_id=decision_id,
                reason_code=GovernanceReasonCode.SELECTION_NOT_RECOMMENDED,
                summary="selection recommendation has no selected model reference",
            )

        selected_key = recommendation.selected_model_reference.profile_key
        active_evaluators = _select_evaluators(
            self.evaluators,
            request.applicable_policies,
        )
        if request.applicable_policies and not active_evaluators:
            return _insufficient_selection_decision(
                request,
                stamp=stamp,
                decision_id=decision_id,
                reason_code=GovernanceReasonCode.NO_APPLICABLE_POLICY_EVALUATOR,
                summary="no injected evaluator matches applicable policy references",
            )

        policy_results: tuple[PolicyEvaluationResult, ...] = tuple(
            evaluator.evaluate(request) for evaluator in active_evaluators
        )
        disposition = _aggregate_disposition(policy_results)
        participation = tuple(
            PolicyParticipationRecord(
                policy_id=item.policy_id,
                policy_version=item.policy_version,
                contribution=item.contribution,
                outcome_summary=_participation_summary(item),
            )
            for item in policy_results
        )
        capability_keys = tuple(
            profile.model_identity.profile_key
            for profile in request.capability_evidence
        )
        data_sources: list[GovernanceDataSourceRef] = [
            GovernanceDataSourceRef(
                source_kind=GovernanceDataSourceKind.MODEL_SELECTION_RECOMMENDATION,
                reference_id=recommendation.decision_metadata.selection_task_id,
            )
        ]
        for key in capability_keys:
            data_sources.append(
                GovernanceDataSourceRef(
                    source_kind=GovernanceDataSourceKind.MODEL_CAPABILITY_PROFILE,
                    reference_id=key,
                )
            )
        for evidence_ref in recommendation.evidence_references:
            data_sources.append(
                GovernanceDataSourceRef(
                    source_kind=GovernanceDataSourceKind.SELECTION_EVIDENCE_REFERENCE,
                    reference_id=evidence_ref.profile_key,
                )
            )

        return GovernanceDecision(
            disposition=disposition,
            reason_references=_reasons_from_results(policy_results),
            policy_references=_policy_refs_from_results(policy_results),
            audit_metadata=GovernanceDecisionAuditMetadata(
                governance_task_id=GOVERNANCE_TASK_ID,
                governance_version=GOVERNANCE_VERSION,
                decision_id=decision_id,
                evaluated_at=stamp,
                scenario_id=request.task_context.scenario_id,
                recommended_profile_key=selected_key,
                applied_policy_refs=_policy_refs_from_results(policy_results),
                policy_participation=participation,
                data_source_refs=tuple(data_sources),
                selection_evidence_refs=recommendation.evidence_references,
                capability_profile_keys=capability_keys,
            ),
        )


__all__ = ["GovernanceEvaluationEngine"]
