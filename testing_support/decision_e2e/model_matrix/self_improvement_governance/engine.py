# © Artur Czarnecki. All rights reserved.

"""Self-improvement governance orchestration via injected plugins (DS-E2E-15J-L12)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.self_improvement_governance.audit_providers import (
    default_audit_provider,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    EvolutionRiskContext,
    EvolutionRiskFinding,
    SelfImprovementGovernanceDecision,
    SelfImprovementGovernanceReason,
    SelfImprovementGovernanceRequest,
    SelfImprovementGovernanceStatus,
    SelfImprovementPolicyRef,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.policy_evaluators import (
    default_policy_evaluators,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.protocol import (
    EvolutionRiskEvaluator,
    SelfImprovementApprovalProvider,
    SelfImprovementApprovalRecord,
    SelfImprovementGovernanceAuditProvider,
    SelfImprovementPolicyEvaluator,
    SelfImprovementPolicyResult,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.risk_evaluators import (
    default_risk_evaluators,
)


_STATUS_SEVERITY: dict[SelfImprovementGovernanceStatus, int] = {
    SelfImprovementGovernanceStatus.APPROVED: 0,
    SelfImprovementGovernanceStatus.REQUIRES_REVIEW: 1,
    SelfImprovementGovernanceStatus.REQUIRES_MORE_EVIDENCE: 2,
    SelfImprovementGovernanceStatus.REJECTED: 3,
}


def _aggregate_policy_status(
    results: tuple[SelfImprovementPolicyResult, ...],
) -> SelfImprovementGovernanceStatus:
    if not results:
        return SelfImprovementGovernanceStatus.REQUIRES_REVIEW
    worst = max(
        results,
        key=lambda item: _STATUS_SEVERITY[item.status_contribution],
    )
    return worst.status_contribution


def _merge_reasons(
    results: tuple[SelfImprovementPolicyResult, ...],
) -> tuple[SelfImprovementGovernanceReason, ...]:
    merged: list[SelfImprovementGovernanceReason] = []
    for item in results:
        merged.extend(item.reasons)
    return tuple(merged)


def _merge_required_actions(
    results: tuple[SelfImprovementPolicyResult, ...],
) -> tuple[str, ...]:
    actions: list[str] = []
    for item in results:
        actions.extend(item.required_actions)
    return tuple(actions)


def _policy_refs(
    results: tuple[SelfImprovementPolicyResult, ...],
) -> tuple[SelfImprovementPolicyRef, ...]:
    return tuple(
        SelfImprovementPolicyRef(
            policy_id=item.policy_id,
            policy_version=item.policy_version,
        )
        for item in results
    )


def _collect_risk_findings(
    evaluators: tuple[EvolutionRiskEvaluator, ...],
    request: SelfImprovementGovernanceRequest,
) -> tuple[EvolutionRiskFinding, ...]:
    findings: list[EvolutionRiskFinding] = []
    for evaluator in evaluators:
        findings.extend(evaluator.assess(request))
    return tuple(findings)


def _with_risk_context(
    request: SelfImprovementGovernanceRequest,
    findings: tuple[EvolutionRiskFinding, ...],
) -> SelfImprovementGovernanceRequest:
    existing = request.risk_context.findings
    combined = existing + findings
    if combined == existing:
        return request
    return SelfImprovementGovernanceRequest(
        evolution_proposal=request.evolution_proposal,
        experiment_result=request.experiment_result,
        evaluation_findings=request.evaluation_findings,
        risk_context=EvolutionRiskContext(
            proposal_id=request.evolution_proposal.proposal_id,
            findings=combined,
        ),
        lifecycle_records=request.lifecycle_records,
    )


@dataclass(frozen=True, slots=True)
class SelfImprovementGovernanceEngine:
    policy_evaluators: tuple[SelfImprovementPolicyEvaluator, ...]
    risk_evaluators: tuple[EvolutionRiskEvaluator, ...]
    approval_provider: SelfImprovementApprovalProvider
    audit_provider: SelfImprovementGovernanceAuditProvider

    def evaluate_governance(
        self,
        request: SelfImprovementGovernanceRequest,
        *,
        evaluated_at: datetime | None = None,
    ) -> SelfImprovementGovernanceDecision:
        stamp = evaluated_at or datetime.now(tz=UTC)
        risk_evaluator_ids = tuple(item.evaluator_id for item in self.risk_evaluators)
        risk_evaluator_versions = tuple(
            item.evaluator_version for item in self.risk_evaluators
        )
        policy_evaluator_ids = tuple(item.policy_id for item in self.policy_evaluators)
        policy_evaluator_versions = tuple(
            item.policy_version for item in self.policy_evaluators
        )

        risk_findings = _collect_risk_findings(self.risk_evaluators, request)
        enriched_request = _with_risk_context(request, risk_findings)

        policy_results = tuple(
            evaluator.evaluate(enriched_request) for evaluator in self.policy_evaluators
        )
        aggregated = _aggregate_policy_status(policy_results)

        approval_record: SelfImprovementApprovalRecord | None = None
        final_status = aggregated
        if aggregated is SelfImprovementGovernanceStatus.APPROVED:
            approval_record = self.approval_provider.decide(
                enriched_request,
                policy_results=policy_results,
                aggregated_status=aggregated,
            )
            final_status = approval_record.outcome_status

        audit = self.audit_provider.build_audit(
            enriched_request,
            risk_findings=risk_findings,
            policy_results=policy_results,
            approval_record=approval_record,
            final_status=final_status,
            evaluated_at=stamp,
            policy_evaluator_ids=policy_evaluator_ids,
            policy_evaluator_versions=policy_evaluator_versions,
            risk_evaluator_ids=risk_evaluator_ids,
            risk_evaluator_versions=risk_evaluator_versions,
        )

        return SelfImprovementGovernanceDecision(
            status=final_status,
            reasons=_merge_reasons(policy_results),
            required_actions=_merge_required_actions(policy_results),
            policy_references=_policy_refs(policy_results),
            audit_metadata=audit,
        )


def default_self_improvement_governance_engine(
    *,
    policy_evaluators: tuple[SelfImprovementPolicyEvaluator, ...] | None = None,
    risk_evaluators: tuple[EvolutionRiskEvaluator, ...] | None = None,
    approval_provider: SelfImprovementApprovalProvider | None = None,
    audit_provider: SelfImprovementGovernanceAuditProvider | None = None,
) -> SelfImprovementGovernanceEngine:
    return SelfImprovementGovernanceEngine(
        policy_evaluators=policy_evaluators or default_policy_evaluators(),
        risk_evaluators=risk_evaluators or default_risk_evaluators(),
        approval_provider=approval_provider or _default_approval_placeholder(),
        audit_provider=audit_provider or default_audit_provider(),
    )


def _default_approval_placeholder() -> SelfImprovementApprovalProvider:
    from testing_support.decision_e2e.model_matrix.self_improvement_governance.approval_providers import (
        HumanSelfImprovementApprovalProvider,
    )

    return HumanSelfImprovementApprovalProvider()


__all__ = [
    "SelfImprovementGovernanceEngine",
    "default_self_improvement_governance_engine",
]
