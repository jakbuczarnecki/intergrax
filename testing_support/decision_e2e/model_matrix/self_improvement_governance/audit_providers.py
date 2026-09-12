# © Artur Czarnecki. All rights reserved.

"""Governance audit metadata providers (DS-E2E-15J-L12)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SELF_IMPROVEMENT_GOVERNANCE_TASK_ID,
    SELF_IMPROVEMENT_GOVERNANCE_VERSION,
    EvolutionRiskFinding,
    SelfImprovementGovernanceAuditMetadata,
    SelfImprovementGovernanceRequest,
    SelfImprovementGovernanceStatus,
    SelfImprovementPolicyRef,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.protocol import (
    SelfImprovementApprovalRecord,
    SelfImprovementPolicyResult,
)

_STANDARD_AUDIT_PROVIDER_ID = "standard_self_improvement_audit"
_STANDARD_AUDIT_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class StandardGovernanceAuditProvider:
    @property
    def provider_id(self) -> str:
        return _STANDARD_AUDIT_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _STANDARD_AUDIT_PROVIDER_VERSION

    def build_audit(
        self,
        request: SelfImprovementGovernanceRequest,
        *,
        risk_findings: tuple[EvolutionRiskFinding, ...],
        policy_results: tuple[SelfImprovementPolicyResult, ...],
        approval_record: SelfImprovementApprovalRecord | None,
        final_status: SelfImprovementGovernanceStatus,
        evaluated_at: datetime,
        policy_evaluator_ids: tuple[str, ...],
        policy_evaluator_versions: tuple[str, ...],
        risk_evaluator_ids: tuple[str, ...],
        risk_evaluator_versions: tuple[str, ...],
    ) -> SelfImprovementGovernanceAuditMetadata:
        policy_refs = tuple(
            SelfImprovementPolicyRef(
                policy_id=item.policy_id,
                policy_version=item.policy_version,
            )
            for item in policy_results
        )
        evaluation_refs = tuple(
            item.evaluation_id for item in request.evaluation_findings
        )
        return SelfImprovementGovernanceAuditMetadata(
            governance_task_id=SELF_IMPROVEMENT_GOVERNANCE_TASK_ID,
            governance_version=SELF_IMPROVEMENT_GOVERNANCE_VERSION,
            proposal_reference=request.evolution_proposal.proposal_id,
            experiment_reference=request.experiment_result.experiment_id,
            evaluation_references=evaluation_refs,
            policy_references=policy_refs,
            risk_evaluator_ids=risk_evaluator_ids,
            risk_evaluator_versions=risk_evaluator_versions,
            policy_evaluator_ids=policy_evaluator_ids,
            policy_evaluator_versions=policy_evaluator_versions,
            approval_provider_id=(
                approval_record.provider_id if approval_record is not None else None
            ),
            approval_provider_version=(
                approval_record.provider_version
                if approval_record is not None
                else None
            ),
            evaluated_at=evaluated_at,
        )


def default_audit_provider() -> StandardGovernanceAuditProvider:
    return StandardGovernanceAuditProvider()


__all__ = [
    "StandardGovernanceAuditProvider",
    "default_audit_provider",
]
