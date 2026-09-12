# © Artur Czarnecki. All rights reserved.

"""Self-improvement approval provider plugins (DS-E2E-15J-L12)."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceRequest,
    SelfImprovementGovernanceStatus,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.protocol import (
    SelfImprovementApprovalRecord,
    SelfImprovementPolicyResult,
)

_HUMAN_APPROVAL_PROVIDER_ID = "human_self_improvement_approval"
_HUMAN_APPROVAL_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class HumanSelfImprovementApprovalProvider:
    """Default gate: policy-approved evolutions still require explicit human sign-off."""

    @property
    def provider_id(self) -> str:
        return _HUMAN_APPROVAL_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _HUMAN_APPROVAL_PROVIDER_VERSION

    def decide(
        self,
        request: SelfImprovementGovernanceRequest,
        *,
        policy_results: tuple[SelfImprovementPolicyResult, ...],
        aggregated_status: SelfImprovementGovernanceStatus,
    ) -> SelfImprovementApprovalRecord:
        return SelfImprovementApprovalRecord(
            approval_id=f"appr:{uuid4()}",
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            outcome_status=SelfImprovementGovernanceStatus.REQUIRES_REVIEW,
            rationale=(
                f"Policies aggregated to {aggregated_status.value}; "
                f"human approval required for proposal {request.evolution_proposal.proposal_id}."
            ),
        )


@dataclass(frozen=True, slots=True)
class RecordedSelfImprovementApprovalProvider:
    """Test / enterprise stub that records an explicit governance outcome."""

    outcome_status: SelfImprovementGovernanceStatus
    provider_id: str = "recorded_approval"
    provider_version: str = "1"

    def decide(
        self,
        request: SelfImprovementGovernanceRequest,
        *,
        policy_results: tuple[SelfImprovementPolicyResult, ...],
        aggregated_status: SelfImprovementGovernanceStatus,
    ) -> SelfImprovementApprovalRecord:
        return SelfImprovementApprovalRecord(
            approval_id=f"appr:{request.evolution_proposal.proposal_id}",
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            outcome_status=self.outcome_status,
            rationale=f"Recorded approval outcome for aggregated {aggregated_status.value}.",
        )


__all__ = [
    "HumanSelfImprovementApprovalProvider",
    "RecordedSelfImprovementApprovalProvider",
]
