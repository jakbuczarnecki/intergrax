# © Artur Czarnecki. All rights reserved.

"""Approval gate providers — explicit outcomes only."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    DecisionEvolutionProposal,
    EvolutionApprovalDecision,
    EvolutionApprovalOutcome,
    EvolutionEvaluationFinding,
    EvolutionExperimentSpec,
)


class HumanReviewApprovalProvider:
    """Default gate: always requires human review — never auto-approves."""

    provider_id = "human_review_approval"
    provider_version = "1"

    def decide(
        self,
        proposal: DecisionEvolutionProposal,
        experiment: EvolutionExperimentSpec,
        evaluations: tuple[EvolutionEvaluationFinding, ...],
    ) -> EvolutionApprovalDecision:
        del evaluations
        return EvolutionApprovalDecision(
            approval_id=f"appr:{proposal.proposal_id}",
            proposal_id=proposal.proposal_id,
            experiment_id=experiment.experiment_id,
            outcome=EvolutionApprovalOutcome.REQUIRES_REVIEW,
            approval_provider_id=self.provider_id,
            approval_provider_version=self.provider_version,
            rationale=(
                "Evolution proposals require explicit human or governance approval; "
                "experiment design does not imply production rollout."
            ),
        )


def default_approval_providers() -> tuple[HumanReviewApprovalProvider, ...]:
    return (HumanReviewApprovalProvider(),)


__all__ = [
    "HumanReviewApprovalProvider",
    "default_approval_providers",
]
