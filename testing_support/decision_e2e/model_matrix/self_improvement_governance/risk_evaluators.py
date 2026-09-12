# © Artur Czarnecki. All rights reserved.

"""Evolution risk assessment plugins (DS-E2E-15J-L12)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    EvolutionRiskFinding,
    EvolutionRiskImpactLevel,
    SelfImprovementGovernanceRequest,
)

_PROPOSAL_RISK_EVALUATOR_ID = "proposal_risk_information"
_PROPOSAL_RISK_EVALUATOR_VERSION = "1"
_HIGH_IMPACT_MARKER = "high impact"


@dataclass(frozen=True, slots=True)
class ProposalRiskInformationEvaluator:
    """Derives structured risk findings from proposal metadata."""

    @property
    def evaluator_id(self) -> str:
        return _PROPOSAL_RISK_EVALUATOR_ID

    @property
    def evaluator_version(self) -> str:
        return _PROPOSAL_RISK_EVALUATOR_VERSION

    def assess(
        self, request: SelfImprovementGovernanceRequest
    ) -> tuple[EvolutionRiskFinding, ...]:
        proposal = request.evolution_proposal
        summary_lower = proposal.risk_information.risk_summary.casefold()
        if _HIGH_IMPACT_MARKER in summary_lower:
            return (
                EvolutionRiskFinding(
                    finding_id=f"risk:{proposal.proposal_id}:high",
                    evaluator_id=self.evaluator_id,
                    evaluator_version=self.evaluator_version,
                    impact_level=EvolutionRiskImpactLevel.HIGH,
                    summary=proposal.risk_information.risk_summary,
                    consequence_description=(
                        "Evolution may materially change production decision behavior."
                    ),
                    required_controls=("human_approval", "extended_monitoring"),
                ),
            )
        return (
            EvolutionRiskFinding(
                finding_id=f"risk:{proposal.proposal_id}:baseline",
                evaluator_id=self.evaluator_id,
                evaluator_version=self.evaluator_version,
                impact_level=EvolutionRiskImpactLevel.LOW,
                summary=proposal.risk_information.risk_summary,
                consequence_description=proposal.risk_information.mitigation_notes,
                required_controls=("standard_governance_review",),
            ),
        )


def default_risk_evaluators() -> tuple[ProposalRiskInformationEvaluator, ...]:
    return (ProposalRiskInformationEvaluator(),)


__all__ = [
    "ProposalRiskInformationEvaluator",
    "default_risk_evaluators",
]
