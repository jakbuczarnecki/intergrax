# © Artur Czarnecki. All rights reserved.

"""Built-in evolution proposal generators — proposals only, no execution."""

from __future__ import annotations

from datetime import datetime

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    AdaptiveDecisionInsight,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    DecisionEvolutionProposal,
    EvolutionRiskInformation,
    EvolutionTargetArea,
)


class ModelChangeProposalGenerator:
    generator_id = "model_change_proposal"
    generator_version = "1"
    target_area = EvolutionTargetArea.MODEL_SELECTION

    def generate(
        self,
        insights: tuple[AdaptiveDecisionInsight, ...],
        *,
        created_at: datetime,
    ) -> tuple[DecisionEvolutionProposal, ...]:
        proposals: list[DecisionEvolutionProposal] = []
        for insight in insights:
            narrative = insight.recommendation.narrative.lower()
            if (
                "model" not in narrative
                and "routing" not in insight.decision_context_reference.decision_subject
            ):
                continue
            proposals.append(
                DecisionEvolutionProposal(
                    proposal_id=f"{self.generator_id}:{insight.insight_id}",
                    generator_id=self.generator_id,
                    generator_version=self.generator_version,
                    source_insight_ids=(insight.insight_id,),
                    target_area=self.target_area,
                    proposed_change=(
                        "Evaluate an alternate model in a controlled experiment "
                        "(variant A: current model, variant B: candidate model)."
                    ),
                    expected_benefit=(
                        "Reduced human approval friction if the candidate model "
                        "performs comparably on quality and safety criteria."
                    ),
                    risk_information=EvolutionRiskInformation(
                        risk_summary=(
                            "Candidate model may underperform on edge cases "
                            "observed in historical evidence."
                        ),
                        mitigation_notes=(
                            "Run experiment in shadow or limited scope; "
                            "require explicit approval before any production use."
                        ),
                    ),
                    created_at=created_at,
                )
            )
        return tuple(proposals)


class PolicyChangeProposalGenerator:
    generator_id = "policy_change_proposal"
    generator_version = "1"
    target_area = EvolutionTargetArea.POLICY

    def generate(
        self,
        insights: tuple[AdaptiveDecisionInsight, ...],
        *,
        created_at: datetime,
    ) -> tuple[DecisionEvolutionProposal, ...]:
        proposals: list[DecisionEvolutionProposal] = []
        for insight in insights:
            if insight.recommendation.recommendation_kind != "business":
                continue
            proposals.append(
                DecisionEvolutionProposal(
                    proposal_id=f"{self.generator_id}:{insight.insight_id}",
                    generator_id=self.generator_id,
                    generator_version=self.generator_version,
                    source_insight_ids=(insight.insight_id,),
                    target_area=self.target_area,
                    proposed_change=(
                        "Draft a policy adjustment proposal for governance review "
                        "(documentation only — no automatic policy rewrite)."
                    ),
                    expected_benefit="Clearer governance alignment with observed friction patterns.",
                    risk_information=EvolutionRiskInformation(
                        risk_summary="Policy drift if approved without full stakeholder review.",
                        mitigation_notes="Keep experiment read-only; approval gate mandatory.",
                    ),
                    created_at=created_at,
                )
            )
        return tuple(proposals)


class ProcessChangeProposalGenerator:
    generator_id = "process_change_proposal"
    generator_version = "1"
    target_area = EvolutionTargetArea.PROCESS

    def generate(
        self,
        insights: tuple[AdaptiveDecisionInsight, ...],
        *,
        created_at: datetime,
    ) -> tuple[DecisionEvolutionProposal, ...]:
        if not insights:
            return ()
        insight = insights[0]
        return (
            DecisionEvolutionProposal(
                proposal_id=f"{self.generator_id}:{insight.insight_id}",
                generator_id=self.generator_id,
                generator_version=self.generator_version,
                source_insight_ids=tuple(item.insight_id for item in insights),
                target_area=self.target_area,
                proposed_change=(
                    "Propose an additional human review checkpoint in the decision "
                    "orchestration playbook (simulation only)."
                ),
                expected_benefit="Fewer failed lifecycle outcomes before production routing.",
                risk_information=EvolutionRiskInformation(
                    risk_summary="Added latency in decision throughput.",
                    mitigation_notes="Measure cost impact in experiment criteria before approval.",
                ),
                created_at=created_at,
            ),
        )


def default_proposal_generators() -> tuple[
    ModelChangeProposalGenerator,
    PolicyChangeProposalGenerator,
    ProcessChangeProposalGenerator,
]:
    return (
        ModelChangeProposalGenerator(),
        PolicyChangeProposalGenerator(),
        ProcessChangeProposalGenerator(),
    )


__all__ = [
    "ModelChangeProposalGenerator",
    "PolicyChangeProposalGenerator",
    "ProcessChangeProposalGenerator",
    "default_proposal_generators",
]
