# © Artur Czarnecki. All rights reserved.

"""Pluggable experiment evaluators — one finding per provider."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    DecisionEvolutionProposal,
    EvaluationCriterionKind,
    EvolutionEvaluationFinding,
    EvolutionExperimentSpec,
)


class QualityEvolutionEvaluator:
    evaluator_id = "quality_evaluator"
    evaluator_version = "1"
    criterion_kind = EvaluationCriterionKind.QUALITY

    def evaluate(
        self,
        experiment: EvolutionExperimentSpec,
        *,
        proposal: DecisionEvolutionProposal,
    ) -> EvolutionEvaluationFinding:
        del proposal
        return EvolutionEvaluationFinding(
            evaluation_id=f"{self.evaluator_id}:{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            criterion_kind=self.criterion_kind,
            finding_summary=(
                "Quality criteria defined for both variants; "
                "collect comparative metrics before any change decision."
            ),
            supports_proposal=len(experiment.variants) >= 2,
        )


class SafetyEvolutionEvaluator:
    evaluator_id = "safety_evaluator"
    evaluator_version = "1"
    criterion_kind = EvaluationCriterionKind.SAFETY

    def evaluate(
        self,
        experiment: EvolutionExperimentSpec,
        *,
        proposal: DecisionEvolutionProposal,
    ) -> EvolutionEvaluationFinding:
        return EvolutionEvaluationFinding(
            evaluation_id=f"{self.evaluator_id}:{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            criterion_kind=self.criterion_kind,
            finding_summary=(
                f"Safety review required for {proposal.target_area.value}; "
                "no automatic promotion from experiment results."
            ),
            supports_proposal=True,
        )


class CostEvolutionEvaluator:
    evaluator_id = "cost_evaluator"
    evaluator_version = "1"
    criterion_kind = EvaluationCriterionKind.COST

    def evaluate(
        self,
        experiment: EvolutionExperimentSpec,
        *,
        proposal: DecisionEvolutionProposal,
    ) -> EvolutionEvaluationFinding:
        del proposal
        return EvolutionEvaluationFinding(
            evaluation_id=f"{self.evaluator_id}:{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            criterion_kind=self.criterion_kind,
            finding_summary="Estimate incremental cost of alternate variant in trial scope.",
            supports_proposal=True,
        )


class ReliabilityEvolutionEvaluator:
    evaluator_id = "reliability_evaluator"
    evaluator_version = "1"
    criterion_kind = EvaluationCriterionKind.RELIABILITY

    def evaluate(
        self,
        experiment: EvolutionExperimentSpec,
        *,
        proposal: DecisionEvolutionProposal,
    ) -> EvolutionEvaluationFinding:
        del proposal
        variant_count = len(experiment.variants)
        return EvolutionEvaluationFinding(
            evaluation_id=f"{self.evaluator_id}:{experiment.experiment_id}",
            experiment_id=experiment.experiment_id,
            evaluator_id=self.evaluator_id,
            evaluator_version=self.evaluator_version,
            criterion_kind=self.criterion_kind,
            finding_summary=(
                "Reliability signals should be observed for full experiment window."
            ),
            supports_proposal=variant_count >= 2,
        )


def default_evaluation_providers() -> tuple[
    QualityEvolutionEvaluator,
    SafetyEvolutionEvaluator,
    CostEvolutionEvaluator,
    ReliabilityEvolutionEvaluator,
]:
    return (
        QualityEvolutionEvaluator(),
        SafetyEvolutionEvaluator(),
        CostEvolutionEvaluator(),
        ReliabilityEvolutionEvaluator(),
    )


__all__ = [
    "CostEvolutionEvaluator",
    "QualityEvolutionEvaluator",
    "ReliabilityEvolutionEvaluator",
    "SafetyEvolutionEvaluator",
    "default_evaluation_providers",
]
