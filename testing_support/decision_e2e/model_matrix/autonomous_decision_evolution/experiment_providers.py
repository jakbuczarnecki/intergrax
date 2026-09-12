# © Artur Czarnecki. All rights reserved.

"""Built-in experiment design providers — no deployment."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    DecisionEvolutionProposal,
    EvaluationCriterion,
    EvaluationCriterionKind,
    EvolutionExperimentSpec,
    EvolutionTargetArea,
    ExperimentVariant,
)


class StandardControlledExperimentProvider:
    provider_id = "standard_controlled_experiment"
    provider_version = "1"

    def prepare_experiment(
        self,
        proposal: DecisionEvolutionProposal,
    ) -> EvolutionExperimentSpec | None:
        variants = _variants_for_area(proposal.target_area)
        if not variants:
            return None
        return EvolutionExperimentSpec(
            experiment_id=f"exp:{proposal.proposal_id}",
            proposal_id=proposal.proposal_id,
            experiment_provider_id=self.provider_id,
            experiment_provider_version=self.provider_version,
            variants=variants,
            evaluation_criteria=_default_criteria(),
            experiment_summary=(
                f"Controlled comparison for proposal {proposal.proposal_id} "
                f"in area {proposal.target_area.value}."
            ),
        )


def _variants_for_area(
    area: EvolutionTargetArea,
) -> tuple[ExperimentVariant, ...]:
    if area is EvolutionTargetArea.MODEL_SELECTION:
        return (
            ExperimentVariant(
                variant_id="variant_a",
                variant_label="Current production model",
                variant_description="Baseline behavior under existing routing.",
            ),
            ExperimentVariant(
                variant_id="variant_b",
                variant_label="Candidate model",
                variant_description="Alternate model evaluated in isolated scope only.",
            ),
        )
    if area is EvolutionTargetArea.POLICY:
        return (
            ExperimentVariant(
                variant_id="variant_a",
                variant_label="Current policy interpretation",
                variant_description="Documented governance baseline.",
            ),
            ExperimentVariant(
                variant_id="variant_b",
                variant_label="Proposed policy interpretation",
                variant_description="Paper exercise for governance review.",
            ),
        )
    return (
        ExperimentVariant(
            variant_id="variant_a",
            variant_label="Current process",
            variant_description="Existing orchestration steps.",
        ),
        ExperimentVariant(
            variant_id="variant_b",
            variant_label="Process with added review",
            variant_description="Simulated additional checkpoint.",
        ),
    )


def _default_criteria() -> tuple[EvaluationCriterion, ...]:
    return tuple(
        EvaluationCriterion(
            criterion_id=f"crit:{kind.value}",
            criterion_kind=kind,
            description=f"Assess experiment along {kind.value} dimension.",
        )
        for kind in EvaluationCriterionKind
    )


def default_experiment_providers() -> tuple[StandardControlledExperimentProvider, ...]:
    return (StandardControlledExperimentProvider(),)


__all__ = [
    "StandardControlledExperimentProvider",
    "default_experiment_providers",
]
