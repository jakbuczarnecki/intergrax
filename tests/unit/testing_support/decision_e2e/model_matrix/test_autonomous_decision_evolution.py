# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    AdaptiveDataSourceKind,
    AdaptiveDataSourceRef,
    AdaptiveDecisionContextReference,
    AdaptiveDecisionInsight,
    AdaptiveDecisionRecommendation,
    ConfidenceLevel,
    HistoricalEvidence,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution import (
    AUTONOMOUS_EVOLUTION_TASK_ID,
    AutonomousDecisionEvolutionEngine,
    AutonomousDecisionEvolutionInput,
    EvolutionApprovalOutcome,
    EvolutionRunStatus,
    EvolutionTargetArea,
    ModelChangeProposalGenerator,
    QualityEvolutionEvaluator,
    default_autonomous_decision_evolution_engine,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    DecisionEvolutionProposal,
    EvaluationCriterionKind,
    EvolutionApprovalDecision,
    EvolutionEvaluationFinding,
    EvolutionExperimentSpec,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.experiment_providers import (
    default_experiment_providers,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.proposal_generators import (
    default_proposal_generators,
)


def _stamp() -> datetime:
    return datetime(2026, 9, 12, 15, 0, 0, tzinfo=UTC)


def _model_routing_insight() -> AdaptiveDecisionInsight:
    ref = AdaptiveDecisionContextReference(
        decision_id="dec-evo-1",
        decision_subject="model_routing_choice",
    )
    evidence = HistoricalEvidence(
        evidence_id="ev-1",
        summary="Prior routing required extra human approval.",
        source_refs=(
            AdaptiveDataSourceRef(
                source_kind=AdaptiveDataSourceKind.LIFECYCLE_RECORD,
                reference_id="prior-1",
            ),
        ),
        context_provider_id="lifecycle_history",
        context_provider_version="1",
    )
    recommendation = AdaptiveDecisionRecommendation(
        recommendation_id="rec-1",
        recommendation_provider_id="technical_recommendation",
        recommendation_provider_version="1",
        recommendation_kind="technical",
        narrative=(
            "Consider evaluating an alternate model — suggestion only; "
            "no automatic routing change."
        ),
        linked_reasoning_insight_ids=("ri-1",),
        confidence=ConfidenceLevel.MEDIUM,
        confidence_score=0.6,
    )
    return AdaptiveDecisionInsight(
        insight_id="adi:rec-1",
        decision_context_reference=ref,
        historical_evidence=(evidence,),
        reasoning_summary="Elevated approval friction for current model.",
        recommendation=recommendation,
        confidence=ConfidenceLevel.MEDIUM,
        confidence_score=0.6,
        generated_at=_stamp(),
        context_provider_ids=("lifecycle_history",),
        reasoning_provider_ids=("risk_reasoning",),
        recommendation_provider_id="technical_recommendation",
        data_source_refs=evidence.source_refs,
    )


def test_proposal_generation_from_adaptive_insight() -> None:
    engine = AutonomousDecisionEvolutionEngine(
        proposal_generators=(ModelChangeProposalGenerator(),),
        experiment_providers=(),
        evaluation_providers=(),
        approval_providers=(),
    )
    result = engine.plan_controlled_evolution(
        AutonomousDecisionEvolutionInput(insights=(_model_routing_insight(),)),
        planned_at=_stamp(),
    )

    assert result.status is EvolutionRunStatus.COMPLETE
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.source_insight_ids == ("adi:rec-1",)
    assert proposal.target_area is EvolutionTargetArea.MODEL_SELECTION
    assert "experiment" in proposal.proposed_change.lower()
    assert result.evolution_records[0].source_insight_ids == ("adi:rec-1",)


class _TrackingExperimentProvider:
    provider_id = "tracking_experiment"
    provider_version = "test-1"
    calls: list[str] = []

    def prepare_experiment(
        self,
        proposal: DecisionEvolutionProposal,
    ) -> EvolutionExperimentSpec:
        _TrackingExperimentProvider.calls.append(proposal.proposal_id)
        return EvolutionExperimentSpec(
            experiment_id=f"exp-track:{proposal.proposal_id}",
            proposal_id=proposal.proposal_id,
            experiment_provider_id=self.provider_id,
            experiment_provider_version=self.provider_version,
            variants=(),
            evaluation_criteria=(),
            experiment_summary="tracking experiment",
        )


def test_experiment_plugin_used_by_engine() -> None:
    _TrackingExperimentProvider.calls.clear()
    engine = AutonomousDecisionEvolutionEngine(
        proposal_generators=(ModelChangeProposalGenerator(),),
        experiment_providers=(_TrackingExperimentProvider(),),
        evaluation_providers=(),
        approval_providers=(),
    )
    result = engine.plan_controlled_evolution(
        AutonomousDecisionEvolutionInput(insights=(_model_routing_insight(),)),
        planned_at=_stamp(),
    )

    assert len(_TrackingExperimentProvider.calls) == 1
    assert len(result.experiments) == 1
    assert result.experiments[0].experiment_provider_id == "tracking_experiment"


def test_evaluation_provider_returns_finding_for_experiment() -> None:
    evaluator = QualityEvolutionEvaluator()
    proposal = ModelChangeProposalGenerator().generate(
        (_model_routing_insight(),),
        created_at=_stamp(),
    )[0]
    experiment = EvolutionExperimentSpec(
        experiment_id="exp-eval-1",
        proposal_id=proposal.proposal_id,
        experiment_provider_id="test",
        experiment_provider_version="1",
        variants=(),
        evaluation_criteria=(),
        experiment_summary="eval test",
    )

    finding = evaluator.evaluate(experiment, proposal=proposal)

    assert finding.evaluator_id == "quality_evaluator"
    assert finding.experiment_id == "exp-eval-1"
    assert finding.criterion_kind is EvaluationCriterionKind.QUALITY


class _FixedOutcomeApprovalProvider:
    def __init__(self, outcome: EvolutionApprovalOutcome) -> None:
        self._outcome = outcome
        self.provider_id = f"fixed_approval_{outcome.value}"
        self.provider_version = "test-1"

    def decide(
        self,
        proposal: DecisionEvolutionProposal,
        experiment: EvolutionExperimentSpec,
        evaluations: tuple[EvolutionEvaluationFinding, ...],
    ) -> EvolutionApprovalDecision:
        del evaluations
        return EvolutionApprovalDecision(
            approval_id=f"appr:{proposal.proposal_id}:{self._outcome.value}",
            proposal_id=proposal.proposal_id,
            experiment_id=experiment.experiment_id,
            outcome=self._outcome,
            approval_provider_id=self.provider_id,
            approval_provider_version=self.provider_version,
            rationale=f"test outcome {self._outcome.value}",
        )


def test_approval_gate_outcomes() -> None:
    insight = (_model_routing_insight(),)
    for outcome in (
        EvolutionApprovalOutcome.APPROVED,
        EvolutionApprovalOutcome.REJECTED,
        EvolutionApprovalOutcome.REQUIRES_REVIEW,
    ):
        engine = AutonomousDecisionEvolutionEngine(
            proposal_generators=default_proposal_generators(),
            experiment_providers=default_experiment_providers(),
            evaluation_providers=(),
            approval_providers=(_FixedOutcomeApprovalProvider(outcome),),
        )
        result = engine.plan_controlled_evolution(
            AutonomousDecisionEvolutionInput(insights=insight),
            planned_at=_stamp(),
        )
        assert result.approval_decisions
        assert result.approval_decisions[0].outcome is outcome


def test_missing_insight_returns_controlled_result_without_error() -> None:
    engine = default_autonomous_decision_evolution_engine()
    result = engine.plan_controlled_evolution(
        AutonomousDecisionEvolutionInput(),
        planned_at=_stamp(),
    )

    assert result.evolution_task_id == AUTONOMOUS_EVOLUTION_TASK_ID
    assert result.status is EvolutionRunStatus.INSUFFICIENT_INPUT
    assert result.proposals == ()
    assert result.experiments == ()
    assert result.evaluation_findings == ()
    assert result.approval_decisions == ()
