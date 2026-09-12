# © Artur Czarnecki. All rights reserved.

"""Autonomous decision evolution orchestration via injected plugins (DS-E2E-15J-L11)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.approval_providers import (
    default_approval_providers,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    AUTONOMOUS_EVOLUTION_TASK_ID,
    AUTONOMOUS_EVOLUTION_VERSION,
    AutonomousDecisionEvolutionAuditMetadata,
    AutonomousDecisionEvolutionInput,
    AutonomousDecisionEvolutionResult,
    ControlledEvolutionRecord,
    DecisionEvolutionProposal,
    EvolutionApprovalDecision,
    EvolutionEvaluationFinding,
    EvolutionExperimentSpec,
    EvolutionRunStatus,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.evaluation_providers import (
    default_evaluation_providers,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.experiment_providers import (
    default_experiment_providers,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.proposal_generators import (
    default_proposal_generators,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.protocol import (
    EvolutionApprovalProvider,
    EvolutionEvaluationProvider,
    EvolutionExperimentProvider,
    EvolutionProposalGenerator,
)


def _empty_audit(
    *,
    planned_at: datetime,
    source_insight_ids: tuple[str, ...],
    proposal_generator_ids: tuple[str, ...],
    experiment_provider_ids: tuple[str, ...],
    evaluation_provider_ids: tuple[str, ...],
    approval_provider_ids: tuple[str, ...],
) -> AutonomousDecisionEvolutionAuditMetadata:
    return AutonomousDecisionEvolutionAuditMetadata(
        engine_task_id=AUTONOMOUS_EVOLUTION_TASK_ID,
        engine_version=AUTONOMOUS_EVOLUTION_VERSION,
        proposal_generator_ids=proposal_generator_ids,
        experiment_provider_ids=experiment_provider_ids,
        evaluation_provider_ids=evaluation_provider_ids,
        approval_provider_ids=approval_provider_ids,
        source_insight_ids=source_insight_ids,
        planned_at=planned_at,
    )


def _generate_proposals(
    generators: tuple[EvolutionProposalGenerator, ...],
    input_data: AutonomousDecisionEvolutionInput,
    created_at: datetime,
) -> tuple[DecisionEvolutionProposal, ...]:
    proposals: list[DecisionEvolutionProposal] = []
    for generator in generators:
        proposals.extend(generator.generate(input_data.insights, created_at=created_at))
    return tuple(proposals)


def _prepare_experiment(
    providers: tuple[EvolutionExperimentProvider, ...],
    proposal: DecisionEvolutionProposal,
) -> EvolutionExperimentSpec | None:
    for provider in providers:
        spec = provider.prepare_experiment(proposal)
        if spec is not None:
            return spec
    return None


def _evaluate_experiment(
    evaluators: tuple[EvolutionEvaluationProvider, ...],
    experiment: EvolutionExperimentSpec,
    proposal: DecisionEvolutionProposal,
) -> tuple[EvolutionEvaluationFinding, ...]:
    return tuple(
        evaluator.evaluate(experiment, proposal=proposal) for evaluator in evaluators
    )


@dataclass(frozen=True, slots=True)
class AutonomousDecisionEvolutionEngine:
    proposal_generators: tuple[EvolutionProposalGenerator, ...]
    experiment_providers: tuple[EvolutionExperimentProvider, ...]
    evaluation_providers: tuple[EvolutionEvaluationProvider, ...]
    approval_providers: tuple[EvolutionApprovalProvider, ...]

    def plan_controlled_evolution(
        self,
        input_data: AutonomousDecisionEvolutionInput,
        *,
        planned_at: datetime | None = None,
    ) -> AutonomousDecisionEvolutionResult:
        stamp = planned_at or datetime.now(tz=UTC)
        generator_ids = tuple(item.generator_id for item in self.proposal_generators)
        experiment_ids = tuple(item.provider_id for item in self.experiment_providers)
        evaluator_ids = tuple(item.evaluator_id for item in self.evaluation_providers)
        approval_ids = tuple(item.provider_id for item in self.approval_providers)
        insight_ids = tuple(item.insight_id for item in input_data.insights)

        base_audit = _empty_audit(
            planned_at=stamp,
            source_insight_ids=insight_ids,
            proposal_generator_ids=generator_ids,
            experiment_provider_ids=experiment_ids,
            evaluation_provider_ids=evaluator_ids,
            approval_provider_ids=approval_ids,
        )

        if not input_data.insights:
            return AutonomousDecisionEvolutionResult(
                evolution_task_id=AUTONOMOUS_EVOLUTION_TASK_ID,
                status=EvolutionRunStatus.INSUFFICIENT_INPUT,
                audit=base_audit,
                proposals=(),
                experiments=(),
                evaluation_findings=(),
                approval_decisions=(),
                evolution_records=(),
            )

        proposals = _generate_proposals(self.proposal_generators, input_data, stamp)
        experiments: list[EvolutionExperimentSpec] = []
        findings: list[EvolutionEvaluationFinding] = []
        approvals: list[EvolutionApprovalDecision] = []
        records: list[ControlledEvolutionRecord] = []

        approval_provider = (
            self.approval_providers[0] if self.approval_providers else None
        )

        for proposal in proposals:
            experiment = _prepare_experiment(self.experiment_providers, proposal)
            if experiment is not None:
                experiments.append(experiment)
            proposal_findings: tuple[EvolutionEvaluationFinding, ...] = ()
            if experiment is not None:
                proposal_findings = _evaluate_experiment(
                    self.evaluation_providers, experiment, proposal
                )
                findings.extend(proposal_findings)
            approval: EvolutionApprovalDecision | None = None
            if experiment is not None and approval_provider is not None:
                approval = approval_provider.decide(
                    proposal, experiment, proposal_findings
                )
                approvals.append(approval)

            records.append(
                ControlledEvolutionRecord(
                    record_id=f"rec:{proposal.proposal_id}",
                    source_insight_ids=proposal.source_insight_ids,
                    proposal=proposal,
                    experiment=experiment,
                    evaluation_findings=proposal_findings,
                    approval=approval,
                    proposal_generator_ids=(proposal.generator_id,),
                    experiment_provider_ids=(
                        (experiment.experiment_provider_id,)
                        if experiment is not None
                        else ()
                    ),
                    evaluation_provider_ids=evaluator_ids,
                    approval_provider_id=(
                        approval.approval_provider_id if approval is not None else None
                    ),
                )
            )

        status = (
            EvolutionRunStatus.COMPLETE
            if proposals
            else EvolutionRunStatus.INSUFFICIENT_INPUT
        )
        return AutonomousDecisionEvolutionResult(
            evolution_task_id=AUTONOMOUS_EVOLUTION_TASK_ID,
            status=status,
            audit=base_audit,
            proposals=proposals,
            experiments=tuple(experiments),
            evaluation_findings=tuple(findings),
            approval_decisions=tuple(approvals),
            evolution_records=tuple(records),
        )


def default_autonomous_decision_evolution_engine(
    *,
    proposal_generators: tuple[EvolutionProposalGenerator, ...] | None = None,
    experiment_providers: tuple[EvolutionExperimentProvider, ...] | None = None,
    evaluation_providers: tuple[EvolutionEvaluationProvider, ...] | None = None,
    approval_providers: tuple[EvolutionApprovalProvider, ...] | None = None,
) -> AutonomousDecisionEvolutionEngine:
    return AutonomousDecisionEvolutionEngine(
        proposal_generators=proposal_generators or default_proposal_generators(),
        experiment_providers=experiment_providers or default_experiment_providers(),
        evaluation_providers=evaluation_providers or default_evaluation_providers(),
        approval_providers=approval_providers or default_approval_providers(),
    )


__all__ = [
    "AutonomousDecisionEvolutionEngine",
    "default_autonomous_decision_evolution_engine",
]
