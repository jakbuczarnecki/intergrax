# © Artur Czarnecki. All rights reserved.

"""Pluggable autonomous decision evolution contracts (DS-E2E-15J-L11)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    AdaptiveDecisionInsight,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    DecisionEvolutionProposal,
    EvolutionApprovalDecision,
    EvolutionEvaluationFinding,
    EvolutionExperimentSpec,
    EvolutionTargetArea,
)


class EvolutionProposalGenerator(Protocol):
    """Creates evolution proposals from insights — never applies changes."""

    @property
    def generator_id(self) -> str: ...

    @property
    def generator_version(self) -> str: ...

    @property
    def target_area(self) -> EvolutionTargetArea: ...

    def generate(
        self,
        insights: tuple[AdaptiveDecisionInsight, ...],
        *,
        created_at: datetime,
    ) -> tuple[DecisionEvolutionProposal, ...]: ...


class EvolutionExperimentProvider(Protocol):
    """Designs a controlled experiment for a proposal — no production rollout."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def prepare_experiment(
        self,
        proposal: DecisionEvolutionProposal,
    ) -> EvolutionExperimentSpec | None: ...


class EvolutionEvaluationProvider(Protocol):
    """Evaluates an experiment design along one dimension — no hidden aggregation."""

    @property
    def evaluator_id(self) -> str: ...

    @property
    def evaluator_version(self) -> str: ...

    def evaluate(
        self,
        experiment: EvolutionExperimentSpec,
        *,
        proposal: DecisionEvolutionProposal,
    ) -> EvolutionEvaluationFinding: ...


class EvolutionApprovalProvider(Protocol):
    """Human or governance gate — never auto-approves from confidence alone."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def decide(
        self,
        proposal: DecisionEvolutionProposal,
        experiment: EvolutionExperimentSpec,
        evaluations: tuple[EvolutionEvaluationFinding, ...],
    ) -> EvolutionApprovalDecision: ...


__all__ = [
    "EvolutionApprovalProvider",
    "EvolutionEvaluationProvider",
    "EvolutionExperimentProvider",
    "EvolutionProposalGenerator",
]
