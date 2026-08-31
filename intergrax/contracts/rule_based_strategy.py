# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Rule-Based deliberation strategy foundation (DS-DELIB-06).

Typed deterministic evaluator boundary for candidate production without LLM,
provider, or platform-owned rule DSL. Host supplies domain logic; Decision
Lifecycle and Verification Pipeline own finalization and assurance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_record import (
    CandidateDecision,
    DecisionArtifactKind,
    DecisionVersionLineage,
    candidate_decision,
)
from intergrax.contracts.decision_strategy import (
    DecisionStrategyKind,
    DecisionStrategyRegistration,
    DecisionStrategyRegistry,
    register_decision_strategy,
    validate_decision_strategy_kind,
)

_RULE_BASED_KIND = validate_decision_strategy_kind("rule_based")

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)


def rule_based_strategy_kind() -> DecisionStrategyKind:
    """Canonical deliberation strategy identity for Rule-Based."""
    return _RULE_BASED_KIND


class RuleBasedEvaluator(Protocol[InputT_contra, OutputT_co]):
    """Deterministic domain evaluation supplied by the host application."""

    def evaluate(self, decision_input: InputT_contra) -> OutputT_co:
        """Map typed deliberation input to typed semantic output."""
        ...


@dataclass(frozen=True, slots=True)
class RuleBasedStrategy(Generic[InputT, OutputT]):
    """Rule-Based deliberation strategy — one typed evaluator per candidate path."""

    evaluator: RuleBasedEvaluator[InputT, OutputT]
    kind: DecisionStrategyKind = field(default=_RULE_BASED_KIND)

    def __post_init__(self) -> None:
        validated_kind = validate_decision_strategy_kind(self.kind)
        if validated_kind != _RULE_BASED_KIND:
            raise ValueError(
                "RuleBasedStrategy.kind must be rule_based "
                f"got {validated_kind!r}",
            )


def evaluate_rule_based_strategy(
    *,
    strategy: RuleBasedStrategy[InputT, OutputT],
    decision_input: InputT,
) -> OutputT:
    """Canonical evaluation seam for Rule-Based deliberation."""
    return strategy.evaluator.evaluate(decision_input)


def rule_based_strategy_registration(
    evaluator: RuleBasedEvaluator[InputT, OutputT],
) -> DecisionStrategyRegistration:
    """Build one explicit Rule-Based registration for host/bootstrap wiring."""
    strategy = RuleBasedStrategy(evaluator=evaluator)
    return DecisionStrategyRegistration(kind=strategy.kind, strategy=strategy)


def register_rule_based_strategy(
    registry: DecisionStrategyRegistry,
    evaluator: RuleBasedEvaluator[InputT, OutputT],
) -> DecisionStrategyRegistry:
    """Register Rule-Based on ``registry``; return a new immutable registry."""
    return register_decision_strategy(
        registry,
        rule_based_strategy_registration(evaluator),
    )


def rule_based_candidate_decision(
    *,
    identity: DecisionIdentity,
    artifact_kind: DecisionArtifactKind,
    payload: OutputT,
    lineage: DecisionVersionLineage | None = None,
) -> CandidateDecision[OutputT]:
    """Assemble a typed candidate from Rule-Based evaluator output."""
    return candidate_decision(
        identity=identity,
        artifact_kind=artifact_kind,
        payload=payload,
        lineage=lineage,
    )
