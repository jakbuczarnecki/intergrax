# © Artur Czarnecki. All rights reserved.

"""Governance policy evaluator plugin contract (DS-E2E-15J-L5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDisposition,
    GovernanceEvaluationRequest,
    GovernanceReasonRef,
)


@dataclass(frozen=True, slots=True)
class PolicyEvaluationResult:
    policy_id: str
    policy_version: str
    contribution: GovernanceDisposition
    reasons: tuple[GovernanceReasonRef, ...]


class PolicyEvaluator(Protocol):
    """Pluggable governance policy; evaluates whether a recommendation may proceed."""

    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    def evaluate(
        self, request: GovernanceEvaluationRequest
    ) -> PolicyEvaluationResult: ...


__all__ = ["PolicyEvaluationResult", "PolicyEvaluator"]
