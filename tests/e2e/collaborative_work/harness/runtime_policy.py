# © Artur Czarnecki. All rights reserved.

"""Deterministic runtime meaningful-side-effect evaluator for E2E scenarios."""

from __future__ import annotations

from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyDecision


class MutableRuntimePolicyEvaluator:
    """Typed runtime policy surface — no mocks, explicit decision control."""

    def __init__(self, decision: PolicyDecision) -> None:
        self._decision = decision
        self.calls: list[MeaningfulSideEffectRequest] = []

    def set_decision(self, decision: PolicyDecision) -> None:
        self._decision = decision

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        self.calls.append(request)
        return self._decision
