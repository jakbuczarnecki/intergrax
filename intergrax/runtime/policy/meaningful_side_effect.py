# © Artur Czarnecki. All rights reserved.

"""Meaningful side-effect policy boundary (GEC-5).

Composes ``MeaningfulSideEffectRequest`` with existing ``PolicyDecision``.
Not a second policy engine — evaluators plug into ``RuntimePolicyEngine`` /
``PolicyEngine`` or are injected as this Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyDecision


@runtime_checkable
class MeaningfulSideEffectEvaluator(Protocol):
    """Injectable evaluator for proposed external side effects."""

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        """Return ALLOW / DENY / REQUIRE_HUMAN (or other PolicyAction). Fail closed."""
        ...
