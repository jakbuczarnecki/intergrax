# © Artur Czarnecki. All rights reserved.

"""Canonical routing evaluator contract (EBH-2E-R6-R2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.llm_adapters.contracts.routing_profile import (
    LLMRoutingProfile,
    RoutingContext,
    RoutingEvaluation,
)


@runtime_checkable
class RoutingEvaluator(Protocol):
    """Pluggable routing rule evaluation (first-match + allowlist)."""

    def evaluate(
        self,
        profile: LLMRoutingProfile,
        context: RoutingContext,
    ) -> RoutingEvaluation:
        """Evaluate routing rules for the given profile and runtime snapshot."""


__all__ = ["RoutingEvaluator"]
