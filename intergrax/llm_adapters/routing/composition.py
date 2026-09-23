# © Artur Czarnecki. All rights reserved.

"""LLM routing composition — default implementation selection (EBH-2E-R6-R2)."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.routing_evaluator import RoutingEvaluator
from intergrax.llm_adapters.routing.evaluator import LLMRoutingEvaluator


def platform_default_routing_evaluator() -> RoutingEvaluator:
    """Return the platform default routing evaluator implementation."""
    return LLMRoutingEvaluator()


def resolve_routing_evaluator(
    routing_evaluator: RoutingEvaluator | None,
) -> RoutingEvaluator:
    """Select injected evaluator or platform default."""
    if routing_evaluator is not None:
        return routing_evaluator
    return platform_default_routing_evaluator()


__all__ = ["platform_default_routing_evaluator", "resolve_routing_evaluator"]
