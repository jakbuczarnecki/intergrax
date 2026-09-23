# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export — canonical: llm_adapters.contracts.routing_profile."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.routing_profile import (
    LLMRoutingProfile,
    LLMRoutingRule,
    LLMRoutingRuleBase,
    RoutingContext,
    RoutingEvaluation,
    RoutingHint,
    RoutingTarget,
)


class AllowlistViolationError(ValueError):
    """Raised when a rule resolves to a profile outside ``allowed_profiles``."""


__all__ = [
    "AllowlistViolationError",
    "LLMRoutingProfile",
    "LLMRoutingRule",
    "LLMRoutingRuleBase",
    "RoutingContext",
    "RoutingEvaluation",
    "RoutingHint",
    "RoutingTarget",
]
