# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export — canonical: llm_adapters.contracts.routing_profile."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.routing_profile import (
    AllowlistViolationError,
    LLMRoutingProfile,
    LLMRoutingRule,
    LLMRoutingRuleBase,
    RoutingContext,
    RoutingEvaluation,
    RoutingHint,
    RoutingTarget,
)


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
