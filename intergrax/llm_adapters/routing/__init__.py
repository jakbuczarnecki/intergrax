# © Artur Czarnecki. All rights reserved.

"""LLM routing rules — Protocol contract and evaluator (M-LLM-X.9)."""

from intergrax.llm_adapters.routing.builtin_rules import (
    BudgetBelowRule,
    BudgetExceededDegradeRule,
    TaskClassRule,
    TokenThresholdRule,
    cheapest_allowed_model_hint,
)
from intergrax.llm_adapters.routing.contracts import (
    LLMRoutingProfile,
    LLMRoutingRule,
    LLMRoutingRuleBase,
    RoutingContext,
    RoutingEvaluation,
    RoutingHint,
    RoutingTarget,
)
from intergrax.llm_adapters.routing.evaluator import (
    AllowlistViolationError,
    LLMRoutingEvaluator,
    effective_allowlist,
    is_profile_allowed,
    profile_identity,
)

__all__ = [
    "AllowlistViolationError",
    "BudgetBelowRule",
    "BudgetExceededDegradeRule",
    "LLMRoutingEvaluator",
    "LLMRoutingProfile",
    "LLMRoutingRule",
    "LLMRoutingRuleBase",
    "RoutingContext",
    "RoutingEvaluation",
    "RoutingHint",
    "RoutingTarget",
    "TaskClassRule",
    "TokenThresholdRule",
    "cheapest_allowed_model_hint",
    "effective_allowlist",
    "is_profile_allowed",
    "profile_identity",
]
