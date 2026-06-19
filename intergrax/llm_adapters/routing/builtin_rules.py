# © Artur Czarnecki. All rights reserved.

"""Built-in LLM routing rules (M-LLM-X.9.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing.contracts import (
    LLMRoutingRuleBase,
    RoutingContext,
    RoutingHint,
    RoutingTarget,
)


def cheapest_allowed_model_hint(allowed_models: tuple[str, ...] | list[str]) -> str | None:
    """Shared degrade helper for budget reactions and routing rules (M-LLM-X.9.6)."""
    models = tuple(allowed_models)
    return models[-1] if models else None


@dataclass
class BudgetBelowRule(LLMRoutingRuleBase):
    """Route to a profile when remaining budget ratio drops below threshold."""

    threshold: float
    profile: LLMProfile
    rule_id: str = "builtin.budget_below"
    priority: int = field(default=10)

    def matches(self, context: RoutingContext) -> bool:
        return self.budget_below(context, self.threshold)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return RoutingTarget(
            profile=self.profile,
            reason=f"budget_remaining<{self.threshold}",
        )


@dataclass
class TaskClassRule(LLMRoutingRuleBase):
    """Route when Nexus task class is in the configured set."""

    classes: tuple[str, ...]
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.task_class"
    priority: int = field(default=5)

    def matches(self, context: RoutingContext) -> bool:
        return self.task_is(context, *self.classes)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return RoutingTarget(
            profile=self.profile,
            hint=self.hint,
            reason=f"task_class={context.task_class}",
        )


@dataclass
class TokenThresholdRule(LLMRoutingRuleBase):
    """Apply a routing hint after token usage crosses a threshold."""

    threshold: int
    hint: RoutingHint = RoutingHint.CHEAPEST
    rule_id: str = "builtin.token_threshold"
    priority: int = field(default=8)

    def matches(self, context: RoutingContext) -> bool:
        return self.tokens_above(context, self.threshold)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return RoutingTarget(
            hint=self.hint,
            reason=f"tokens_used>{self.threshold}",
        )


@dataclass
class BudgetExceededDegradeRule(LLMRoutingRuleBase):
    """Unifies ``BudgetReactionProfile.degrade_model`` with routing (M-LLM-X.9.6)."""

    rule_id: str = "builtin.budget_degrade"
    priority: int = field(default=15)

    def matches(self, context: RoutingContext) -> bool:
        return context.budget_degrade_active

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return RoutingTarget(
            hint=RoutingHint.CHEAPEST,
            reason="budget_degrade_active",
        )
