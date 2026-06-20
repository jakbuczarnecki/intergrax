# © Artur Czarnecki. All rights reserved.

"""Built-in LLM routing rules (M-LLM-X.9.3 · M-LLM-X.10.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing.contracts import (
    LLMRoutingRule,
    LLMRoutingRuleBase,
    RoutingContext,
    RoutingHint,
    RoutingTarget,
)


def cheapest_allowed_model_hint(allowed_models: tuple[str, ...] | list[str]) -> str | None:
    """Shared degrade helper for budget reactions and routing rules (M-LLM-X.9.6)."""
    models = tuple(allowed_models)
    return models[-1] if models else None


def _routing_target(
    *,
    profile: LLMProfile | None = None,
    hint: RoutingHint | None = None,
    reason: str,
) -> RoutingTarget:
    return RoutingTarget(profile=profile, hint=hint, reason=reason)


@dataclass
class BudgetBelowRule(LLMRoutingRuleBase):
    """Route when remaining budget ratio drops below threshold."""

    threshold: float
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.budget_below"
    priority: int = field(default=10)

    def matches(self, context: RoutingContext) -> bool:
        return self.budget_below(context, self.threshold)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"budget_remaining<{self.threshold}",
        )


@dataclass
class BudgetAboveRule(LLMRoutingRuleBase):
    """Route when remaining budget ratio is above threshold."""

    threshold: float
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.budget_above"
    priority: int = field(default=10)

    def matches(self, context: RoutingContext) -> bool:
        return self.budget_above(context, self.threshold)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"budget_remaining>{self.threshold}",
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
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"task_class={context.task_class}",
        )


TaskClassInRule = TaskClassRule


@dataclass
class TaskClassNotInRule(LLMRoutingRuleBase):
    """Route when task class is outside the configured set."""

    classes: tuple[str, ...]
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.task_class_not_in"
    priority: int = field(default=5)

    def matches(self, context: RoutingContext) -> bool:
        return context.task_class is not None and context.task_class not in self.classes

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"task_class_not_in={self.classes}",
        )


@dataclass
class TokenThresholdRule(LLMRoutingRuleBase):
    """Apply a routing hint after token usage crosses a threshold."""

    threshold: int
    hint: RoutingHint = RoutingHint.CHEAPEST
    profile: LLMProfile | None = None
    rule_id: str = "builtin.token_threshold"
    priority: int = field(default=8)

    def matches(self, context: RoutingContext) -> bool:
        return self.tokens_above(context, self.threshold)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"tokens_used>{self.threshold}",
        )


TokenUsedAboveRule = TokenThresholdRule


@dataclass
class TokenUsedBelowRule(LLMRoutingRuleBase):
    """Route when token usage is below a threshold."""

    threshold: int
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.token_used_below"
    priority: int = field(default=8)

    def matches(self, context: RoutingContext) -> bool:
        return self.tokens_below(context, self.threshold)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"tokens_used<{self.threshold}",
        )


@dataclass
class StepIndexAtLeastRule(LLMRoutingRuleBase):
    """Route when the current step index reaches a minimum."""

    min_step: int
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.step_index_at_least"
    priority: int = field(default=6)

    def matches(self, context: RoutingContext) -> bool:
        return self.step_at_least(context, self.min_step)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"step_index>={self.min_step}",
        )


@dataclass
class StepIndexBelowRule(LLMRoutingRuleBase):
    """Route when the current step index is below a maximum."""

    max_step: int
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.step_index_below"
    priority: int = field(default=6)

    def matches(self, context: RoutingContext) -> bool:
        return self.step_below(context, self.max_step)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"step_index<{self.max_step}",
        )


@dataclass
class AgentIdInRule(LLMRoutingRuleBase):
    """Route when agent id is in the configured set."""

    agent_ids: tuple[str, ...]
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.agent_id_in"
    priority: int = field(default=7)

    def matches(self, context: RoutingContext) -> bool:
        return self.agent_in(context, *self.agent_ids)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"agent_id={context.agent_id}",
        )


@dataclass
class TenantIdInRule(LLMRoutingRuleBase):
    """Route when tenant id is in the configured set."""

    tenant_ids: tuple[str, ...]
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.tenant_id_in"
    priority: int = field(default=7)

    def matches(self, context: RoutingContext) -> bool:
        return self.tenant_in(context, *self.tenant_ids)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason=f"tenant_id={context.tenant_id}",
        )


@dataclass
class ModelHintPresentRule(LLMRoutingRuleBase):
    """Route when the caller supplied a non-empty model hint."""

    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.model_hint_present"
    priority: int = field(default=4)

    def matches(self, context: RoutingContext) -> bool:
        return bool(context.model_hint and context.model_hint.strip())

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason="model_hint_present",
        )


@dataclass
class PolicyHintRule(LLMRoutingRuleBase):
    """Always matches and resolves to a policy hint (use with low priority)."""

    hint: RoutingHint
    rule_id: str = "builtin.policy_hint"
    priority: int = field(default=-50)

    def matches(self, context: RoutingContext) -> bool:
        return True

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(hint=self.hint, reason=f"policy_hint_{self.hint.value}")


@dataclass
class CompositeAllRule(LLMRoutingRuleBase):
    """Route when all nested rules match."""

    rules: tuple[LLMRoutingRule, ...]
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.composite_all"
    priority: int = field(default=12)

    def matches(self, context: RoutingContext) -> bool:
        return all(rule.matches(context) for rule in self.rules)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason="composite_all",
        )


@dataclass
class CompositeAnyRule(LLMRoutingRuleBase):
    """Route when any nested rule matches."""

    rules: tuple[LLMRoutingRule, ...]
    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.composite_any"
    priority: int = field(default=11)

    def matches(self, context: RoutingContext) -> bool:
        return any(rule.matches(context) for rule in self.rules)

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason="composite_any",
        )


@dataclass
class AlwaysRule(LLMRoutingRuleBase):
    """Unconditional fallback rule."""

    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    rule_id: str = "builtin.always"
    priority: int = field(default=-100)

    def matches(self, context: RoutingContext) -> bool:
        return True

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(
            profile=self.profile,
            hint=self.hint,
            reason="always",
        )


@dataclass
class BudgetExceededDegradeRule(LLMRoutingRuleBase):
    """Unifies ``BudgetReactionProfile.degrade_model`` with routing (M-LLM-X.9.6)."""

    rule_id: str = "builtin.budget_degrade"
    priority: int = field(default=15)

    def matches(self, context: RoutingContext) -> bool:
        return context.budget_degrade_active

    def resolve(self, context: RoutingContext) -> RoutingTarget:
        return _routing_target(hint=RoutingHint.CHEAPEST, reason="budget_degrade_active")


BUILTIN_ROUTING_RULE_TYPES: tuple[type[LLMRoutingRuleBase], ...] = (
    AgentIdInRule,
    AlwaysRule,
    BudgetAboveRule,
    BudgetBelowRule,
    BudgetExceededDegradeRule,
    CompositeAllRule,
    CompositeAnyRule,
    ModelHintPresentRule,
    PolicyHintRule,
    StepIndexAtLeastRule,
    StepIndexBelowRule,
    TaskClassInRule,
    TaskClassNotInRule,
    TaskClassRule,
    TenantIdInRule,
    TokenThresholdRule,
    TokenUsedAboveRule,
    TokenUsedBelowRule,
)
