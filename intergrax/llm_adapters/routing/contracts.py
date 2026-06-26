# © Artur Czarnecki. All rights reserved.

"""LLM routing rule contracts (M-LLM-X.9 · ADR-LLM-003)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from intergrax.llm_adapters.registry.profile import LLMProfile


class RoutingHint(str, Enum):
    """Policy hints consumed by :class:`~intergrax.llm_adapters.registry.model_router.ModelRouter`."""

    BALANCED = "balanced"
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    QUALITY = "quality"


class RoutingContext(BaseModel):
    """Immutable routing snapshot — no side effects in rule evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_class: str | None = None
    budget_remaining_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    tokens_used: int | None = Field(default=None, ge=0)
    step_index: int | None = Field(default=None, ge=0)
    model_hint: str | None = None
    tenant_id: str = "default"
    agent_id: str | None = None
    budget_degrade_active: bool = False


class RoutingTarget(BaseModel):
    """Output of a matched routing rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: LLMProfile | None = None
    hint: RoutingHint | None = None
    reason: str = ""


@runtime_checkable
class LLMRoutingRule(Protocol):
    """Author-facing routing rule — built-in or custom Tier-3 subclass."""

    @property
    def rule_id(self) -> str: ...

    @property
    def priority(self) -> int: ...

    def matches(self, context: RoutingContext) -> bool: ...

    def resolve(self, context: RoutingContext) -> RoutingTarget: ...


class LLMRoutingRuleBase(ABC):
    """Optional base class with helper methods for custom rules."""

    rule_id: str
    priority: int = 0

    @abstractmethod
    def matches(self, context: RoutingContext) -> bool: ...

    @abstractmethod
    def resolve(self, context: RoutingContext) -> RoutingTarget: ...

    def budget_below(self, context: RoutingContext, ratio: float) -> bool:
        return (
            context.budget_remaining_ratio is not None
            and context.budget_remaining_ratio < ratio
        )

    def task_is(self, context: RoutingContext, *classes: str) -> bool:
        return context.task_class is not None and context.task_class in classes

    def tokens_above(self, context: RoutingContext, threshold: int) -> bool:
        return context.tokens_used is not None and context.tokens_used > threshold

    def budget_above(self, context: RoutingContext, ratio: float) -> bool:
        return (
            context.budget_remaining_ratio is not None
            and context.budget_remaining_ratio > ratio
        )

    def tokens_below(self, context: RoutingContext, threshold: int) -> bool:
        return context.tokens_used is not None and context.tokens_used < threshold

    def step_at_least(self, context: RoutingContext, min_step: int) -> bool:
        return context.step_index is not None and context.step_index >= min_step

    def step_below(self, context: RoutingContext, max_step: int) -> bool:
        return context.step_index is not None and context.step_index < max_step

    def agent_in(self, context: RoutingContext, *agent_ids: str) -> bool:
        return context.agent_id is not None and context.agent_id in agent_ids

    def tenant_in(self, context: RoutingContext, *tenant_ids: str) -> bool:
        return context.tenant_id in tenant_ids


class LLMRoutingProfile(BaseModel):
    """Tier-3 routing posture — rules are Python objects (not JSON-serializable)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    default_profile: LLMProfile
    allowed_profiles: tuple[LLMProfile, ...] = ()
    rules: SkipJsonSchema[tuple[LLMRoutingRule, ...]] = ()


class RoutingEvaluation(BaseModel):
    """Result of evaluator pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matched_rule_id: str | None
    target: RoutingTarget
    routing_reason: str
    selected_profile: LLMProfile
    policy_route_hint: str | None = None
