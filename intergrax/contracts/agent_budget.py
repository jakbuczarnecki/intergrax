# © Artur Czarnecki. All rights reserved.

"""Token budget limits and reaction contracts (architecture §25.5 · ACP-TOK-2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BudgetLimitEnforcement(StrEnum):
    """Whether a resolved token limit is hard-enforced by the harness."""

    HARD = "hard"
    ADVISORY = "advisory"


class BudgetExceededReaction(StrEnum):
    """Environment policy when a hard limit is crossed (Tier-3 configurable)."""

    ABORT = "abort"
    HITL = "hitl"
    DEGRADE_MODEL = "degrade_model"
    NOTIFY_ONLY = "notify_only"
    CUSTOM_HOOK = "custom_hook"


class BudgetNotifyChannel(StrEnum):
    """Integration slug references for user/operator notification (not vendor SDKs)."""

    IN_APP = "in_app"
    WEBHOOK = "webhook"
    SLACK = "slack"
    EMAIL = "email"
    TRACE_ONLY = "trace_only"


class AgentBudgetSlice(BaseModel):
    """Per-agent token budget assigned from application manifest (architecture §30.3 · §25.5)."""

    model_config = ConfigDict(extra="forbid")

    max_total_tokens: int | None = Field(default=None, ge=1)
    max_llm_calls: int | None = Field(default=None, ge=1)
    enforcement: BudgetLimitEnforcement = BudgetLimitEnforcement.HARD
    warn_threshold_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class BudgetReactionProfile(BaseModel):
    """
    Application-level policy for budget threshold and exceed reactions (§25.5.3).

    Wired on ``ApplicationEnvironmentProfile.cost_profile.budget_reaction``.
    """

    model_config = ConfigDict(extra="forbid")

    on_agent_limit_exceeded: BudgetExceededReaction = BudgetExceededReaction.ABORT
    on_environment_limit_exceeded: BudgetExceededReaction = BudgetExceededReaction.ABORT
    notify_channels: list[BudgetNotifyChannel] = Field(default_factory=list)
    warn_threshold_ratio: float = Field(default=0.80, ge=0.0, le=1.0)
    custom_hook_id: str | None = None
    user_message_template: str | None = None


class ResolvedBudgetLimits(BaseModel):
    """Materialized limits exposed read-only to agents at step boundary."""

    model_config = ConfigDict(extra="forbid")

    agent_tokens_limit: int | None = None
    agent_tokens_remaining: int | None = None
    agent_enforcement: BudgetLimitEnforcement = BudgetLimitEnforcement.ADVISORY
    environment_tokens_limit: int | None = None
    environment_tokens_remaining: int | None = None
    environment_enforcement: BudgetLimitEnforcement = BudgetLimitEnforcement.ADVISORY
    warn_threshold_ratio: float = Field(default=0.80, ge=0.0, le=1.0)
    limit_source: Literal["none", "binding", "environment", "request", "merged"] = "none"
