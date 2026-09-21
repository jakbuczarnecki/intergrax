# © Artur Czarnecki. All rights reserved.

"""Elastic capacity contracts (ECP-1.2 / ECP-1.3)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


from intergrax.contracts.scaling_policy import (
    ScalingActionKind,
    ScalingPolicy,
    ScalingRule,
    ScalingTarget,
)


class CapacitySignal(BaseModel):
    """Observed capacity signal sample."""

    model_config = ConfigDict(extra="forbid")

    signal_id: str = Field(default_factory=lambda: f"sig_{uuid4().hex}")
    target: ScalingTarget
    metric_name: str
    value: float
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScalingAction(BaseModel):
    """Ordered scaling action."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(default_factory=lambda: f"act_{uuid4().hex}")
    kind: ScalingActionKind
    target: ScalingTarget
    delta: int = 1
    reason: str = ""


class ScalingActionPlan(BaseModel):
    """Immutable evaluated plan (ECP-3.3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(default_factory=lambda: f"splan_{uuid4().hex}")
    actions: tuple[ScalingAction, ...] = ()
    evaluation_status: Literal["noop", "planned", "denied", "hitl_required"] = "noop"
