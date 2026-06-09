# © Artur Czarnecki. All rights reserved.

"""Elastic capacity contracts (ECP-1.2 / ECP-1.3)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ScalingTarget(str, Enum):
    """Provision targets for scaling actions (ECP-1.3)."""

    NEXUS_HOST = "nexus_host"
    CELERY_POOL = "celery_pool"
    MODALITY_POOL = "modality_pool"
    ORCHESTRATION_CEILING = "orchestration_ceiling"


class ScalingActionKind(str, Enum):
    SCALE_K8S_DEPLOYMENT = "scale_k8s_deployment"
    SCALE_CELERY_WORKERS = "scale_celery_workers"
    RAISE_ORCHESTRATION_CEILING = "raise_orchestration_ceiling"
    REQUEST_HITL = "request_hitl"


class CapacitySignal(BaseModel):
    """Observed capacity signal sample."""

    model_config = ConfigDict(extra="forbid")

    signal_id: str = Field(default_factory=lambda: f"sig_{uuid4().hex}")
    target: ScalingTarget
    metric_name: str
    value: float
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScalingRule(BaseModel):
    """Single scaling rule with hysteresis thresholds."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    target: ScalingTarget
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    action_kind: ScalingActionKind
    delta: int = 1
    cooldown_seconds: int = Field(default=300, ge=0)


class ScalingPolicy(BaseModel):
    """Host scaling policy envelope."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    require_hitl_for_scale_up: bool = False
    max_actions_per_hour: int = Field(default=6, ge=1, le=120)
    rules: list[ScalingRule] = Field(default_factory=list)


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
