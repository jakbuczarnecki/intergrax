# © Artur Czarnecki. All rights reserved.

"""Elastic capacity scaling policy contracts (ECP-1.2 / ECP-1.3)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ScalingTarget(str, Enum):
    NEXUS_HOST = "nexus_host"
    CELERY_POOL = "celery_pool"
    MODALITY_POOL = "modality_pool"
    ORCHESTRATION_CEILING = "orchestration_ceiling"


class ScalingActionKind(str, Enum):
    SCALE_K8S_DEPLOYMENT = "scale_k8s_deployment"
    SCALE_CELERY_WORKERS = "scale_celery_workers"
    RAISE_ORCHESTRATION_CEILING = "raise_orchestration_ceiling"
    REQUEST_HITL = "request_hitl"


class ScalingRule(BaseModel):
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
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    require_hitl_for_scale_up: bool = False
    max_actions_per_hour: int = Field(default=6, ge=1, le=120)
    rules: list[ScalingRule] = Field(default_factory=list)
