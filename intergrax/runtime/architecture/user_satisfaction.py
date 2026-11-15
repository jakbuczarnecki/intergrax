# © Artur Czarnecki. All rights reserved.

"""User satisfaction capture schema and online-eval bridge (MVP-EVOL.5)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.architecture.online_evaluation_registry import (
    FileOnlineEvaluationRegistry,
    TenantOnlineEvaluationObservation,
    default_online_evaluation_registry,
)


class SatisfactionSignal(str, Enum):  # noqa: UP042 - str mixin kept: str() output is part of the public contract
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CSAT = "csat"
    NPS = "nps"


class UserSatisfactionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"sat_{uuid4().hex[:12]}")
    tenant_id: str
    task_id: str
    run_id: str
    agent_id: str = ""
    signal: SatisfactionSignal
    score: float = Field(ge=0.0, le=10.0)
    comment: str = ""
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def satisfaction_to_online_observation(event: UserSatisfactionEvent) -> OnlineEvaluationObservation:
    """Map an event to a tenant-stamped observation (a blank tenant fails closed)."""
    passed = event.signal is SatisfactionSignal.THUMBS_UP or event.score >= 7.0
    normalized = (
        event.score / 10.0
        if event.signal in {SatisfactionSignal.CSAT, SatisfactionSignal.NPS}
        else (1.0 if passed else 0.0)
    )
    return TenantOnlineEvaluationObservation(
        tenant_id=event.tenant_id,
        observation_id=event.event_id,
        run_id=event.run_id,
        agent_id=event.agent_id or "unknown",
        mode=OnlineEvaluationMode.ONLINE,
        scenario_id=f"user_satisfaction:{event.signal.value}",
        passed=passed,
        score=normalized,
        recorded_at=event.recorded_at,
    )


def record_user_satisfaction(
    event: UserSatisfactionEvent,
    *,
    registry: FileOnlineEvaluationRegistry | None = None,
) -> OnlineEvaluationObservation:
    observation = satisfaction_to_online_observation(event)
    target = registry or default_online_evaluation_registry()
    target.append(observation)
    return observation
