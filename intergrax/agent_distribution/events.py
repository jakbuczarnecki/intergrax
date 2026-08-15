# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pure domain transition events for Agent Distribution (AP-4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class AgentDistributionEvent:
    """Bounded audit event emitted by domain services — no bus coupling."""

    event_type: str
    aggregate_id: str
    occurred_at: datetime
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransitionResult[T]:
    """Service transition outcome with optional audit events."""

    value: T
    events: tuple[AgentDistributionEvent, ...] = ()


def distribution_event(
    event_type: str,
    aggregate_id: str,
    **attributes: str,
) -> AgentDistributionEvent:
    return AgentDistributionEvent(
        event_type=event_type,
        aggregate_id=aggregate_id,
        occurred_at=datetime.now(UTC),
        attributes={key: str(value) for key, value in attributes.items()},
    )
