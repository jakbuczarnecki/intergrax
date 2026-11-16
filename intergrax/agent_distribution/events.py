# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pure domain transition events for Agent Distribution (AP-4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

# Canonical lifecycle event vocabulary (PDC-D1-R017). One event per transition.
EVENT_INSTALLATION_CREATED: Final = "installation.created"
EVENT_INSTALLATION_VERIFIED: Final = "installation.verified"
EVENT_INSTALLATION_ACTIVATED: Final = "installation.activated"
EVENT_INSTALLATION_FAILED_CANDIDATE: Final = "installation.failed_candidate"
EVENT_INSTALLATION_REVOKED: Final = "installation.revoked"
EVENT_INSTALLATION_TOMBSTONED: Final = "installation.tombstoned"
EVENT_INSTALLATION_RETIRED: Final = "installation.retired"
EVENT_RUNTIME_REVISION_CREATED: Final = "runtime_revision.created"
EVENT_RUNTIME_REVISION_VALIDATED: Final = "runtime_revision.validated"
EVENT_RUNTIME_REVISION_ACTIVATED: Final = "runtime_revision.activated"


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
