# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reusable guarded hosted process bootstrap primitive (DG-001B R2)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar
from uuid import uuid4

from pydantic import JsonValue

from intergrax.contracts.event_severity import EventSeverity
from intergrax.hosting.contracts.context import HostedApplicationEventPublisher
from intergrax.hosting.contracts.events import (
    HostedApplicationEvent,
    HostedApplicationEventType,
)
from intergrax.hosting.contracts.identity import normalize_application_id
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.contracts.public_data import (
    validate_bounded_identifier,
    validate_instance_id,
)

_LOGGER = logging.getLogger(__name__)

BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE = "bootstrap_unhandled_exception"

_T = TypeVar("_T")


def _mint_hosted_application_event_id() -> str:
    """Mint a bounded hosting event identifier compatible with event validation."""
    return validate_bounded_identifier(f"evt-{uuid4()}", field_name="event_id")


class HostedProcessBootstrapPhase(str, Enum):
    CONFIGURATION = "configuration"
    COMPOSITION = "composition"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    WORKER_CONSTRUCTION = "worker_construction"
    STARTUP = "startup"


@dataclass(frozen=True, slots=True)
class HostedProcessBootstrapFailureFacts:
    phase: HostedProcessBootstrapPhase
    reason_code: str
    exception_type: str
    process_role: str


def hosted_process_bootstrap_failure_payload(
    facts: HostedProcessBootstrapFailureFacts,
) -> dict[str, JsonValue]:
    """Bounded public payload facts for bootstrap APPLICATION_FAILED events."""
    return {
        "phase": facts.phase.value,
        "reason_code": facts.reason_code,
        "exception_type": facts.exception_type,
        "process_role": facts.process_role,
    }


@dataclass(frozen=True, slots=True)
class HostedProcessBootstrapContext:
    """Immutable identity context for guarded hosted process bootstrap."""

    application_id: str
    instance_id: str
    process_role: str

    def __post_init__(self) -> None:
        normalized_application_id = normalize_application_id(self.application_id)
        if normalized_application_id != self.application_id:
            raise ValueError(
                "application_id must be canonical (already normalized before construction)"
            )
        validated_instance_id = validate_instance_id(self.instance_id)
        if validated_instance_id != self.instance_id:
            raise ValueError(
                "instance_id must be canonical (already normalized before construction)"
            )
        validated_process_role = validate_bounded_identifier(
            self.process_role,
            field_name="process_role",
        )
        if validated_process_role != self.process_role:
            raise ValueError(
                "process_role must be canonical (already normalized before construction)"
            )

    @classmethod
    def create(
        cls,
        *,
        application_id: str,
        process_role: str,
    ) -> HostedProcessBootstrapContext:
        """Mint a new instance_id once before bootstrap callback execution."""
        return cls(
            application_id=normalize_application_id(application_id),
            instance_id=validate_instance_id(str(uuid4())),
            process_role=validate_bounded_identifier(
                process_role,
                field_name="process_role",
            ),
        )


async def run_guarded_hosted_process_bootstrap(
    *,
    context: HostedProcessBootstrapContext,
    phase: HostedProcessBootstrapPhase,
    event_publisher: HostedApplicationEventPublisher,
    bootstrap: Callable[[], _T],
) -> _T:
    """Run a sync bootstrap callback and emit APPLICATION_FAILED on application errors."""
    try:
        return bootstrap()
    except Exception as exc:
        event_id = _mint_hosted_application_event_id()
        occurred_at = datetime.now(UTC)
        facts = HostedProcessBootstrapFailureFacts(
            phase=phase,
            reason_code=BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE,
            exception_type=type(exc).__name__,
            process_role=context.process_role,
        )
        failure_event = HostedApplicationEvent(
            event_id=event_id,
            event_type=HostedApplicationEventType.APPLICATION_FAILED,
            occurred_at=occurred_at,
            application_id=context.application_id,
            instance_id=context.instance_id,
            lifecycle_state=HostedApplicationLifecycleState.FAILED,
            severity=EventSeverity.ERROR,
            payload=hosted_process_bootstrap_failure_payload(facts),
        )
        try:
            await event_publisher.publish(failure_event)
        except Exception as publish_exc:
            _LOGGER.error(
                "hosted process bootstrap failure event publish failed",
                extra={
                    "application_id": context.application_id,
                    "instance_id": context.instance_id,
                    "process_role": context.process_role,
                    "event_id": event_id,
                    "publish_exception_type": type(publish_exc).__name__,
                },
            )
        raise
