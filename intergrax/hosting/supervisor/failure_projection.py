# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic supervisor pre-engine failure event projection (DG-001D R2)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.event_severity import EventSeverity
from intergrax.hosting.contracts.events import (
    HostedApplicationEvent,
    HostedApplicationEventType,
)
from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.errors import HostedApplicationSupervisorError
from intergrax.hosting.process_bootstrap import (
    HostedProcessBootstrapFailureFacts,
    hosted_process_bootstrap_failure_payload,
)

HOSTED_APPLICATION_SUPERVISOR_PROCESS_ROLE = "hosted_application_supervisor"


def supervisor_pre_engine_failure_to_hosted_event(
    *,
    application_id: str,
    instance_id: str,
    failure: HostedApplicationSupervisorError,
    occurred_at: datetime,
) -> HostedApplicationEvent:
    """Project a bounded APPLICATION_FAILED event from a typed supervisor failure."""
    facts = HostedProcessBootstrapFailureFacts(
        phase=failure.phase,
        reason_code=failure.reason.value,
        exception_type=HostedApplicationSupervisorError.__name__,
        process_role=HOSTED_APPLICATION_SUPERVISOR_PROCESS_ROLE,
    )
    return HostedApplicationEvent(
        event_type=HostedApplicationEventType.APPLICATION_FAILED,
        occurred_at=occurred_at,
        application_id=application_id,
        instance_id=instance_id,
        lifecycle_state=HostedApplicationLifecycleState.FAILED,
        severity=EventSeverity.ERROR,
        payload=hosted_process_bootstrap_failure_payload(facts),
    )
