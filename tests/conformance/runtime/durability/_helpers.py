# © Artur Czarnecki. All rights reserved.

"""Shared helpers for P0C-8 durability conformance tests."""

from __future__ import annotations

from intergrax.runtime.background_execution.admission_wiring import (
    BackgroundExecutionAdmissionDependencies,
)
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentry,
    admit_background_execution_reentry,
)
from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef


def transport_ref(
    *,
    tenant_id: str,
    task_id: str,
    provider: str = "broker",
) -> BackgroundTransportExecutionRef:
    return BackgroundTransportExecutionRef(
        tenant_id=tenant_id,
        provider=provider,
        transport_task_id=task_id,
    )


def admit(
    *,
    transport: BackgroundTransportExecutionRef,
    deps: BackgroundExecutionAdmissionDependencies,
) -> BackgroundExecutionReentry:
    return admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
