# © Artur Czarnecki. All rights reserved.

"""TaskEnvelope intake bridge (FAUDIT-INTAKE.1)."""

from __future__ import annotations

from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.runtime.interactions.factory import intake_payload_to_task
from intergrax.runtime.task.task import Task


def intake_envelope_to_task(envelope: TaskEnvelope) -> Task:
    """Materialize canonical Task from TaskEnvelope."""
    return Task.from_envelope(envelope)


def intake_payload_to_envelope(
    payload: dict[str, object],
    *,
    tenant_id: str,
    user_id: str | None = None,
) -> TaskEnvelope:
    """Normalize interaction payload to TaskEnvelope before Task materialization."""
    task = intake_payload_to_task(payload, tenant_id=tenant_id, user_id=user_id)
    return task.to_envelope()
