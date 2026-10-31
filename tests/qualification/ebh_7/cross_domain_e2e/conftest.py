# © Artur Czarnecki. All rights reserved.

"""Fixtures for EBH-7 cross-domain platform proofs."""

from __future__ import annotations

import pytest
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.events.active_runtime_event_recorder import (
    bind_active_runtime_event_recorder,
    reset_active_runtime_event_recorder,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.execution.failure_evidence.active_context import (
    ActiveExecutionEvidenceContext,
    bind_active_execution_evidence_context,
    reset_active_execution_evidence_context,
)
from intergrax.runtime.execution.failure_evidence.runtime_event_recorder import (
    RuntimeEventExecutionFailureEvidenceRecorder,
)


@pytest.fixture
def governed_execution():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    task_id = mint_task_id()
    id_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        task_id=task_id,
    )
    bus = RuntimeEventBus(record_history=True)
    failure_recorder = RuntimeEventExecutionFailureEvidenceRecorder(bus)
    evidence_token = bind_active_execution_evidence_context(
        ActiveExecutionEvidenceContext(
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            recorder=failure_recorder,
        ),
    )
    recorder_binding = bind_active_runtime_event_recorder(bus, tenant_id="tenant-a")
    try:
        yield run_id, task_id, attempt_id, execution_id, bus
    finally:
        reset_active_runtime_event_recorder(recorder_binding)
        reset_active_execution_evidence_context(evidence_token)
        reset_active_execution_identity(id_token)
