# © Artur Czarnecki. All rights reserved.

"""OBS-CONTRACT-BOUNDARY-1-R1 — canonical RuntimeEvent contract ownership gates."""

from __future__ import annotations

import pytest

from intergrax.contracts.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.runtime_event import (
    RuntimeEvent as LegacyRuntimeEvent,
    RuntimeEventType as LegacyRuntimeEventType,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]


def test_legacy_runtime_event_import_path_is_canonical_contract_type() -> None:
    assert LegacyRuntimeEvent is RuntimeEvent
    assert LegacyRuntimeEventType is RuntimeEventType


def test_runtime_event_model_dump_round_trip_preserves_identity_fields() -> None:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_event_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )
    from intergrax.contracts.execution_phase import ExecutionPhase

    event = RuntimeEvent(
        event_id=mint_event_id(),
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        correlation_id="corr-1",
    )
    payload = event.model_dump(mode="json")
    restored = RuntimeEvent.model_validate(payload)
    assert restored.event_id == event.event_id
    assert restored.task_id == event.task_id
    assert restored.run_id == event.run_id
    assert restored.attempt_id == event.attempt_id
    assert restored.execution_id == event.execution_id
    assert restored.event_type == RuntimeEventType.STEP_STARTED
    assert restored.schema_version == "runtime_event.v2"
