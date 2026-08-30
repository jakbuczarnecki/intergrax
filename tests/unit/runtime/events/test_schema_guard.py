# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError, assert_runtime_event_schema
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.event_severity import EventSeverity

pytestmark = pytest.mark.gate


def _event(**overrides) -> RuntimeEvent:
    base = dict(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
        severity=EventSeverity.INFO,
    )
    base.update(overrides)
    return RuntimeEvent(**base)


def test_assert_runtime_event_schema_accepts_canonical_event():
    assert_runtime_event_schema(_event())


def test_assert_runtime_event_schema_rejects_unknown_version():
    with pytest.raises(RuntimeEventSchemaError):
        assert_runtime_event_schema(_event(schema_version="runtime_event.v99"))


def test_assert_runtime_event_schema_rejects_phase_mismatch():
    with pytest.raises(RuntimeEventSchemaError):
        assert_runtime_event_schema(
            _event(
                event_type=RuntimeEventType.TASK_COMPLETED,
                phase=ExecutionPhase.INTAKE,
            )
        )
