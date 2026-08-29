# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.events.schema_guard import RuntimeEventSchemaError
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from intergrax.contracts.execution_phase import ExecutionPhase

pytestmark = pytest.mark.gate


def test_validating_store_rejects_invalid_schema():
    inner = InMemoryRuntimeEventStore()
    store = ValidatingRuntimeEventPersistence(inner)
    run_id = mint_run_id()
    event = RuntimeEvent(
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
        schema_version="runtime_event.v99",
        **runtime_event_test_identity(),
    )
    with pytest.raises(RuntimeEventSchemaError):
        store.append(event, tenant_id="lab")
    assert inner.list_for_run(run_id, tenant_id="lab") == []
