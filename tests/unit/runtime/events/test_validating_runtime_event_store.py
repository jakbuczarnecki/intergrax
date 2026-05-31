# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
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
    event = RuntimeEvent(
        task_id="t1",
        run_id="r1",
        event_type=RuntimeEventType.TASK_CREATED,
        phase=ExecutionPhase.INTAKE,
        schema_version="runtime_event.v99",
    )
    with pytest.raises(RuntimeEventSchemaError):
        store.append(event, tenant_id="lab")
    assert inner.list_for_run("r1", tenant_id="lab") == []
