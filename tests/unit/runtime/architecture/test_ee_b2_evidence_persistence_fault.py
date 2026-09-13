# © Artur Czarnecki. All rights reserved.

"""EE-B2 — mandatory evidence persistence fault injection."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
    MandatoryEvidencePersistenceError,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from testing_support.chaos.failing_persistence import FailOnAppendPersistence
from testing_support.chaos.fault_plan import FailOnCall

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b2_mandatory_evidence_append_failure_fail_closed() -> None:
    inner = InMemoryRuntimeEventStore()
    persistence = FailOnAppendPersistence(
        inner,
        fail_on=FailOnCall(call_number=1, message="append_fault"),
    )
    bus = RuntimeEventBus(persistence=persistence, record_history=True)
    event = sample_runtime_event(tenant_id="tenant-ee-b2")
    with pytest.raises(MandatoryEvidencePersistenceError):
        bus.record(event, tenant_id="tenant-ee-b2")
    assert bus.history == []
    assert inner.list_for_task(event.task_id, tenant_id="tenant-ee-b2") == []


def test_ee_b2_best_effort_append_failure_does_not_block_handler_path() -> None:
    from intergrax.runtime.events.evidence_durability import (
        EvidencePersistenceRequirement,
    )
    from intergrax.runtime.events.runtime_persistence_resilience import (
        resolve_runtime_persistence_failure,
    )
    from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
        EvidencePersistenceBoundaryError,
    )

    event = sample_runtime_event(tenant_id="tenant-be")
    resolve_runtime_persistence_failure(
        requirement=EvidencePersistenceRequirement.BEST_EFFORT,
        failure=EvidencePersistenceBoundaryError("sink down"),
        event_type=event.event_type,
    )
