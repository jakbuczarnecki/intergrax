# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P4-R2 requirement spine and mandatory durability."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.contracts.execution_integration_configuration_provenance_requirement import (
    derive_integration_configuration_provenance_requirement_event_id,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.integrations.execution_integration_configuration_requirement_fact import (
    build_requirement_fact_from_pin_record,
)
from intergrax.runtime.events.evidence_durability import (
    EvidencePersistenceRequirement,
    evidence_persistence_requirement,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import MandatoryEvidencePersistenceError
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.execution.integration_configuration_provenance_requirement_recorder import (
    RuntimeEventIntegrationConfigurationProvenanceRequirementRecorder,
)
from intergrax.applications._shared.integrations.integration_configuration_provenance_requirement_commit import (
    RuntimeEventBusIntegrationConfigurationProvenanceRequirementCommitPort,
)
from tests.unit.applications.integrations.test_trace_x_p5_r2_p2_persistence import (
    _EXEC,
    _provenance_configured_adopted,
    _recovery_staging,
    _subject,
)
from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
    ExecutionIntegrationConfigurationPinRecord,
)

pytestmark = pytest.mark.unit


def _pin_record() -> ExecutionIntegrationConfigurationPinRecord:
    return ExecutionIntegrationConfigurationPinRecord(
        subject=_subject(),
        provenance=_provenance_configured_adopted(),
        requirement_recovery_staging=_recovery_staging(),
    )


def test_requirement_event_mandatory_durability_classification() -> None:
    fact = build_requirement_fact_from_pin_record(_pin_record())
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    recorder = RuntimeEventIntegrationConfigurationProvenanceRequirementRecorder(bus)
    result = recorder.commit(fact)
    assert result.event_id is not None
    events = store.list_positioned_for_run(fact.run_id, tenant_id=fact.tenant_id)
    assert len(events) == 1
    assert (
        evidence_persistence_requirement(events[0].event)
        is EvidencePersistenceRequirement.MANDATORY
    )


def test_requirement_event_exact_retry_equality() -> None:
    fact = build_requirement_fact_from_pin_record(_pin_record())
    store = InMemoryRuntimeEventStore()
    recorder = RuntimeEventIntegrationConfigurationProvenanceRequirementRecorder(
        RuntimeEventBus(persistence=store),
    )
    first = recorder.commit(fact)
    second = recorder.commit(fact)
    assert first.event_id == second.event_id
    assert store.list_positioned_for_run(fact.run_id, tenant_id=fact.tenant_id)[0].position.value == 1


class _FailingPersistence:
    def append(self, event, *, tenant_id: str):
        raise MandatoryEvidencePersistenceError("forced")

    def query_run(self, tenant_id, run_id, *, limit=1000):
        return ()


def test_requirement_persistence_failure_fail_closed() -> None:
    fact = build_requirement_fact_from_pin_record(_pin_record())
    bus = RuntimeEventBus(persistence=_FailingPersistence())
    port = RuntimeEventBusIntegrationConfigurationProvenanceRequirementCommitPort(bus)
    result = port.commit_configured_adopted_requirement(fact)
    from intergrax.contracts.execution_integration_configuration_provenance_requirement import (
        ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus,
    )

    assert (
        result.status
        is ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus.PERSISTENCE_UNAVAILABLE
    )


def test_deterministic_event_id_stable() -> None:
    subject = _subject()
    first = derive_integration_configuration_provenance_requirement_event_id(
        tenant_id="tenant-a",
        execution_id=_EXEC,
        subject=subject,
    )
    second = derive_integration_configuration_provenance_requirement_event_id(
        tenant_id="tenant-a",
        execution_id=_EXEC,
        subject=subject,
    )
    assert first == second


def test_requirement_event_type_registered() -> None:
    assert (
        RuntimeEventType.INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED.value
        == "integration_configuration_provenance_requirement_committed"
    )
