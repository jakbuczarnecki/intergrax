# © Artur Czarnecki. All rights reserved.

"""P0C-7A — durable terminal authority for background re-entry composition."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
    ExecutionTerminalRecord,
    ExecutionTerminalStore,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.background_execution.admission_wiring import (
    validate_background_execution_admission_durability,
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentryAdmissionError,
    BackgroundExecutionReentryDisposition,
    admit_background_execution_reentry,
)
from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef
from intergrax.runtime.execution.execution_terminal import (
    ExecutionTerminalService,
    InMemoryExecutionTerminalStore,
    KvExecutionTerminalStore,
)
from intergrax.runtime.execution.execution_terminal.persistence import (
    DocumentStoreExecutionTerminalStore,
    decode_terminal_record,
    encode_terminal_record,
    normalize_terminal_record,
)
from tests.unit.runtime.background_execution.reentry_admission_doubles import (
    InMemoryKVStore,
    make_document_store_admission_dependencies,
    make_kv_admission_dependencies,
)


pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"


def _transport(task_id: str = "transport-1") -> BackgroundTransportExecutionRef:
    return BackgroundTransportExecutionRef(
        tenant_id=_TENANT,
        provider="broker",
        transport_task_id=task_id,
    )


def _sample_record(**overrides: object) -> ExecutionTerminalRecord:
    base = {
        "tenant_id": _TENANT,
        "task_id": "task-1",
        "run_id": None,
        "outcome": ExecutionTerminalOutcome.COMPLETED,
        "reason": "completed",
        "recorded_at_utc": "2026-01-01T00:00:00Z",
    }
    base.update(overrides)
    return ExecutionTerminalRecord(**base)


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda: KvExecutionTerminalStore(InMemoryKVStore()),
        lambda: DocumentStoreExecutionTerminalStore(InMemoryDocumentStore()),
    ],
    ids=["kv", "document_store"],
)
def test_terminal_store_put_if_absent_winner_and_duplicate(store_factory) -> None:
    store = store_factory()
    record = normalize_terminal_record(_sample_record())
    assert store.put_if_absent(record) is True
    assert store.put_if_absent(record) is False
    assert store.load_record(tenant_id=_TENANT, task_id="task-1") == record


def test_terminal_store_conflict_write_returns_false() -> None:
    store = KvExecutionTerminalStore(InMemoryKVStore())
    first = normalize_terminal_record(_sample_record(outcome=ExecutionTerminalOutcome.COMPLETED))
    second = normalize_terminal_record(_sample_record(outcome=ExecutionTerminalOutcome.FAILED))
    assert store.put_if_absent(first) is True
    assert store.put_if_absent(second) is False
    loaded = store.load_record(tenant_id=_TENANT, task_id="task-1")
    assert loaded is not None
    assert loaded.outcome is ExecutionTerminalOutcome.COMPLETED


def test_terminal_store_tenant_isolation() -> None:
    kv = InMemoryKVStore()
    store = KvExecutionTerminalStore(kv)
    record_a = normalize_terminal_record(_sample_record(tenant_id="tenant-a", task_id="shared"))
    record_b = normalize_terminal_record(_sample_record(tenant_id="tenant-b", task_id="shared"))
    assert store.put_if_absent(record_a) is True
    assert store.put_if_absent(record_b) is True
    assert store.load_record(tenant_id="tenant-a", task_id="shared") == record_a
    assert store.load_record(tenant_id="tenant-b", task_id="shared") == record_b


def test_terminal_store_corrupt_record_fails_closed() -> None:
    kv = InMemoryKVStore()
    kv.set(_TENANT, "execution_terminal:task-1", b"not-json")
    store = KvExecutionTerminalStore(kv)
    with pytest.raises(ExecutionTerminalError):
        store.load_record(tenant_id=_TENANT, task_id="task-1")


def test_terminal_store_restart_via_new_adapter_instance() -> None:
    kv = InMemoryKVStore()
    store_a = KvExecutionTerminalStore(kv)
    record = normalize_terminal_record(_sample_record())
    store_a.put_if_absent(record)
    store_b = KvExecutionTerminalStore(kv)
    assert store_b.load_record(tenant_id=_TENANT, task_id="task-1") == record
    assert store_b.is_durable is True


def test_terminal_codec_round_trip() -> None:
    record = normalize_terminal_record(_sample_record())
    assert decode_terminal_record(encode_terminal_record(record)) == record


@pytest.mark.parametrize(
    "outcome",
    [
        ExecutionTerminalOutcome.COMPLETED,
        ExecutionTerminalOutcome.FAILED,
        ExecutionTerminalOutcome.CANCELLED,
    ],
)
def test_terminal_restart_matrix_denies_redelivery(outcome: ExecutionTerminalOutcome) -> None:
    kv = InMemoryKVStore()
    deps_a = make_kv_admission_dependencies(kv)
    transport = _transport(task_id=f"restart-{outcome.value}")
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps_a.identity_persistence,
        attempt_lifecycle=deps_a.attempt_lifecycle,
        execution_terminal=deps_a.execution_terminal,
    )
    deps_a.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=outcome,
    )
    deps_b = make_kv_admission_dependencies(kv)
    redelivery = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps_b.identity_persistence,
        attempt_lifecycle=deps_b.attempt_lifecycle,
        execution_terminal=deps_b.execution_terminal,
    )
    assert redelivery.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED


def test_no_terminal_record_allows_execute_after_restart() -> None:
    kv = InMemoryKVStore()
    deps_a = make_kv_admission_dependencies(kv)
    transport = _transport(task_id="no-terminal")
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps_a.identity_persistence,
        attempt_lifecycle=deps_a.attempt_lifecycle,
        execution_terminal=deps_a.execution_terminal,
    )
    deps_b = make_kv_admission_dependencies(kv)
    second = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps_b.identity_persistence,
        attempt_lifecycle=deps_b.attempt_lifecycle,
        execution_terminal=deps_b.execution_terminal,
    )
    assert first.disposition is BackgroundExecutionReentryDisposition.EXECUTE
    assert second.disposition is BackgroundExecutionReentryDisposition.EXECUTE
    assert second.identity.task_id == first.identity.task_id


def test_a2_active_with_cancelled_terminal_blocks_without_a3() -> None:
    kv = InMemoryKVStore()
    deps = make_kv_admission_dependencies(kv)
    transport = _transport(task_id="a2-cancelled")
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    a2 = deps.attempt_lifecycle.transition_to_next_attempt(
        tenant_id=first.identity.tenant_id,
        run_id=first.identity.run_id,
        expected_attempt_id=first.identity.attempt_id,
        reason=AttemptTransitionReason.RETRY,
    ).active_attempt_id
    deps.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
    )
    restarted = make_kv_admission_dependencies(kv)
    redelivery = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=restarted.identity_persistence,
        attempt_lifecycle=restarted.attempt_lifecycle,
        execution_terminal=restarted.execution_terminal,
    )
    assert redelivery.identity.attempt_id == a2
    assert redelivery.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED


def test_store_unavailable_fails_closed_on_redelivery() -> None:
    deps = make_kv_admission_dependencies()
    transport = _transport(task_id="store-down")
    admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    broken_terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    broken_terminal._store.load_record = MagicMock(  # type: ignore[method-assign]
        side_effect=ExecutionTerminalError("store unavailable"),
    )
    with pytest.raises(BackgroundExecutionReentryAdmissionError, match="terminal"):
        admit_background_execution_reentry(
            transport_ref=transport,
            identity_persistence=deps.identity_persistence,
            attempt_lifecycle=deps.attempt_lifecycle,
            execution_terminal=broken_terminal,
        )


def test_production_wiring_uses_durable_terminal_authorities() -> None:
    kv = InMemoryKVStore()
    admission = wire_background_execution_admission_dependencies(kv_store=kv)
    assert admission.attempt_lifecycle.store.is_durable is True
    assert admission.execution_terminal.store.is_durable is True
    validate_background_execution_admission_durability(admission)


def test_document_store_production_wiring_uses_durable_terminal() -> None:
    store = InMemoryDocumentStore()
    admission = wire_background_execution_admission_dependencies(document_store=store)
    assert admission.execution_terminal.store.is_durable is True
    assert isinstance(admission.execution_terminal.store, DocumentStoreExecutionTerminalStore)


def test_production_wiring_rejects_in_memory_terminal_override() -> None:
    with pytest.raises(ExecutionTerminalError, match="durable execution terminal"):
        wire_background_execution_admission_dependencies(
            kv_store=InMemoryKVStore(),
            execution_terminal_store=InMemoryExecutionTerminalStore(),
        )


class _CustomDurableTerminalStore(ExecutionTerminalStore):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ExecutionTerminalRecord] = {}

    @property
    def is_durable(self) -> bool:
        return True

    def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        return self._records.get((tenant_id, task_id))

    def put_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        key = (record.tenant_id, record.task_id)
        if key in self._records:
            return False
        self._records[key] = record
        return True


def test_custom_durable_terminal_store_plugs_into_admission_dependencies() -> None:
    custom = _CustomDurableTerminalStore()
    admission = wire_background_execution_admission_dependencies(
        kv_store=InMemoryKVStore(),
        execution_terminal_store=custom,
    )
    assert admission.execution_terminal.store is custom
    assert admission.execution_terminal.store.is_durable is True


def test_wire_never_returns_in_memory_terminal_for_kv_platform_store() -> None:
    admission = wire_background_execution_admission_dependencies(kv_store=InMemoryKVStore())
    assert isinstance(admission.execution_terminal.store, KvExecutionTerminalStore)
    assert not isinstance(admission.execution_terminal.store, InMemoryExecutionTerminalStore)


def test_wire_never_returns_in_memory_terminal_for_document_platform_store() -> None:
    admission = wire_background_execution_admission_dependencies(
        document_store=InMemoryDocumentStore(),
    )
    assert isinstance(admission.execution_terminal.store, DocumentStoreExecutionTerminalStore)
    assert not isinstance(admission.execution_terminal.store, InMemoryExecutionTerminalStore)


def test_kv_admission_dependencies_use_separate_key_namespaces() -> None:
    kv = InMemoryKVStore()
    deps = make_kv_admission_dependencies(kv)
    transport = _transport(task_id="namespace-proof")
    admitted = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    deps.execution_terminal.commit_terminal_outcome(
        tenant_id=admitted.identity.tenant_id,
        task_id=str(admitted.identity.task_id),
        run_id=admitted.identity.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
    )
    keys = {key for (_tenant, key) in kv._data}
    assert any(key.startswith("bg_exec_identity:") for key in keys)
    assert any(key.startswith("attempt_lifecycle:") for key in keys)
    assert any(key.startswith("execution_terminal:") for key in keys)
    assert not any(key.startswith("bg_exec_identity:execution_terminal:") for key in keys)
