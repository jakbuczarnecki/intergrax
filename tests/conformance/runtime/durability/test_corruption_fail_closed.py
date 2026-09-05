# © Artur Czarnecki. All rights reserved.

"""Corruption and store-unavailable fail-closed proofs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptLifecycleError
from intergrax.contracts.execution_terminal import ExecutionTerminalError
from intergrax.runtime.background_execution.admission_wiring import (
    BackgroundExecutionAdmissionDependencies,
)
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentryAdmissionError,
)
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from intergrax.runtime.execution.execution_terminal.persistence import KvExecutionTerminalStore

from tests.conformance.runtime.durability._helpers import admit, transport_ref
from tests.conformance.runtime.durability.provider_factories import (
    DurableAdmissionBacking,
    DurableProviderKind,
    create_admission_dependencies,
)
from tests.conformance.runtime.durability.restart import fresh_admission_composition
from tests.unit.runtime.background_execution.reentry_admission_doubles import InMemoryKVStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _inject_identity_corruption(
    backing: DurableAdmissionBacking,
    *,
    tenant_id: str,
    provider: str,
    transport_task_id: str,
) -> None:
    if backing.kind is not DurableProviderKind.KV or backing.kv_store is None:
        raise AssertionError("identity corruption injection requires KV backing")
    backing.kv_store.set(
        tenant_id,
        f"bg_exec_identity:{provider}:{transport_task_id}",
        b"not-valid",
    )


def _inject_attempt_corruption(
    backing: DurableAdmissionBacking,
    *,
    tenant_id: str,
    run_id: str,
) -> None:
    if backing.kind is not DurableProviderKind.KV or backing.kv_store is None:
        raise AssertionError("attempt corruption injection requires KV backing")
    backing.kv_store.set(tenant_id, f"attempt_lifecycle:{run_id}", b"not-json")


def _inject_terminal_corruption(
    backing: DurableAdmissionBacking,
    *,
    tenant_id: str,
    task_id: str,
) -> None:
    if backing.kind is not DurableProviderKind.KV or backing.kv_store is None:
        raise AssertionError("terminal corruption injection requires KV backing")
    backing.kv_store.set(tenant_id, f"execution_terminal:{task_id}", b"not-json")


def test_corrupt_identity_record_fails_closed_on_restart() -> None:
    backing = DurableAdmissionBacking.fresh_kv()
    process_a = create_admission_dependencies(backing)
    transport = transport_ref(tenant_id="tenant-corrupt", task_id="identity-corrupt")
    admit(transport=transport, deps=process_a)
    _inject_identity_corruption(
        backing,
        tenant_id=transport.tenant_id,
        provider=transport.provider,
        transport_task_id=transport.transport_task_id,
    )
    process_b = fresh_admission_composition(backing)
    with pytest.raises(BackgroundExecutionReentryAdmissionError):
        admit(transport=transport, deps=process_b)


def test_corrupt_attempt_lifecycle_record_fails_closed_on_restart() -> None:
    backing = DurableAdmissionBacking.fresh_kv()
    process_a = create_admission_dependencies(backing)
    transport = transport_ref(tenant_id="tenant-corrupt", task_id="attempt-corrupt")
    first = admit(transport=transport, deps=process_a)
    _inject_attempt_corruption(
        backing,
        tenant_id=first.identity.tenant_id,
        run_id=str(first.identity.run_id),
    )
    process_b = fresh_admission_composition(backing)
    with pytest.raises(BackgroundExecutionReentryAdmissionError):
        admit(transport=transport, deps=process_b)


def test_corrupt_terminal_record_fails_closed_on_restart() -> None:
    backing = DurableAdmissionBacking.fresh_kv()
    process_a = create_admission_dependencies(backing)
    transport = transport_ref(tenant_id="tenant-corrupt", task_id="terminal-corrupt")
    first = admit(transport=transport, deps=process_a)
    _inject_terminal_corruption(
        backing,
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
    )
    process_b = fresh_admission_composition(backing)
    with pytest.raises(BackgroundExecutionReentryAdmissionError):
        admit(transport=transport, deps=process_b)


def test_terminal_store_corruption_fails_closed_via_adapter() -> None:
    kv = InMemoryKVStore()
    kv.set("tenant-corrupt", "execution_terminal:task-1", b"not-json")
    store = KvExecutionTerminalStore(kv)
    with pytest.raises(ExecutionTerminalError):
        store.load_record(tenant_id="tenant-corrupt", task_id="task-1")


def test_store_unavailable_fails_closed_on_redelivery_after_restart() -> None:
    backing = DurableAdmissionBacking.fresh_kv()
    process_a = create_admission_dependencies(backing)
    transport = transport_ref(tenant_id="tenant-down", task_id="store-down")
    admit(transport=transport, deps=process_a)
    process_b = fresh_admission_composition(backing)
    broken_terminal = ExecutionTerminalService(process_b.execution_terminal.store)
    broken_terminal._store.load_record = MagicMock(  # type: ignore[method-assign]
        side_effect=ExecutionTerminalError("store unavailable"),
    )
    with pytest.raises(BackgroundExecutionReentryAdmissionError, match="terminal"):
        admit(
            transport=transport,
            deps=BackgroundExecutionAdmissionDependencies(
                identity_persistence=process_b.identity_persistence,
                attempt_lifecycle=process_b.attempt_lifecycle,
                execution_terminal=broken_terminal,
            ),
        )


def test_unsupported_attempt_lifecycle_schema_fails_closed() -> None:
    backing = DurableAdmissionBacking.fresh_kv()
    process_a = create_admission_dependencies(backing)
    transport = transport_ref(tenant_id="tenant-schema", task_id="schema")
    first = admit(transport=transport, deps=process_a)
    if backing.kv_store is not None:
        backing.kv_store.set(
            first.identity.tenant_id,
            f"attempt_lifecycle:{first.identity.run_id}",
            b'{"schema_version":999}',
        )
    process_b = fresh_admission_composition(backing)
    with pytest.raises(BackgroundExecutionReentryAdmissionError):
        admit(transport=transport, deps=process_b)
