# © Artur Czarnecki. All rights reserved.

"""Shared doubles for background execution re-entry admission tests."""

from __future__ import annotations

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.background_execution.admission_wiring import (
    BackgroundExecutionAdmissionDependencies,
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.execution_terminal import (
    ExecutionTerminalService,
    InMemoryExecutionTerminalStore,
)


class InMemoryKVStore(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        current = self.get(tenant_id, key)
        if expected is None and current is not None:
            return False
        if expected is not None and current != expected:
            return False
        self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
        return True


def make_inmemory_admission_dependencies() -> BackgroundExecutionAdmissionDependencies:
    return BackgroundExecutionAdmissionDependencies(
        identity_persistence=wire_background_execution_admission_dependencies(
            kv_store=InMemoryKVStore(),
        ).identity_persistence,
        attempt_lifecycle=AttemptLifecycleService(InMemoryAttemptLifecycleStore()),
        execution_terminal=ExecutionTerminalService(InMemoryExecutionTerminalStore()),
    )


def make_kv_admission_dependencies(
    kv_store: DistributedKVStore | None = None,
) -> BackgroundExecutionAdmissionDependencies:
    return wire_background_execution_admission_dependencies(
        kv_store=kv_store or InMemoryKVStore(),
    )


def make_document_store_admission_dependencies(
    document_store: InMemoryDocumentStore | None = None,
) -> BackgroundExecutionAdmissionDependencies:
    return wire_background_execution_admission_dependencies(
        document_store=document_store or InMemoryDocumentStore(),
    )


def admission_kwargs(deps: BackgroundExecutionAdmissionDependencies) -> dict[str, object]:
    return {
        "identity_persistence": deps.identity_persistence,
        "attempt_lifecycle": deps.attempt_lifecycle,
        "execution_terminal": deps.execution_terminal,
    }
