# © Artur Czarnecki. All rights reserved.

"""Test doubles for DIAG-1I causal evidence admission path tests."""

from __future__ import annotations

from unittest.mock import Mock

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)


class _KV(DistributedKVStore):
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


def make_kv_store() -> _KV:
    return _KV()


def make_causal_persistence() -> InMemoryCausalEvidencePersistence:
    return InMemoryCausalEvidencePersistence()


def failing_causal_persistence() -> CausalEvidencePersistence:
    persistence = Mock(spec=CausalEvidencePersistence)
    persistence.append.side_effect = RuntimeError("backend unavailable")
    return persistence
