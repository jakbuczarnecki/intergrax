# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-D (PT-FC4) — STRICT skill execution pinning refuses non-durable backing stores.

Negative tests NT-1..NT-8 plus lab/positive guards PT-3 and PT-4. Every refusal is a
``SkillResolutionError`` carrying ``DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.distributed.providers.redis_kv_store import RedisKVStore
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.skills.execution_pinning_durability import DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG
from intergrax.skills.execution_pinning_persistence import (
    DocumentStoreSkillExecutionPinningStore,
    KvSkillExecutionPinningStore,
)
from intergrax.skills.resolver import SkillResolutionError
from intergrax.skills.skill_execution_pinning_wiring import resolve_skill_execution_pinning_wiring

pytestmark = pytest.mark.unit


class InMemoryKV(DistributedKVStore):
    """Non-durable KV double: inherits the default-False capability."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(self, tenant_id: str, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        del ttl_seconds
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
        del ttl_seconds
        if self._data.get((tenant_id, key)) != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        return True


class DuckKV:
    """Duck-typed KV: the four methods, but neither the ABC nor the capability member."""

    def get(self, tenant_id: str, key: str) -> bytes | None:
        del tenant_id, key
        return None

    def set(self, tenant_id: str, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        del tenant_id, key, value, ttl_seconds

    def delete(self, tenant_id: str, key: str) -> None:
        del tenant_id, key

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        del tenant_id, key, expected, new_value, ttl_seconds
        return True


class DocStoreDouble(InMemoryDocumentStore):
    """Reference in-memory document store with a settable process-restart capability."""

    def __init__(self, *, survives: bool) -> None:
        super().__init__()
        self._survives = survives

    @property
    def survives_process_restart(self) -> bool:
        return self._survives


class DurableKV(InMemoryKV):
    @property
    def survives_process_restart(self) -> bool:
        return True


_REFUSED = pytest.raises(SkillResolutionError, match=DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG)


def test_strict_in_memory_kv_store_is_refused() -> None:
    with _REFUSED:
        resolve_skill_execution_pinning_wiring(production_mode=True, kv_store=InMemoryKV())


def test_strict_kv_without_capability_member_is_refused() -> None:
    with _REFUSED:
        resolve_skill_execution_pinning_wiring(
            production_mode=True,
            kv_store=DuckKV(),  # type: ignore[arg-type]
        )


def test_strict_in_memory_document_store_is_refused() -> None:
    store = InMemoryDocumentStore()
    assert store.survives_process_restart is False
    with _REFUSED:
        resolve_skill_execution_pinning_wiring(production_mode=True, document_store=store)


def test_strict_document_store_declaring_false_is_refused() -> None:
    with _REFUSED:
        resolve_skill_execution_pinning_wiring(
            production_mode=True,
            document_store=DocStoreDouble(survives=False),
        )


def test_strict_redis_without_attestation_is_refused() -> None:
    store = RedisKVStore(SimpleNamespace())
    with _REFUSED:
        resolve_skill_execution_pinning_wiring(production_mode=True, kv_store=store)


def test_strict_non_durable_kv_with_durable_document_store_is_refused() -> None:
    # Decided (OQ-1): KV stays authoritative; no silent fallback to the document store.
    with _REFUSED:
        resolve_skill_execution_pinning_wiring(
            production_mode=True,
            kv_store=InMemoryKV(),
            document_store=DocStoreDouble(survives=True),
        )


def test_strict_explicit_non_durable_pinning_store_is_refused() -> None:
    with _REFUSED:
        resolve_skill_execution_pinning_wiring(
            production_mode=True,
            pinning_store=KvSkillExecutionPinningStore(InMemoryKV()),
        )


class _NoIsDurable:
    pass


class _NonBoolIsDurable:
    @property
    def is_durable(self) -> object:
        return "yes"


@pytest.mark.parametrize("pinning_store", [_NoIsDurable(), _NonBoolIsDurable()])
def test_strict_explicit_pinning_store_without_or_nonbool_is_durable_is_refused(
    pinning_store: object,
) -> None:
    with _REFUSED:
        resolve_skill_execution_pinning_wiring(
            production_mode=True,
            pinning_store=pinning_store,  # type: ignore[arg-type]
        )


def test_strict_durable_document_store_is_accepted() -> None:
    wiring = resolve_skill_execution_pinning_wiring(
        production_mode=True,
        document_store=DocStoreDouble(survives=True),
    )
    assert isinstance(wiring.pinning_store, DocumentStoreSkillExecutionPinningStore)
    assert wiring.pinning_store.is_durable is True


def test_strict_durable_kv_double_is_accepted() -> None:
    wiring = resolve_skill_execution_pinning_wiring(production_mode=True, kv_store=DurableKV())
    assert isinstance(wiring.pinning_store, KvSkillExecutionPinningStore)
    assert wiring.pinning_store.is_durable is True


def test_lab_mode_still_allows_in_memory_stores() -> None:
    kv_wiring = resolve_skill_execution_pinning_wiring(production_mode=False, kv_store=InMemoryKV())
    assert kv_wiring.pinning_store.is_durable is False
    doc_wiring = resolve_skill_execution_pinning_wiring(
        production_mode=False,
        document_store=InMemoryDocumentStore(),
    )
    assert doc_wiring.pinning_store.is_durable is False
    assert resolve_skill_execution_pinning_wiring(production_mode=False).pinning_store.is_durable is False
