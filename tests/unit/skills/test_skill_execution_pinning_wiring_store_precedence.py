# © Artur Czarnecki. All rights reserved.

"""SKILL-X — resolver store precedence when KV and document backends coexist."""

from __future__ import annotations

import pytest
from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore, DocumentRecord
from intergrax.skills.execution_binding import InMemorySkillExecutionPinningStore
from intergrax.skills.execution_pinning_durability import DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG
from intergrax.skills.execution_pinning_persistence import (
    DocumentStoreSkillExecutionPinningStore,
    KvSkillExecutionPinningStore,
    wire_skill_execution_pinning_store,
)
from intergrax.skills.resolver import SkillResolutionError
from intergrax.skills.skill_execution_pinning_wiring import resolve_skill_execution_pinning_wiring

pytestmark = pytest.mark.unit


class InMemoryKVStore(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}
        self.write_count = 0

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
        del ttl_seconds
        self._data[(tenant_id, key)] = value
        self.write_count += 1

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
        current = self._data.get((tenant_id, key))
        if current != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        self.write_count += 1
        return True


class InMemoryConditionalDocumentStore(ConditionalDocumentStore):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], DocumentRecord] = {}
        self.write_count = 0

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        return self._records.get((partition_key, row_key))

    def put(self, document: DocumentRecord) -> None:
        self._records[(document.partition_key, document.row_key)] = document
        self.write_count += 1

    def put_if_absent(self, document: DocumentRecord) -> bool:
        key = (document.partition_key, document.row_key)
        if key in self._records:
            return False
        self._records[key] = document
        self.write_count += 1
        return True

    def delete(self, partition_key: str, row_key: str) -> None:
        self._records.pop((partition_key, row_key), None)

    def query_partition(self, partition_key: str, *, page_size: int | None = None):
        del partition_key, page_size
        raise NotImplementedError


class PlainDocumentStore:
    """Minimal DocumentStore shape without ConditionalDocumentStore protocol."""

    def get(self, partition_key: str, row_key: str) -> DocumentRecord | None:
        del partition_key, row_key
        return None

    def put(self, document: DocumentRecord) -> None:
        del document


def test_both_backends_resolve_to_kv_store_t1() -> None:
    kv = InMemoryKVStore()
    doc = InMemoryConditionalDocumentStore()
    wiring = resolve_skill_execution_pinning_wiring(
        production_mode=True,
        kv_store=kv,
        document_store=doc,
    )
    assert isinstance(wiring.pinning_store, KvSkillExecutionPinningStore)
    assert wiring.pinning_store.is_durable is True


def test_both_with_plain_document_store_resolves_to_kv_t2() -> None:
    kv = InMemoryKVStore()
    doc = PlainDocumentStore()
    wiring = resolve_skill_execution_pinning_wiring(
        production_mode=True,
        kv_store=kv,
        document_store=doc,
    )
    assert isinstance(wiring.pinning_store, KvSkillExecutionPinningStore)


def test_kv_is_write_target_when_both_supplied_t3() -> None:
    kv = InMemoryKVStore()
    doc = InMemoryConditionalDocumentStore()
    wiring = resolve_skill_execution_pinning_wiring(
        production_mode=True,
        kv_store=kv,
        document_store=doc,
    )
    assert isinstance(wiring.pinning_store, KvSkillExecutionPinningStore)
    assert doc.write_count == 0


def test_document_store_only_uses_document_pinning_t4() -> None:
    doc = InMemoryConditionalDocumentStore()
    wiring = resolve_skill_execution_pinning_wiring(
        production_mode=True,
        document_store=doc,
    )
    assert isinstance(wiring.pinning_store, DocumentStoreSkillExecutionPinningStore)


def test_non_conditional_document_only_raises_type_error_t5() -> None:
    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        resolve_skill_execution_pinning_wiring(
            production_mode=True,
            document_store=PlainDocumentStore(),
        )


def test_no_stores_in_production_raises_durable_required_t6() -> None:
    with pytest.raises(SkillResolutionError, match=DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG):
        resolve_skill_execution_pinning_wiring(production_mode=True)


def test_explicit_pinning_store_wins_t7() -> None:
    explicit = InMemorySkillExecutionPinningStore()
    kv = InMemoryKVStore()
    doc = InMemoryConditionalDocumentStore()
    wiring = resolve_skill_execution_pinning_wiring(
        production_mode=True,
        pinning_store=explicit,
        kv_store=kv,
        document_store=doc,
    )
    assert wiring.pinning_store is explicit


def test_leaf_still_rejects_dual_backends_t8() -> None:
    kv = InMemoryKVStore()
    doc = InMemoryConditionalDocumentStore()
    with pytest.raises(ValueError, match="not both"):
        wire_skill_execution_pinning_store(kv_store=kv, document_store=doc)


def test_reference_strict_composition_shape_resolves_kv() -> None:
    platform = build_reference_production_platform_persistence()
    wiring = resolve_skill_execution_pinning_wiring(
        production_mode=True,
        kv_store=platform.kv_store,
        document_store=platform.document_store,
    )
    assert isinstance(wiring.pinning_store, KvSkillExecutionPinningStore)
