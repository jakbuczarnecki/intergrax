# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-D (PT-FC4) — KV durability capability and typed attestation (NT-9..NT-13)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from intergrax.distributed.contracts.kv_store import (
    DistributedKVStore,
    KVDurabilityAttestation,
    KVDurabilityMechanism,
    kv_store_survives_process_restart,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.skills.execution_pinning_durability import DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG
from intergrax.skills.execution_pinning_persistence import (
    DocumentStoreSkillExecutionPinningStore,
    KvSkillExecutionPinningStore,
)
from intergrax.skills.resolver import SkillResolutionError
from intergrax.skills.skill_execution_pinning_wiring import resolve_skill_execution_pinning_wiring

pytestmark = pytest.mark.unit


class _KV(DistributedKVStore):
    def __init__(self, capability: object = None) -> None:
        self._capability = capability

    @property
    def survives_process_restart(self) -> bool:
        if self._capability is None:
            return super().survives_process_restart
        return self._capability  # type: ignore[return-value]

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


class _TruthyDocStore(InMemoryDocumentStore):
    @property
    def survives_process_restart(self) -> bool:
        return "yes"  # type: ignore[return-value]


def test_kv_capability_defaults_to_false() -> None:
    assert _KV().survives_process_restart is False
    assert kv_store_survives_process_restart(_KV()) is False


@pytest.mark.parametrize("truthy", [1, "yes", MagicMock()])
def test_kv_capability_truthy_non_bool_is_not_durable(truthy: object) -> None:
    store = _KV(truthy)
    assert kv_store_survives_process_restart(store) is False
    with pytest.raises(SkillResolutionError, match=DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG):
        resolve_skill_execution_pinning_wiring(production_mode=True, kv_store=store)
    assert KvSkillExecutionPinningStore(store).is_durable is False


def test_kv_capability_true_bool_is_durable() -> None:
    store = _KV(True)
    assert kv_store_survives_process_restart(store) is True
    assert KvSkillExecutionPinningStore(store).is_durable is True


def test_document_store_capability_truthy_non_bool_is_not_durable() -> None:
    store = _TruthyDocStore()
    assert DocumentStoreSkillExecutionPinningStore(store).is_durable is False
    with pytest.raises(SkillResolutionError, match=DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG):
        resolve_skill_execution_pinning_wiring(production_mode=True, document_store=store)


def test_attestation_rejects_empty_attested_by() -> None:
    with pytest.raises(ValueError):
        KVDurabilityAttestation(
            mechanism=KVDurabilityMechanism.REDIS_AOF_FSYNC_ALWAYS,
            attested_by="  ",
            procedure_ref="x",
        )


def test_attestation_rejects_empty_procedure_ref() -> None:
    with pytest.raises(ValueError):
        KVDurabilityAttestation(
            mechanism=KVDurabilityMechanism.REDIS_AOF_FSYNC_EVERYSEC,
            attested_by="x",
            procedure_ref="",
        )


def test_attestation_rejects_non_enum_mechanism() -> None:
    with pytest.raises(ValueError):
        KVDurabilityAttestation(
            mechanism="redis_aof_fsync_always",  # type: ignore[arg-type]
            attested_by="x",
            procedure_ref="y",
        )


@pytest.mark.parametrize("mechanism", list(KVDurabilityMechanism))
def test_attestation_accepts_both_aof_mechanisms(mechanism: KVDurabilityMechanism) -> None:
    attestation = KVDurabilityAttestation(mechanism=mechanism, attested_by="ref-a", procedure_ref="ref-b")
    assert attestation.mechanism is mechanism


def test_mechanism_set_is_the_two_decided_aof_modes() -> None:
    assert {m.value for m in KVDurabilityMechanism} == {
        "redis_aof_fsync_always",
        "redis_aof_fsync_everysec",
    }


def test_duck_typed_kv_without_member_is_not_durable() -> None:
    class Duck:
        pass

    assert kv_store_survives_process_restart(Duck()) is False  # type: ignore[arg-type]
