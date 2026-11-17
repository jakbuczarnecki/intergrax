# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-D (PT-FC4) — Redis KV durability only through a typed attestation (NT-14, PT-2).

No live Redis is used; a placeholder client object stands in. The attestation is declared, not probed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from intergrax.distributed.contracts.kv_store import (
    KVDurabilityAttestation,
    KVDurabilityMechanism,
    kv_store_survives_process_restart,
)
from intergrax.distributed.providers.redis_kv_store import RedisKVStore
from intergrax.integrations.providers.key_value_cache.redis.bundle import (
    create_redis_integration,
    create_redis_kv_store,
)
from intergrax.skills.execution_pinning_persistence import KvSkillExecutionPinningStore
from intergrax.skills.skill_execution_pinning_wiring import resolve_skill_execution_pinning_wiring

pytestmark = pytest.mark.unit


def _attestation(
    mechanism: KVDurabilityMechanism = KVDurabilityMechanism.REDIS_AOF_FSYNC_ALWAYS,
) -> KVDurabilityAttestation:
    return KVDurabilityAttestation(
        mechanism=mechanism,
        attested_by="test-attester-ref",
        procedure_ref="docs/project/architecture/SKILLS.md",
    )


def test_redis_kv_store_default_is_not_attested() -> None:
    store = RedisKVStore(SimpleNamespace())
    assert store.durability_attestation is None
    assert store.survives_process_restart is False


@pytest.mark.parametrize("bad", [True, {"mechanism": "redis_aof_fsync_always"}, "attested", 1])
def test_redis_kv_store_rejects_untyped_attestation(bad: object) -> None:
    with pytest.raises(TypeError):
        RedisKVStore(SimpleNamespace(), durability_attestation=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("mechanism", list(KVDurabilityMechanism))
def test_strict_redis_with_typed_attestation_is_accepted(mechanism: KVDurabilityMechanism) -> None:
    attestation = _attestation(mechanism)
    store = RedisKVStore(SimpleNamespace(), durability_attestation=attestation)
    assert store.durability_attestation is attestation
    assert store.survives_process_restart is True
    assert kv_store_survives_process_restart(store) is True

    wiring = resolve_skill_execution_pinning_wiring(production_mode=True, kv_store=store)
    assert isinstance(wiring.pinning_store, KvSkillExecutionPinningStore)
    assert wiring.pinning_store.is_durable is True


def test_bundle_composition_root_passes_attestation_through() -> None:
    attestation = _attestation()
    client = SimpleNamespace()
    bundle = create_redis_integration(client=client, durability_attestation=attestation)  # type: ignore[arg-type]
    assert kv_store_survives_process_restart(bundle.kv_store) is True
    store = create_redis_kv_store(client=client, durability_attestation=attestation)  # type: ignore[arg-type]
    assert kv_store_survives_process_restart(store) is True


def test_bundle_composition_root_default_is_not_attested() -> None:
    store = create_redis_kv_store(client=SimpleNamespace())  # type: ignore[arg-type]
    assert kv_store_survives_process_restart(store) is False
