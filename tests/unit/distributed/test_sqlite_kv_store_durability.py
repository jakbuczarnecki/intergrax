# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-D (PT-FC4) — Sqlite reference KV declares durability (PT-1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.execution_binding import (
    bind_resolved_skill_pack,
    recover_skill_execution_binding,
)
from intergrax.skills.execution_pinning_persistence import KvSkillExecutionPinningStore
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolver
from intergrax.skills.skill_execution_pinning_wiring import resolve_skill_execution_pinning_wiring

pytestmark = pytest.mark.unit


def test_sqlite_kv_declares_process_restart_durability(tmp_path: Path) -> None:
    store = SqliteDistributedKVStore(tmp_path / "kv.sqlite3")
    assert store.survives_process_restart is True


def test_strict_sqlite_kv_is_accepted_and_pin_survives_new_instance(tmp_path: Path) -> None:
    db_path = tmp_path / "kv.sqlite3"
    first = SqliteDistributedKVStore(db_path)
    wiring = resolve_skill_execution_pinning_wiring(production_mode=True, kv_store=first)
    assert isinstance(wiring.pinning_store, KvSkillExecutionPinningStore)
    assert wiring.pinning_store.is_durable is True

    registry = SkillRegistry()
    registry.register(SkillManifest(skill_id="a.pack", description="a.pack", version="1.0.0", tool_ids=("tool.x",)))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=wiring.pinning_store,
        resolved_composition=composition,
    )

    # Restart simulation: a second store instance on the same file recovers the pin through the pin store.
    second = SqliteDistributedKVStore(db_path)
    recovered = recover_skill_execution_binding(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=KvSkillExecutionPinningStore(second),
    )
    assert recovered.resolved_pack.snapshot_digest == binding.resolved_pack.snapshot_digest
    assert second.survives_process_restart is True
