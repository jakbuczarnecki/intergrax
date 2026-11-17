# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-D (PT-FC4) — Sqlite reference KV declares durability (PT-1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.skills.execution_pinning_persistence import KvSkillExecutionPinningStore
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

    assert first.compare_and_set("tenant-a", "skill_execution_pinning:exec-1", None, b"pin-bytes") is True

    # Restart simulation: a second store instance on the same file reads the same pin.
    second = SqliteDistributedKVStore(db_path)
    assert second.get("tenant-a", "skill_execution_pinning:exec-1") == b"pin-bytes"
    assert second.survives_process_restart is True
