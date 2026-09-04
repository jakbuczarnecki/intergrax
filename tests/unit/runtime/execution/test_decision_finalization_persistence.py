# © Artur Czarnecki. All rights reserved.

"""DS-REC-01 — durable atomic finalization persistence conformance."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.execution.decision_finalization_conformance import (
    assert_concurrent_finalization_race,
    assert_concurrent_idempotent_replay,
    assert_decision_finalization_persistence_conformance,
)
from intergrax.runtime.execution.in_memory_decision_finalization_persistence import (
    InMemoryDecisionFinalizationPersistence,
)
from intergrax.runtime.execution.sqlite_decision_finalization_persistence import (
    SQLiteDecisionFinalizationPersistence,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_in_memory_finalization_conformance() -> None:
    assert_decision_finalization_persistence_conformance(
        InMemoryDecisionFinalizationPersistence,
        label="in_memory",
    )


def test_sqlite_finalization_conformance(tmp_path: Path) -> None:
    db_path = tmp_path / "finalization.db"

    def _factory() -> SQLiteDecisionFinalizationPersistence:
        return SQLiteDecisionFinalizationPersistence(db_path=db_path)

    assert_decision_finalization_persistence_conformance(_factory, label="sqlite")


def test_in_memory_concurrent_race() -> None:
    store = InMemoryDecisionFinalizationPersistence()
    assert_concurrent_finalization_race(lambda: store, label="in_memory")


def test_sqlite_concurrent_race(tmp_path: Path) -> None:
    db_path = tmp_path / "finalization-race.db"
    store = SQLiteDecisionFinalizationPersistence(db_path=db_path)
    assert_concurrent_finalization_race(lambda: store, label="sqlite")


def test_in_memory_concurrent_idempotent_replay() -> None:
    store = InMemoryDecisionFinalizationPersistence()
    assert_concurrent_idempotent_replay(lambda: store, label="in_memory")


def test_sqlite_concurrent_idempotent_replay(tmp_path: Path) -> None:
    db_path = tmp_path / "finalization-idempotent.db"
    store = SQLiteDecisionFinalizationPersistence(db_path=db_path)
    assert_concurrent_idempotent_replay(lambda: store, label="sqlite")


def test_sqlite_finalization_does_not_use_unconditional_overwrite() -> None:
    source = Path(
        "intergrax/runtime/execution/sqlite_decision_finalization_persistence.py",
    ).read_text(encoding="utf-8")
    assert "ON CONFLICT" not in source
    assert "DO UPDATE" not in source
    assert "INSERT INTO decision_finalizations" in source
