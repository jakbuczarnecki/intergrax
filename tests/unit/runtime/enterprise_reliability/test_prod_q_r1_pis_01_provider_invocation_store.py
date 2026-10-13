# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-PIS-01 — durable ProviderInvocationStore and atomic dispatch admission."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pytest

from applications.governed_contractor_application.host.provider_invocation_lifecycle import (
    GovernedProviderInvocationDispatchGate,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.contracts.provider_invocation_store import (
    ProviderInvocationConflictError,
    ProviderInvocationIntentWriteResult,
    ProviderInvocationOutcomeConflictError,
    ProviderInvocationPersistenceError,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_lifecycle import (
    persist_provider_invocation_intent,
    persist_provider_invocation_outcome,
)
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    SQLiteProviderInvocationStore,
    open_provider_invocation_store_at,
)

pytestmark = [pytest.mark.unit]

_T0 = datetime(2026, 10, 13, 10, 0, 0, tzinfo=timezone.utc)


def _invocation(invocation_id: str = "inv-pis-01") -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id=invocation_id,
        provider_id="provider.test",
        operation="create_work",
        task_id="task-1",
        run_id="run-1",
        request_digest="sha256:" + ("cd" * 32),
        started_at=_T0,
    )


def _outcome(invocation_id: str = "inv-pis-01") -> ProviderInvocationOutcome:
    return ProviderInvocationOutcome(
        invocation_id=invocation_id,
        status=ProviderInvocationStatus.SUCCEEDED,
        completed_at=_T0,
        response_digest="sha256:" + ("ef" * 32),
    )


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteProviderInvocationStore:
    store = open_provider_invocation_store_at(tmp_path / "provider_invocations.db")
    yield store
    store.close()


def test_pis_n1_equivalent_repeated_intent(sqlite_store: SQLiteProviderInvocationStore) -> None:
    inv = _invocation()
    assert sqlite_store.insert_invocation_if_absent(inv) is (
        ProviderInvocationIntentWriteResult.INSERTED
    )
    assert sqlite_store.insert_invocation_if_absent(inv) is (
        ProviderInvocationIntentWriteResult.EQUIVALENT_NOOP
    )
    sqlite_store.put_invocation(inv)
    persist_provider_invocation_intent(sqlite_store, inv)
    assert sqlite_store.get_invocation(inv.invocation_id) == inv


def test_pis_n2_conflicting_intent(sqlite_store: SQLiteProviderInvocationStore) -> None:
    inv = _invocation()
    sqlite_store.put_invocation(inv)
    other = inv.model_copy(update={"task_id": "task-2"})
    with pytest.raises(ProviderInvocationConflictError):
        sqlite_store.put_invocation(other)
    with pytest.raises(ProviderInvocationConflictError):
        sqlite_store.insert_invocation_if_absent(other)


def test_pis_n3_outcome_idempotency_and_conflict(
    sqlite_store: SQLiteProviderInvocationStore,
) -> None:
    inv = _invocation()
    sqlite_store.put_invocation(inv)
    outcome = _outcome()
    sqlite_store.put_outcome(outcome)
    sqlite_store.put_outcome(outcome)
    persist_provider_invocation_outcome(sqlite_store, outcome)
    other = outcome.model_copy(update={"status": ProviderInvocationStatus.FAILED})
    with pytest.raises(ProviderInvocationOutcomeConflictError):
        sqlite_store.put_outcome(other)


def test_pis_n4_concurrent_first_writer(sqlite_store: SQLiteProviderInvocationStore) -> None:
    inv = _invocation("inv-concurrent")
    results: list[ProviderInvocationIntentWriteResult] = []

    def _admit() -> None:
        results.append(sqlite_store.insert_invocation_if_absent(inv))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_admit) for _ in range(16)]
        for future in as_completed(futures):
            future.result()

    assert results.count(ProviderInvocationIntentWriteResult.INSERTED) == 1
    assert results.count(ProviderInvocationIntentWriteResult.EQUIVALENT_NOOP) == 15


def test_pis_n5_failed_intent_persistence_prevents_execute(
    sqlite_store: SQLiteProviderInvocationStore,
) -> None:
    inv = _invocation("inv-fail-intent")

    class _FailOnInsert(SQLiteProviderInvocationStore):
        def insert_invocation_if_absent(self, invocation: ProviderInvocation):
            raise ProviderInvocationPersistenceError("simulated intent commit failure")

    gate = GovernedProviderInvocationDispatchGate(store=_FailOnInsert(sqlite_store.db_path), clock=lambda: _T0)
    calls = 0

    def _execute() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(ProviderInvocationPersistenceError):
        gate.dispatch_after_intent_persisted(inv, _execute)
    assert calls == 0


def test_pis_n6_process_restart_restores_records(tmp_path: Path) -> None:
    db_path = tmp_path / "restart.db"
    inv = _invocation("inv-restart")
    outcome = _outcome("inv-restart")
    store_a = open_provider_invocation_store_at(db_path)
    store_a.put_invocation(inv)
    store_a.put_outcome(outcome)
    store_a.close()

    script = f"""
import sys
from pathlib import Path
sys.path.insert(0, r"{Path.cwd()}")
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    open_provider_invocation_store_at,
)
store = open_provider_invocation_store_at(Path(r"{db_path}"))
assert store.get_invocation("inv-restart") is not None
assert store.get_outcome("inv-restart") is not None
store.close()
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        timeout=15,
        cwd=Path.cwd(),
    )


def test_pis_n8_corrupt_sqlite_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"not-a-sqlite-file\x00\x01")
    with pytest.raises(ProviderInvocationPersistenceError):
        open_provider_invocation_store_at(db_path)


def test_pis_n11_gate_reentry_no_second_execute(sqlite_store: SQLiteProviderInvocationStore) -> None:
    inv = _invocation("inv-gate-reentry")
    gate = GovernedProviderInvocationDispatchGate(store=sqlite_store, clock=lambda: _T0)
    calls = 0

    def _execute() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert gate.dispatch_after_intent_persisted(inv, _execute) == "ok"
    assert calls == 1
    with pytest.raises(ProviderInvocationPersistenceError, match="dispatch_not_admitted"):
        gate.dispatch_after_intent_persisted(inv, _execute)
    assert calls == 1


def test_pis_n12_concurrent_gate_at_most_one_execute(
    sqlite_store: SQLiteProviderInvocationStore,
) -> None:
    inv = _invocation("inv-gate-concurrent")
    gate = GovernedProviderInvocationDispatchGate(store=sqlite_store, clock=lambda: _T0)
    calls = 0
    lock = threading.Lock()

    def _execute() -> str:
        nonlocal calls
        with lock:
            calls += 1
        return "ok"

    errors: list[BaseException] = []

    def _dispatch() -> None:
        try:
            gate.dispatch_after_intent_persisted(inv, _execute)
        except ProviderInvocationPersistenceError as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_dispatch) for _ in range(12)]
        for future in as_completed(futures):
            future.result()

    assert calls == 1
    assert len(errors) == 11


def test_pis_n13_uncertain_commit_no_dispatch(
    sqlite_store: SQLiteProviderInvocationStore,
) -> None:
    inv = _invocation("inv-uncertain")
    boundary_store = _CommitBoundaryTestStore(sqlite_store.db_path)
    boundary_store.fail_commit = True
    gate = GovernedProviderInvocationDispatchGate(store=boundary_store, clock=lambda: _T0)
    calls = 0

    def _execute() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(ProviderInvocationPersistenceError):
        gate.dispatch_after_intent_persisted(inv, _execute)
    assert calls == 0
    assert boundary_store.get_invocation(inv.invocation_id) is None
    boundary_store.close()


class _CommitBoundaryTestStore(SQLiteProviderInvocationStore):
    """Injects SQLite commit/rollback failures at the store commit boundary."""

    fail_commit: bool = False
    noop_rollback: bool = False

    @classmethod
    def _connection_commit(cls, conn: sqlite3.Connection) -> None:
        if getattr(cls, "_active_fail_commit", False):
            raise sqlite3.OperationalError("simulated commit failure at boundary")
        super()._connection_commit(conn)

    @classmethod
    def _connection_rollback(cls, conn: sqlite3.Connection) -> None:
        if getattr(cls, "_active_noop_rollback", False):
            return
        super()._connection_rollback(conn)

    def _commit_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        pending_invocation_id: str | None = None,
    ) -> None:
        type(self)._active_fail_commit = self.fail_commit
        type(self)._active_noop_rollback = self.noop_rollback
        try:
            super()._commit_transaction(conn, pending_invocation_id=pending_invocation_id)
        finally:
            type(self)._active_fail_commit = False
            type(self)._active_noop_rollback = False


def _open_raw_sqlite(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def test_pis_n14_malformed_invocation_json_fails_closed(
    sqlite_store: SQLiteProviderInvocationStore,
) -> None:
    conn = _open_raw_sqlite(sqlite_store.db_path)
    conn.execute(
        "INSERT INTO provider_invocation_intents (invocation_id, payload) VALUES (?, ?)",
        ("inv-bad-json", "{not-json"),
    )
    conn.commit()
    conn.close()
    with pytest.raises(ProviderInvocationPersistenceError, match="corrupt"):
        sqlite_store.get_invocation("inv-bad-json")


def test_pis_n15_malformed_outcome_json_fails_closed(
    sqlite_store: SQLiteProviderInvocationStore,
) -> None:
    conn = _open_raw_sqlite(sqlite_store.db_path)
    conn.execute(
        "INSERT INTO provider_invocation_outcomes (invocation_id, payload) VALUES (?, ?)",
        ("inv-bad-outcome", "[]"),
    )
    conn.commit()
    conn.close()
    with pytest.raises(ProviderInvocationPersistenceError, match="corrupt|invalid"):
        sqlite_store.get_outcome("inv-bad-outcome")


def test_pis_n16_invalid_typed_invocation_payload_fails_closed(
    sqlite_store: SQLiteProviderInvocationStore,
) -> None:
    conn = _open_raw_sqlite(sqlite_store.db_path)
    conn.execute(
        "INSERT INTO provider_invocation_intents (invocation_id, payload) VALUES (?, ?)",
        (
            "inv-bad-typed",
            '{"schema_version":"provider_invocation.v1","invocation_id":"x"}',
        ),
    )
    conn.commit()
    conn.close()
    with pytest.raises(ProviderInvocationPersistenceError, match="invalid"):
        sqlite_store.get_invocation("inv-bad-typed")


def test_pis_n17_missing_schema_metadata_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "missing-meta.db"
    conn = _open_raw_sqlite(db_path)
    conn.execute(
        """
        CREATE TABLE provider_invocation_store_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """,
    )
    conn.execute(
        """
        CREATE TABLE provider_invocation_intents (
            invocation_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        """,
    )
    conn.execute(
        """
        CREATE TABLE provider_invocation_outcomes (
            invocation_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        """,
    )
    conn.commit()
    conn.close()
    with pytest.raises(ProviderInvocationPersistenceError, match="missing"):
        open_provider_invocation_store_at(db_path)


def test_pis_n18_unsupported_schema_version_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "future-schema.db"
    conn = _open_raw_sqlite(db_path)
    for ddl in (
        """
        CREATE TABLE provider_invocation_store_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE provider_invocation_intents (
            invocation_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE provider_invocation_outcomes (
            invocation_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        """,
    ):
        conn.execute(ddl)
    conn.execute(
        "INSERT INTO provider_invocation_store_meta (key, value) VALUES (?, ?)",
        ("schema_version", "99"),
    )
    conn.commit()
    conn.close()
    with pytest.raises(ProviderInvocationPersistenceError, match="unsupported"):
        open_provider_invocation_store_at(db_path)


def test_pis_n19_incomplete_table_inventory_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "incomplete.db"
    conn = _open_raw_sqlite(db_path)
    conn.execute(
        """
        CREATE TABLE provider_invocation_store_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """,
    )
    conn.execute(
        "INSERT INTO provider_invocation_store_meta (key, value) VALUES (?, ?)",
        ("schema_version", "1"),
    )
    conn.commit()
    conn.close()
    with pytest.raises(ProviderInvocationPersistenceError, match="incomplete"):
        open_provider_invocation_store_at(db_path)


def test_pis_n20_corrupt_read_on_insert_path_preserves_conflict(
    sqlite_store: SQLiteProviderInvocationStore,
) -> None:
    inv = _invocation("inv-insert-decode")
    sqlite_store.put_invocation(inv)
    conn = _open_raw_sqlite(sqlite_store.db_path)
    conn.execute(
        "UPDATE provider_invocation_intents SET payload = ? WHERE invocation_id = ?",
        ("{}", "inv-insert-decode"),
    )
    conn.commit()
    conn.close()
    with pytest.raises(ProviderInvocationPersistenceError):
        sqlite_store.insert_invocation_if_absent(inv)
    other = inv.model_copy(update={"task_id": "task-other"})
    with pytest.raises(ProviderInvocationPersistenceError):
        sqlite_store.insert_invocation_if_absent(other)


def test_pis_n21_commit_boundary_marks_uncertain_when_row_survives(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "uncertain-commit.db"
    store = _CommitBoundaryTestStore(db_path)
    inv = _invocation("inv-survives-commit")
    store.fail_commit = True
    store.noop_rollback = True

    with pytest.raises(ProviderInvocationPersistenceError):
        store.insert_invocation_if_absent(inv)

    store.close()
    reopened = open_provider_invocation_store_at(db_path)
    with pytest.raises(ProviderInvocationPersistenceError, match="uncertain"):
        reopened.get_invocation(inv.invocation_id)
    reopened.close()


def test_pis_n22_no_dispatch_when_store_sealed_after_uncertain_write(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "no-dispatch-uncertain.db"
    store = _CommitBoundaryTestStore(db_path)
    inv = _invocation("inv-no-dispatch")
    store.fail_commit = True
    store.noop_rollback = True

    with pytest.raises(ProviderInvocationPersistenceError):
        store.insert_invocation_if_absent(inv)
    store.close()

    store_reopened = open_provider_invocation_store_at(db_path)
    gate = GovernedProviderInvocationDispatchGate(store=store_reopened, clock=lambda: _T0)
    calls = 0

    def _execute() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(ProviderInvocationPersistenceError, match="uncertain"):
        gate.dispatch_after_intent_persisted(inv, _execute)
    assert calls == 0
    store_reopened.close()
