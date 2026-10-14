# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1-R6-R8 — canonical store transaction safety & nested-lock closure."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from unittest import mock

import pytest

from intergrax.agent_distribution.activation_domain_coordinator import (
    _open_sqlite_admission_domain_lease,
    _sqlite_admission_domain_lease,
)
from intergrax.agent_distribution.serving_execution_admission import (
    ServingExecutionAdmissionError,
)
from intergrax.agent_distribution.sqlite_stores import (
    SqliteAgentDistributionDatabase,
    build_sqlite_agent_distribution_store_bundle,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r7_lease_cleanup_rollback_admission_evidence import (
    _lock_released,
)
from tests.unit.agent_distribution.test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure import (
    _seeded_coordinator,
)
from tests.unit.applications.test_prod_q_r1_r7_d1b_serving_runtime_handoff import (
    _minimal_projection_bundle,
)

pytestmark = [pytest.mark.unit]


def _mutation_raises(db: SqliteAgentDistributionDatabase, operation) -> None:
    with pytest.raises(Exception):
        with db.mutation_transaction() as state:
            operation(state)


def test_mutation_connect_failure_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-connect.db")
    with mock.patch.object(
        bundle.database,
        "_connect",
        side_effect=sqlite3.OperationalError("connect failed"),
    ):
        _mutation_raises(bundle.database, lambda _state: None)
    assert _lock_released(bundle.database)


def test_mutation_begin_failure_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-begin.db")
    real_connect = bundle.database._connect

    class _Conn:
        __slots__ = ("_inner",)

        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, statement, *args, **kwargs):
            if "BEGIN IMMEDIATE" in str(statement):
                raise sqlite3.OperationalError("begin failed")
            return self._inner.execute(statement, *args, **kwargs)

        def rollback(self) -> None:
            self._inner.rollback()

        def close(self) -> None:
            self._inner.close()

    with mock.patch.object(bundle.database, "_connect", side_effect=lambda: _Conn(real_connect())):
        with pytest.raises(ServingExecutionAdmissionError):
            with bundle.database.mutation_transaction():
                pass
    assert _lock_released(bundle.database)


def test_mutation_load_state_failure_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-load.db")
    with mock.patch.object(
        bundle.database,
        "_load_state",
        side_effect=RuntimeError("load failed"),
    ):
        with pytest.raises(RuntimeError, match="load failed"):
            with bundle.database.mutation_transaction():
                pass
    assert _lock_released(bundle.database)


def test_mutation_save_state_failure_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-save.db")
    with mock.patch.object(
        bundle.database,
        "_save_state",
        side_effect=sqlite3.OperationalError("save failed"),
    ):
        with pytest.raises(ServingExecutionAdmissionError):
            with bundle.database.mutation_transaction() as state:
                state.installations.clear()
    assert _lock_released(bundle.database)


def test_mutation_commit_failure_releases_lock(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-commit.db")
    real_connect = bundle.database._connect

    class _Conn:
        __slots__ = ("_inner",)

        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, *args, **kwargs):
            return self._inner.execute(*args, **kwargs)

        def commit(self) -> None:
            raise sqlite3.OperationalError("commit rejected")

        def rollback(self) -> None:
            self._inner.rollback()

        def close(self) -> None:
            self._inner.close()

    with mock.patch.object(bundle.database, "_connect", side_effect=lambda: _Conn(real_connect())):
        with pytest.raises(ServingExecutionAdmissionError, match="mutation failed"):
            with bundle.database.mutation_transaction() as state:
                state.installations.clear()
    assert _lock_released(bundle.database)


def test_mutation_commit_and_rollback_failure_uncertain(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-commit-rb.db")
    real_connect = bundle.database._connect

    class _Conn:
        __slots__ = ("_inner",)

        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, *args, **kwargs):
            return self._inner.execute(*args, **kwargs)

        def commit(self) -> None:
            raise sqlite3.OperationalError("commit rejected")

        def rollback(self) -> None:
            raise sqlite3.OperationalError("rollback failed")

        def close(self) -> None:
            self._inner.close()

    with mock.patch.object(bundle.database, "_connect", side_effect=lambda: _Conn(real_connect())):
        with pytest.raises(
            ServingExecutionAdmissionError,
            match="transaction outcome uncertain",
        ):
            with bundle.database.mutation_transaction() as state:
                state.installations.clear()
    assert _lock_released(bundle.database)


def test_mutation_close_failure_after_successful_commit(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-close.db")
    real_connect = bundle.database._connect

    class _Conn:
        __slots__ = ("_inner",)

        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, *args, **kwargs):
            return self._inner.execute(*args, **kwargs)

        def commit(self) -> None:
            self._inner.commit()

        def rollback(self) -> None:
            self._inner.rollback()

        def close(self) -> None:
            raise sqlite3.OperationalError("close failed")

    with mock.patch.object(bundle.database, "_connect", side_effect=lambda: _Conn(real_connect())):
        with pytest.raises(
            ServingExecutionAdmissionError,
            match="connection close failed",
        ):
            with bundle.database.mutation_transaction() as state:
                state.installations.clear()
    assert _lock_released(bundle.database)


def test_nested_same_thread_store_mutation_rejected(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-nested.db")
    with pytest.raises(ServingExecutionAdmissionError, match="nested activation domain lock"):
        with bundle.database.mutation_transaction():
            with bundle.database.mutation_transaction():
                pass
    assert _lock_released(bundle.database)


def test_nested_lease_and_store_mutate_fails_closed(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-lease-nested.db")
    with pytest.raises(ServingExecutionAdmissionError, match="nested activation domain lock"):
        with _sqlite_admission_domain_lease(bundle.database):
            with bundle.database.mutation_transaction():
                pass
    assert _lock_released(bundle.database)


def test_concurrent_activation_domain_lock_serializes(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-concurrent.db")
    second_finished = threading.Event()

    def _second() -> None:
        lease = _open_sqlite_admission_domain_lease(bundle.database)
        lease.rollback()
        lease.close()
        second_finished.set()

    with _sqlite_admission_domain_lease(bundle.database) as lease:
        worker = threading.Thread(target=_second)
        worker.start()
        assert not second_finished.wait(timeout=0.2)
        lease.rollback()
    worker.join(timeout=5.0)
    assert second_finished.is_set()


def test_activation_domain_lock_released_exactly_once(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-once.db")
    with bundle.database.mutation_transaction():
        pass
    assert _lock_released(bundle.database)
    with bundle.database.mutation_transaction():
        pass
    assert _lock_released(bundle.database)


def test_healthy_mutation_after_failed_acquisition(tmp_path: Path) -> None:
    bundle = build_sqlite_agent_distribution_store_bundle(tmp_path / "r8-recover.db")
    with mock.patch.object(
        bundle.database,
        "_load_state",
        side_effect=sqlite3.OperationalError("first failure"),
    ):
        with pytest.raises(ServingExecutionAdmissionError):
            with bundle.database.mutation_transaction():
                pass
    with bundle.database.mutation_transaction():
        pass
    assert _lock_released(bundle.database)


def test_d1b_reconstructed_adapter_gate_closed_until_coordinated_open(tmp_path: Path) -> None:
    adapter, unit_ref, _digest, revision, *_rest = _seeded_coordinator(tmp_path)
    adapter.stop(revision, serving_unit_ref=unit_ref)
    with pytest.raises(Exception):
        adapter.reserve_unit_execution(unit_ref)
    rebuilt_root = tmp_path / "rebuilt"
    rebuilt_root.mkdir(parents=True, exist_ok=True)
    fresh_adapter, _accessor, fresh_unit, *_ = _minimal_projection_bundle(rebuilt_root)
    assert fresh_adapter._units[fresh_unit].serving_admission_authorized is False
    with pytest.raises(Exception):
        fresh_adapter.reserve_unit_execution(fresh_unit)
