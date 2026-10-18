# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-IMPL-01 — SQLite GCA durable host bundle and partial-finalization tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import pytest

from governed_contractor_application.host.governed_external_work_host_stores import (
    GovernedExternalWorkHostStoreBundle,
    validate_production_governed_external_work_host_store_bundle,
)
from governed_contractor_application.host.lifecycle_states import (
    GovernedExternalWorkHostState,
)
from governed_contractor_application.host.sqlite_governed_external_work_host_store_bundle import (
    GovernedExternalWorkHostPersistenceError,
    open_sqlite_governed_external_work_host_store_bundle_at,
)
from governed_contractor_application.host.stores import GovernedExecutionCreateFollowOnConflictError
from intergrax.contracts.governed_external_work_create_follow_on import (
    GovernedExternalWorkCreateFollowOnRecordV1,
)
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle

pytestmark = [pytest.mark.unit, pytest.mark.gate]

# NOT PROVEN at authoring time — resource budget pytest=0; execute only after operator approval:
# uv run pytest -p no:xdist tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_gca_sqlite_durable_recovery.py::test_sqlite_ger_round_trip_close_reopen_new_backend -q
# uv run pytest -p no:xdist tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_gca_sqlite_durable_recovery.py::test_f02_persist_round_trip_close_reopen -q
# uv run pytest -p no:xdist tests/unit/applications/governed_contractor/test_prod_q_r1_r7_q_gca_sqlite_durable_recovery.py::test_sqlite_host_state_round_trip_close_reopen_new_backend -q


def _sample_bundle_record(execution_id: str) -> GovernedExternalWorkCreateFollowOnRecordV1:
    from intergrax.contracts.external_work import (
        ExternalTaskCorrelation,
        ExternalWorkSnapshot,
        ExternalWorkStatus,
    )

    now = datetime.now(UTC)
    snapshot = ExternalWorkSnapshot(
        correlation=ExternalTaskCorrelation(
            task_id="task-1",
            provider_id="prov",
            external_task_id="ew-1",
        ),
        status=ExternalWorkStatus.QUOTE_AVAILABLE,
        created_at=now,
        updated_at=now,
    )
    return GovernedExternalWorkCreateFollowOnRecordV1(
        execution_id=execution_id,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-a",
        principal_id="principal-a",
        create_action="external_work.create",
        create_provider_invocation_id="inv-1",
        snapshot=snapshot,
        provider="prov",
        snapshot_content_digest="digest",
        recorded_at=datetime.now(UTC),
    )


def test_sqlite_bundle_reports_durable_and_validates_for_production(
    tmp_path: Path,
) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    validate_production_governed_external_work_host_store_bundle(backend.bundle)
    assert backend.bundle.is_durable is True
    backend.close()


def test_sqlite_host_state_round_trip_close_reopen_new_backend(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    execution_id = "exec-state-1"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.bundle.execution_store.put_state(
        execution_id,
        GovernedExternalWorkHostState.HOST_FINALIZATION_INCOMPLETE,
    )
    backend.close()
    reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
    assert (
        reopened.bundle.execution_store.get_state(execution_id)
        is GovernedExternalWorkHostState.HOST_FINALIZATION_INCOMPLETE
    )
    reopened.close()


def test_f02_persist_round_trip_close_reopen(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    execution_id = "exec-f02-1"
    record = _sample_bundle_record(execution_id)
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.bundle.execution_store.put_create_follow_on_record(execution_id, record)
    backend.close()
    reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
    loaded = reopened.bundle.execution_store.get_create_follow_on_record(execution_id)
    assert loaded is not None
    assert str(loaded.execution_id) == execution_id
    reopened.close()


def test_f02_conflict_raises(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    execution_id = "exec-conflict"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    store = backend.bundle.execution_store
    first = _sample_bundle_record(execution_id)
    second = first.model_copy(update={"snapshot_content_digest": "other"})
    store.put_create_follow_on_record(execution_id, first)
    with pytest.raises(GovernedExecutionCreateFollowOnConflictError):
        store.put_create_follow_on_record(execution_id, second)
    backend.close()


def test_corrupt_f02_fail_closed(tmp_path: Path) -> None:
    db = tmp_path / "gca.db"
    backend = open_sqlite_governed_external_work_host_store_bundle_at(db)
    backend.close()
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO gca_create_follow_on (execution_id, payload) VALUES (?, ?)",
        ("exec-bad", "not-json"),
    )
    conn.commit()
    conn.close()
    reopened = open_sqlite_governed_external_work_host_store_bundle_at(db)
    with pytest.raises(ValueError, match="corrupted_create_follow_on"):
        reopened.bundle.execution_store.get_create_follow_on_record("exec-bad")
    reopened.close()


def test_closed_sqlite_backend_rejects_writes(tmp_path: Path) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    backend.close()
    with pytest.raises(GovernedExternalWorkHostPersistenceError):
        backend.bundle.execution_store.put_state(
            "x",
            GovernedExternalWorkHostState.REQUESTED,
        )


def test_policy_bundle_round_trip(tmp_path: Path) -> None:
    backend = open_sqlite_governed_external_work_host_store_bundle_at(
        tmp_path / "gca.db",
    )
    bundle = ImmutableRuntimePolicyBundle(
        bundle_id="b1",
        version="1",
        rules=(),
    )
    backend.bundle.bundle_store.put_bundle(bundle)
    loaded = backend.bundle.bundle_store.get_bundle("b1", "1")
    assert loaded is not None
    assert loaded.bundle_id == "b1"
    backend.close()
