# © Artur Czarnecki. All rights reserved.

"""AW-4A — PostgreSQL wake-up receipt integration tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.autonomous_work.postgresql_repository import (
    AutonomousWorkSchemaVersionError,
    PostgreSQLAutonomousWorkStore,
)
from intergrax.autonomous_work.wake_up_service import WorkerWakeUpService
from intergrax.contracts.autonomous_work import (
    WorkerLifecycleState,
    WorkerWakeUpDisposition,
    WorkerWakeUpSignal,
    WorkerWakeUpSourceKind,
    mint_wake_up_id,
)
from intergrax.contracts.autonomous_work.references import WakeUpSourceRef
from tests.integration.autonomous_work.conftest import open_bundle, resolve_postgresql_config
from tests.unit.autonomous_work import repository_contracts as contract_suite
from tests.unit.autonomous_work import wake_up_receipt_repository_contracts as receipt_contracts
from tests.unit.autonomous_work.test_wake_up_service import _FixedClock

pytestmark = [pytest.mark.integration, pytest.mark.network]


def test_postgresql_fresh_database_bootstraps_schema_v3(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    with postgresql_autonomous_work_bundle.store.transaction() as conn:
        row = conn.execute(
            "SELECT schema_version FROM autonomous_work_schema_meta WHERE id = 1"
        ).fetchone()
        assert row is not None
        assert int(row["schema_version"]) == 3


def test_postgresql_repeated_v3_bootstrap_is_idempotent(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    postgresql_autonomous_work_bundle.close()
    reopened = open_bundle(schema_name)
    try:
        with reopened.store.transaction() as conn:
            row = conn.execute(
                "SELECT schema_version FROM autonomous_work_schema_meta WHERE id = 1"
            ).fetchone()
            assert row is not None
            assert int(row["schema_version"]) == 3
    finally:
        reopened.close()


def test_postgresql_wake_up_receipt_contract_suite(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    bundle = open_bundle(schema_name)
    try:
        receipt_contracts.run_wake_up_receipt_contract_suite(
            lambda: bundle.worker_wake_up_receipt
        )
    finally:
        bundle.close()


def test_postgresql_wake_up_receipt_conflicting_source_kind(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    bundle = open_bundle(schema_name)
    try:
        receipt_contracts.test_wake_up_receipt_conflicting_source_kind(
            lambda: bundle.worker_wake_up_receipt
        )
    finally:
        bundle.close()


def test_postgresql_wake_up_receipt_concurrent_duplicate(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    bundle = open_bundle(schema_name)
    try:
        receipt_contracts.test_wake_up_receipt_concurrent_duplicate_one_wins(
            lambda: bundle.worker_wake_up_receipt
        )
    finally:
        bundle.close()


def test_postgresql_wake_up_receipt_concurrent_conflicting(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    bundle = open_bundle(schema_name)
    try:
        receipt_contracts.test_wake_up_receipt_concurrent_conflicting_one_claimed_one_conflict(
            lambda: bundle.worker_wake_up_receipt
        )
    finally:
        bundle.close()


def test_postgresql_schema_v2_to_v3_migration_preserves_existing_data(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    created = postgresql_autonomous_work_bundle.worker_instance.create(
        contract_suite.worker_instance()
    )
    binding = postgresql_autonomous_work_bundle.worker_principal_binding.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=created.worker_instance_id,
        )
    )
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    with store.transaction() as conn:
        conn.execute("DROP TABLE IF EXISTS aw_worker_wake_up_receipts")
        conn.execute(
            "UPDATE autonomous_work_schema_meta SET schema_version = %s WHERE id = 1",
            (2,),
        )

    migrated_bundle = open_bundle(schema_name)
    try:
        loaded = migrated_bundle.worker_instance.get(
            worker_instance_id=created.worker_instance_id
        )
        assert loaded == created
        loaded_binding = migrated_bundle.worker_principal_binding.get(
            worker_instance_id=created.worker_instance_id
        )
        assert loaded_binding == binding
        with migrated_bundle.store.transaction() as conn:
            row = conn.execute(
                "SELECT schema_version FROM autonomous_work_schema_meta WHERE id = 1"
            ).fetchone()
            assert row is not None
            assert int(row["schema_version"]) == 3
            table_row = conn.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'aw_worker_wake_up_receipts'
                """
            ).fetchone()
            assert table_row is not None
    finally:
        migrated_bundle.close()


def test_postgresql_schema_v1_to_latest_preserves_existing_data(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    created = postgresql_autonomous_work_bundle.worker_instance.create(
        contract_suite.worker_instance()
    )
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    with store.transaction() as conn:
        conn.execute("DROP TABLE IF EXISTS aw_worker_wake_up_receipts")
        conn.execute("DROP TABLE IF EXISTS aw_worker_principal_bindings")
        conn.execute(
            "UPDATE autonomous_work_schema_meta SET schema_version = %s WHERE id = 1",
            (1,),
        )

    migrated_bundle = open_bundle(schema_name)
    try:
        loaded = migrated_bundle.worker_instance.get(
            worker_instance_id=created.worker_instance_id
        )
        assert loaded == created
        binding = contract_suite.worker_principal_binding(
            worker_instance_id=created.worker_instance_id,
        )
        persisted = migrated_bundle.worker_principal_binding.create(binding)
        assert persisted == binding
        with migrated_bundle.store.transaction() as conn:
            row = conn.execute(
                "SELECT schema_version FROM autonomous_work_schema_meta WHERE id = 1"
            ).fetchone()
            assert row is not None
            assert int(row["schema_version"]) == 3
    finally:
        migrated_bundle.close()


def test_postgresql_migration_v2_to_v3_atomicity_on_failure(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from intergrax.integrations.providers.relational_store.postgresql.session import (
        PostgreSQLConnectionProvider,
    )

    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    with store.transaction() as conn:
        conn.execute("DROP TABLE IF EXISTS aw_worker_wake_up_receipts")
        conn.execute(
            "UPDATE autonomous_work_schema_meta SET schema_version = %s WHERE id = 1",
            (2,),
        )

    def failing_migration(self, session):  # type: ignore[no-untyped-def]
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS aw_worker_wake_up_receipts (
                worker_instance_id TEXT NOT NULL,
                wake_up_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                accepted_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (worker_instance_id, wake_up_id)
            );
            """
        )
        raise AutonomousWorkSchemaVersionError("controlled migration failure")

    monkeypatch.setattr(PostgreSQLAutonomousWorkStore, "_migrate_v2_to_v3", failing_migration)
    with pytest.raises(AutonomousWorkSchemaVersionError, match="controlled migration failure"):
        open_bundle(schema_name)

    config = resolve_postgresql_config()
    assert config is not None
    provider = PostgreSQLConnectionProvider(config)
    with provider.connection() as conn:
        row = conn.execute(
            f"SELECT schema_version FROM {schema_name}.autonomous_work_schema_meta WHERE id = 1"
        ).fetchone()
        assert row is not None
        assert int(row["schema_version"]) == 2
        table_row = conn.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = 'aw_worker_wake_up_receipts'
            """,
            (schema_name,),
        ).fetchone()
        assert table_row is None

    monkeypatch.undo()
    finalized = open_bundle(schema_name)
    finalized.close()


def test_postgresql_newer_schema_version_fails_closed(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    store = postgresql_autonomous_work_bundle.store
    with store.transaction() as conn:
        conn.execute(
            "UPDATE autonomous_work_schema_meta SET schema_version = %s WHERE id = 1",
            (4,),
        )
    schema_name = store.schema_name
    postgresql_autonomous_work_bundle.close()
    with pytest.raises(AutonomousWorkSchemaVersionError):
        open_bundle(schema_name)


def test_postgresql_wake_up_restart_duplicate(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    worker = postgresql_autonomous_work_bundle.worker_instance.create(
        contract_suite.worker_instance(lifecycle_state=WorkerLifecycleState.IDLE)
    )
    wake_id = mint_wake_up_id()
    signal = WorkerWakeUpSignal(
        wake_up_id=wake_id,
        worker_instance_id=worker.worker_instance_id,
        source_kind=WorkerWakeUpSourceKind.EXTERNAL_EVENT,
        source_ref=WakeUpSourceRef("source/external/restart"),
        occurred_at=datetime(2026, 9, 3, 11, 59, tzinfo=UTC),
        delivery_identity=wake_id,
        correlation_ref=None,
    )
    service = WorkerWakeUpService(
        worker_instance_repository=postgresql_autonomous_work_bundle.worker_instance,
        continuity_state_repository=postgresql_autonomous_work_bundle.work_continuity_state,
        wake_up_receipt_repository=postgresql_autonomous_work_bundle.worker_wake_up_receipt,
        clock=_FixedClock(datetime(2026, 9, 3, 12, 0, tzinfo=UTC)),
    )
    assert service.accept(signal).disposition == WorkerWakeUpDisposition.ACCEPTED
    postgresql_autonomous_work_bundle.close()

    reopened = open_bundle(schema_name)
    try:
        replay_service = WorkerWakeUpService(
            worker_instance_repository=reopened.worker_instance,
            continuity_state_repository=reopened.work_continuity_state,
            wake_up_receipt_repository=reopened.worker_wake_up_receipt,
            clock=_FixedClock(datetime(2026, 9, 3, 12, 1, tzinfo=UTC)),
        )
        assert replay_service.accept(signal).disposition == WorkerWakeUpDisposition.DUPLICATE
    finally:
        reopened.close()


def test_postgresql_wake_up_restart_conflicting(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    worker = postgresql_autonomous_work_bundle.worker_instance.create(
        contract_suite.worker_instance(lifecycle_state=WorkerLifecycleState.IDLE)
    )
    wake_id = mint_wake_up_id()
    accepted_signal = WorkerWakeUpSignal(
        wake_up_id=wake_id,
        worker_instance_id=worker.worker_instance_id,
        source_kind=WorkerWakeUpSourceKind.QUEUE_DELIVERY,
        source_ref=WakeUpSourceRef("queue/order-123"),
        occurred_at=datetime(2026, 9, 3, 11, 59, tzinfo=UTC),
        delivery_identity=mint_wake_up_id(),
        correlation_ref=None,
    )
    conflicting_signal = WorkerWakeUpSignal(
        wake_up_id=wake_id,
        worker_instance_id=worker.worker_instance_id,
        source_kind=WorkerWakeUpSourceKind.OPERATOR,
        source_ref=WakeUpSourceRef("operator/manual"),
        occurred_at=datetime(2026, 9, 3, 11, 59, tzinfo=UTC),
        delivery_identity=mint_wake_up_id(),
        correlation_ref=None,
    )
    service = WorkerWakeUpService(
        worker_instance_repository=postgresql_autonomous_work_bundle.worker_instance,
        continuity_state_repository=postgresql_autonomous_work_bundle.work_continuity_state,
        wake_up_receipt_repository=postgresql_autonomous_work_bundle.worker_wake_up_receipt,
        clock=_FixedClock(datetime(2026, 9, 3, 12, 0, tzinfo=UTC)),
    )
    assert service.accept(accepted_signal).disposition == WorkerWakeUpDisposition.ACCEPTED
    postgresql_autonomous_work_bundle.close()

    reopened = open_bundle(schema_name)
    try:
        replay_service = WorkerWakeUpService(
            worker_instance_repository=reopened.worker_instance,
            continuity_state_repository=reopened.work_continuity_state,
            wake_up_receipt_repository=reopened.worker_wake_up_receipt,
            clock=_FixedClock(datetime(2026, 9, 3, 12, 1, tzinfo=UTC)),
        )
        assert (
            replay_service.accept(conflicting_signal).disposition
            == WorkerWakeUpDisposition.CONFLICT
        )
    finally:
        reopened.close()


def test_postgresql_wake_up_cross_connection_duplicate(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    bundle_a = postgresql_autonomous_work_bundle
    bundle_b = open_bundle(bundle_a.store.schema_name)
    receipt = receipt_contracts.wake_up_receipt()
    try:
        from intergrax.autonomous_work.repository import WorkerWakeUpReceiptClaimStatus

        first = bundle_a.worker_wake_up_receipt.claim(receipt)
        second = bundle_b.worker_wake_up_receipt.claim(receipt)
        assert first.status is WorkerWakeUpReceiptClaimStatus.CLAIMED
        assert second.status is WorkerWakeUpReceiptClaimStatus.DUPLICATE
        assert second.receipt == first.receipt
    finally:
        bundle_b.close()
