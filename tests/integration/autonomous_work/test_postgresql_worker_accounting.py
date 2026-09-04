# © Artur Czarnecki. All rights reserved.

"""AW-5B — PostgreSQL worker accounting qualification (fresh schema, migration, concurrency)."""

from __future__ import annotations

import multiprocessing
from datetime import UTC, datetime, timedelta

import pytest

import intergrax.autonomous_work.postgresql_repository as pg_repo_module
from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.autonomous_work.postgresql_repository import (
    AutonomousWorkSchemaVersionError,
    PostgreSQLAutonomousWorkStore,
)
from intergrax.autonomous_work.worker_accounting_windows import (
    monthly_window_bounds,
    worker_accounting_window,
)
from intergrax.autonomous_work.repository import WorkerWakeUpReceiptClaimStatus
from intergrax.autonomous_work.worker_budget_ports import WorkerAccountingConflict
from intergrax.contracts.autonomous_work.execution_dispatch import WorkerExecutionSourceKind
from intergrax.contracts.autonomous_work.profile_reference import (
    BudgetProfileRef,
    ProfileVersion,
)
from intergrax.contracts.autonomous_work.worker_budget_accounting import (
    WorkerAccountingWindowKind,
    WorkerBudgetAdmissionDisposition,
    WorkerBudgetAdmissionReason,
    WorkerBudgetPolicy,
    WorkerBudgetReserveRequest,
    WorkerLogicalDispatchRef,
)
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
)
from intergrax.runtime.execution.budget.models import BudgetUsageTotals
from tests.integration.autonomous_work.conftest import (
    materialization_options_for_schema,
    materialize_bundle,
    open_bundle,
    open_bundle_with_options,
    resolve_postgresql_config,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite
from tests.unit.autonomous_work.wake_up_receipt_repository_contracts import wake_up_receipt

pytestmark = [pytest.mark.integration, pytest.mark.network]

_UTC = UTC
_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=_UTC)
_PROFILE_V1 = BudgetProfileRef(profile_id="budget/test", version=ProfileVersion(1))
_PROFILE_V2 = BudgetProfileRef(profile_id="budget/test", version=ProfileVersion(2))
_CURRENT_SCHEMA_VERSION = pg_repo_module._SCHEMA_VERSION
_ACCOUNTING_TABLE = "aw_worker_accounting_snapshots"


def _logical_dispatch(
    *,
    worker_id: str,
    source_ref: str,
    source_kind: WorkerExecutionSourceKind = WorkerExecutionSourceKind.OPERATOR,
) -> WorkerLogicalDispatchRef:
    return WorkerLogicalDispatchRef(
        worker_instance_id=worker_id,
        source_kind=source_kind,
        source_ref=source_ref,
    )


def _reserve_request(
    *,
    worker_id: str,
    source_ref: str,
    policy: WorkerBudgetPolicy,
    profile_ref: BudgetProfileRef = _PROFILE_V1,
    reserved_at: datetime = _NOW,
) -> WorkerBudgetReserveRequest:
    return WorkerBudgetReserveRequest(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref=source_ref),
        budget_profile_ref=profile_ref,
        policy=policy,
        source_kind=WorkerExecutionSourceKind.OPERATOR,
        reserved_at=reserved_at,
    )


def _schema_version(store: PostgreSQLAutonomousWorkStore) -> int:
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT schema_version FROM autonomous_work_schema_meta WHERE id = 1"
        ).fetchone()
        assert row is not None
        return int(row["schema_version"])


def _table_exists(store: PostgreSQLAutonomousWorkStore, table_name: str) -> bool:
    with store.transaction() as conn:
        row = conn.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = current_schema() AND table_name = %s
            """,
            (table_name,),
        ).fetchone()
        return row is not None


def _downgrade_schema(
    store: PostgreSQLAutonomousWorkStore,
    *,
    target_version: int,
) -> None:
    tables_by_version = {
        4: [_ACCOUNTING_TABLE],
        3: [_ACCOUNTING_TABLE, "aw_goal_evaluation_cadence_states"],
        2: [
            _ACCOUNTING_TABLE,
            "aw_goal_evaluation_cadence_states",
            "aw_worker_wake_up_receipts",
        ],
        1: [
            _ACCOUNTING_TABLE,
            "aw_goal_evaluation_cadence_states",
            "aw_worker_wake_up_receipts",
            "aw_worker_principal_bindings",
        ],
    }
    with store.transaction() as conn:
        for table_name in tables_by_version.get(target_version, []):
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            "UPDATE autonomous_work_schema_meta SET schema_version = %s WHERE id = 1",
            (target_version,),
        )


def _seed_v4_domain_data(bundle: AutonomousWorkRepositories) -> dict[str, object]:
    created = bundle.worker_instance.create(contract_suite.worker_instance())
    binding = bundle.worker_principal_binding.create(
        contract_suite.worker_principal_binding(
            worker_instance_id=created.worker_instance_id,
        )
    )
    receipt = wake_up_receipt(worker_instance_id=created.worker_instance_id)
    claim = bundle.worker_wake_up_receipt.claim(receipt)
    assert claim.status is WorkerWakeUpReceiptClaimStatus.CLAIMED
    cadence = bundle.goal_evaluation_cadence_state.record_evaluated(
        goal_id=contract_suite.worker_goal().goal_id,
        evaluated_at=_NOW,
    )
    return {
        "worker": created,
        "binding": binding,
        "receipt": receipt,
        "cadence": cadence,
    }


def test_postgresql_fresh_schema_v5_includes_accounting_table_and_reserve(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    assert _schema_version(store) == _CURRENT_SCHEMA_VERSION
    assert _table_exists(store, _ACCOUNTING_TABLE)

    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(max_concurrent_executions=2)
    result = postgresql_autonomous_work_bundle.worker_accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="fresh-proof", policy=policy)
    )
    assert result.disposition is WorkerBudgetAdmissionDisposition.ALLOWED


def test_postgresql_v4_to_v5_migration_preserves_existing_data(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    seeded = _seed_v4_domain_data(postgresql_autonomous_work_bundle)
    created = seeded["worker"]
    binding = seeded["binding"]
    receipt = seeded["receipt"]
    cadence = seeded["cadence"]
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    _downgrade_schema(store, target_version=4)
    assert not _table_exists(store, _ACCOUNTING_TABLE)

    migrated_bundle = open_bundle(schema_name)
    try:
        assert _schema_version(migrated_bundle.store) == _CURRENT_SCHEMA_VERSION  # type: ignore[arg-type]
        assert _table_exists(migrated_bundle.store, _ACCOUNTING_TABLE)  # type: ignore[arg-type]
        loaded = migrated_bundle.worker_instance.get(
            worker_instance_id=created.worker_instance_id  # type: ignore[union-attr]
        )
        assert loaded == created
        loaded_binding = migrated_bundle.worker_principal_binding.get(
            worker_instance_id=created.worker_instance_id  # type: ignore[union-attr]
        )
        assert loaded_binding == binding
        loaded_receipt = migrated_bundle.worker_wake_up_receipt.get(
            worker_instance_id=created.worker_instance_id,  # type: ignore[union-attr]
            wake_up_id=receipt.wake_up_id,  # type: ignore[union-attr]
        )
        assert loaded_receipt == receipt
        loaded_cadence = migrated_bundle.goal_evaluation_cadence_state.get(
            goal_id=cadence.goal_id  # type: ignore[union-attr]
        )
        assert loaded_cadence == cadence
        worker_id = created.worker_instance_id  # type: ignore[union-attr]
        result = migrated_bundle.worker_accounting.reserve(
            _reserve_request(worker_id=worker_id, source_ref="post-migrate", policy=WorkerBudgetPolicy())
        )
        assert result.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    finally:
        migrated_bundle.close()


def test_postgresql_migration_v4_to_v5_atomicity_on_failure(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = _seed_v4_domain_data(postgresql_autonomous_work_bundle)
    created = seeded["worker"]
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    _downgrade_schema(store, target_version=4)

    def failing_migration(self, session):  # type: ignore[no-untyped-def]
        session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_ACCOUNTING_TABLE} (
                worker_instance_id TEXT NOT NULL PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                revision INTEGER NOT NULL
            );
            """
        )
        raise AutonomousWorkSchemaVersionError("controlled migration failure")

    monkeypatch.setattr(PostgreSQLAutonomousWorkStore, "_migrate_v4_to_v5", failing_migration)
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
        assert int(row["schema_version"]) == 4
        table_row = conn.execute(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema_name, _ACCOUNTING_TABLE),
        ).fetchone()
        assert table_row is None
        worker_row = conn.execute(
            f"""
            SELECT record_json FROM {schema_name}.aw_worker_instances
            WHERE worker_instance_id = %s
            """,
            (created.worker_instance_id.strip(),),  # type: ignore[union-attr]
        ).fetchone()
        assert worker_row is not None

    monkeypatch.undo()
    migrated_bundle = open_bundle(schema_name)
    try:
        loaded = migrated_bundle.worker_instance.get(
            worker_instance_id=created.worker_instance_id  # type: ignore[union-attr]
        )
        assert loaded == created
        assert _schema_version(migrated_bundle.store) == _CURRENT_SCHEMA_VERSION  # type: ignore[arg-type]
        assert _table_exists(migrated_bundle.store, _ACCOUNTING_TABLE)  # type: ignore[arg-type]
    finally:
        migrated_bundle.close()


@pytest.mark.parametrize("from_version", [1, 2, 3, 4])
def test_postgresql_migration_chain_reaches_current_version(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
    from_version: int,
) -> None:
    created = postgresql_autonomous_work_bundle.worker_instance.create(
        contract_suite.worker_instance()
    )
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    _downgrade_schema(store, target_version=from_version)

    migrated_bundle = open_bundle(schema_name)
    try:
        loaded = migrated_bundle.worker_instance.get(
            worker_instance_id=created.worker_instance_id
        )
        assert loaded == created
        assert _schema_version(migrated_bundle.store) == _CURRENT_SCHEMA_VERSION  # type: ignore[arg-type]
        assert _table_exists(migrated_bundle.store, _ACCOUNTING_TABLE)  # type: ignore[arg-type]
    finally:
        migrated_bundle.close()


def test_postgresql_schema_current_to_current_is_idempotent(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    assert _schema_version(store) == _CURRENT_SCHEMA_VERSION
    reopened = open_bundle(schema_name)
    try:
        assert _schema_version(reopened.store) == _CURRENT_SCHEMA_VERSION  # type: ignore[arg-type]
        assert _table_exists(reopened.store, _ACCOUNTING_TABLE)  # type: ignore[arg-type]
    finally:
        reopened.close()


def test_postgresql_unsupported_schema_version_v6_fails_closed(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    with postgresql_autonomous_work_bundle.store.transaction() as conn:
        conn.execute(
            "UPDATE autonomous_work_schema_meta SET schema_version = %s WHERE id = 1",
            (6,),
        )
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    options = materialization_options_for_schema(schema_name)
    with pytest.raises(AutonomousWorkSchemaVersionError):
        materialize_bundle(options)


def _multiprocess_reserve_attempt(
    dsn: str,
    schema_name: str,
    worker_id: str,
    source_ref: str,
    barrier: multiprocessing.Barrier,
    result_queue: multiprocessing.Queue[tuple[str, str, str | None]],
) -> None:
    bundle = open_bundle_with_options({"dsn": dsn, "schema_name": schema_name})
    try:
        barrier.wait(timeout=30)
        policy = WorkerBudgetPolicy(max_concurrent_executions=1)
        result = bundle.worker_accounting.reserve(
            _reserve_request(worker_id=worker_id, source_ref=source_ref, policy=policy)
        )
        reason = result.evidence.reason.value if result.evidence and result.evidence.reason else None
        result_queue.put((source_ref, result.disposition.value, reason))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put((source_ref, "ERROR", type(exc).__name__))
    finally:
        bundle.close()


def test_postgresql_multiprocess_concurrency_limit_one_allowed_one_denied(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    config = resolve_postgresql_config()
    assert config is not None
    dsn = config.connection_string()
    assert dsn
    worker_id = contract_suite.mint_worker_instance_id()
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    barrier = multiprocessing.Barrier(2)
    result_queue: multiprocessing.Queue[tuple[str, str, str | None]] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_multiprocess_reserve_attempt,
            args=(dsn, schema_name, worker_id, source_ref, barrier, result_queue),
        )
        for source_ref in ("proc-a", "proc-b")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    outcomes = [result_queue.get(timeout=30) for _ in processes]
    allowed = [item for item in outcomes if item[1] == WorkerBudgetAdmissionDisposition.ALLOWED.value]
    denied = [
        item
        for item in outcomes
        if item[1] == WorkerBudgetAdmissionDisposition.DENIED.value
        and item[2] == WorkerBudgetAdmissionReason.CONCURRENCY_LIMIT_EXCEEDED.value
    ]
    assert len(allowed) == 1
    assert len(denied) == 1


def _multiprocess_daily_limit_attempt(
    dsn: str,
    schema_name: str,
    worker_id: str,
    source_ref: str,
    barrier: multiprocessing.Barrier,
    result_queue: multiprocessing.Queue[tuple[str, str, str | None]],
) -> None:
    bundle = open_bundle_with_options({"dsn": dsn, "schema_name": schema_name})
    try:
        barrier.wait(timeout=30)
        policy = WorkerBudgetPolicy(daily_execution_limit=1, max_concurrent_executions=2)
        reserve = bundle.worker_accounting.reserve(
            _reserve_request(worker_id=worker_id, source_ref=source_ref, policy=policy)
        )
        if reserve.disposition is not WorkerBudgetAdmissionDisposition.ALLOWED:
            reason = (
                reserve.evidence.reason.value
                if reserve.evidence and reserve.evidence.reason
                else None
            )
            result_queue.put((source_ref, reserve.disposition.value, reason))
            return
        execution_id = mint_execution_id()
        bundle.worker_accounting.bind_execution(
            logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref=source_ref),
            execution_id=execution_id,
            bound_at=_NOW,
        )
        result_queue.put((source_ref, reserve.disposition.value, None))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put((source_ref, "ERROR", type(exc).__name__))
    finally:
        bundle.close()


def test_postgresql_cross_connection_daily_limit_no_lost_update(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    config = resolve_postgresql_config()
    assert config is not None
    dsn = config.connection_string()
    assert dsn
    worker_id = contract_suite.mint_worker_instance_id()
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    barrier = multiprocessing.Barrier(2)
    result_queue: multiprocessing.Queue[tuple[str, str, str | None]] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_multiprocess_daily_limit_attempt,
            args=(dsn, schema_name, worker_id, source_ref, barrier, result_queue),
        )
        for source_ref in ("daily-a", "daily-b")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    outcomes = [result_queue.get(timeout=30) for _ in processes]
    allowed = [item for item in outcomes if item[1] == WorkerBudgetAdmissionDisposition.ALLOWED.value]
    denied = [
        item
        for item in outcomes
        if item[1] == WorkerBudgetAdmissionDisposition.DENIED.value
        and item[2] == WorkerBudgetAdmissionReason.DAILY_LIMIT_EXCEEDED.value
    ]
    assert len(allowed) == 1
    assert len(denied) == 1


def test_postgresql_restart_active_reservation_and_terminal_release(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(max_concurrent_executions=1, daily_execution_limit=5)
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    accounting = postgresql_autonomous_work_bundle.worker_accounting
    reserve = accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="restart-active", policy=policy)
    )
    assert reserve.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="restart-active"),
        execution_id=execution_id,
        bound_at=_NOW,
    )
    postgresql_autonomous_work_bundle.close()

    reopened = open_bundle(schema_name)
    try:
        denied = reopened.worker_accounting.reserve(
            _reserve_request(worker_id=worker_id, source_ref="after-restart", policy=policy)
        )
        assert denied.disposition is WorkerBudgetAdmissionDisposition.DENIED
        assert denied.evidence.reason is WorkerBudgetAdmissionReason.CONCURRENCY_LIMIT_EXCEEDED
        reopened.worker_accounting.release_execution(
            worker_instance_id=worker_id,
            execution_id=execution_id,
            released_at=_NOW + timedelta(seconds=1),
        )
    finally:
        reopened.close()

    final = open_bundle(schema_name)
    try:
        allowed = final.worker_accounting.reserve(
            _reserve_request(worker_id=worker_id, source_ref="after-release", policy=policy)
        )
        assert allowed.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    finally:
        final.close()


def test_postgresql_restart_preserves_daily_and_monthly_counters(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(
        daily_execution_limit=10,
        monthly_execution_limit=10,
        max_concurrent_executions=5,
    )
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    accounting = postgresql_autonomous_work_bundle.worker_accounting
    accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="persist-counts", policy=policy)
    )
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="persist-counts"),
        execution_id=execution_id,
        bound_at=_NOW,
    )
    accounting.release_execution(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        released_at=_NOW,
    )
    daily_window = worker_accounting_window(
        worker_instance_id=worker_id,
        window_kind=WorkerAccountingWindowKind.DAILY,
        at=_NOW,
    )
    monthly_window = worker_accounting_window(
        worker_instance_id=worker_id,
        window_kind=WorkerAccountingWindowKind.MONTHLY,
        at=_NOW,
    )
    daily_before = accounting.get_window_state(window=daily_window)
    monthly_before = accounting.get_window_state(window=monthly_window)
    assert daily_before is not None
    assert monthly_before is not None
    assert daily_before.execution_count == 1
    assert monthly_before.execution_count == 1
    postgresql_autonomous_work_bundle.close()

    reopened = open_bundle(schema_name)
    try:
        daily_after = reopened.worker_accounting.get_window_state(window=daily_window)
        monthly_after = reopened.worker_accounting.get_window_state(window=monthly_window)
        assert daily_after is not None
        assert monthly_after is not None
        assert daily_after.execution_count == 1
        assert monthly_after.execution_count == 1
    finally:
        reopened.close()


def test_postgresql_utc_daily_and_monthly_rollover(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(
        daily_execution_limit=1,
        monthly_execution_limit=5,
        max_concurrent_executions=5,
    )
    accounting = postgresql_autonomous_work_bundle.worker_accounting
    late_day = datetime(2026, 9, 4, 23, 59, tzinfo=_UTC)
    next_day = datetime(2026, 9, 5, 0, 1, tzinfo=_UTC)
    first = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="late-day",
            policy=policy,
            reserved_at=late_day,
        )
    )
    assert first.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="late-day"),
        execution_id=execution_id,
        bound_at=late_day,
    )
    accounting.release_execution(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        released_at=late_day + timedelta(minutes=1),
    )
    denied_same_day = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="same-day",
            policy=policy,
            reserved_at=late_day + timedelta(seconds=30),
        )
    )
    assert denied_same_day.disposition is WorkerBudgetAdmissionDisposition.DENIED
    allowed_next_day = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="next-day",
            policy=policy,
            reserved_at=next_day,
        )
    )
    assert allowed_next_day.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    month_end = datetime(2026, 9, 30, 23, 59, tzinfo=_UTC)
    next_month = datetime(2026, 10, 1, 0, 1, tzinfo=_UTC)
    month_start, month_end_bound = monthly_window_bounds(month_end)
    assert month_start == datetime(2026, 9, 1, 0, 0, tzinfo=_UTC)
    assert month_end_bound == datetime(2026, 10, 1, 0, 0, tzinfo=_UTC)
    monthly_window_sep = worker_accounting_window(
        worker_instance_id=worker_id,
        window_kind=WorkerAccountingWindowKind.MONTHLY,
        at=month_end,
    )
    monthly_state_sep = accounting.get_window_state(window=monthly_window_sep)
    assert monthly_state_sep is not None
    assert monthly_state_sep.execution_count >= 1
    reserve_oct = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="october",
            policy=policy,
            reserved_at=next_month,
        )
    )
    assert reserve_oct.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    oct_execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="october"),
        execution_id=oct_execution_id,
        bound_at=next_month,
    )
    accounting.release_execution(
        worker_instance_id=worker_id,
        execution_id=oct_execution_id,
        released_at=next_month + timedelta(minutes=1),
    )
    monthly_window_oct = worker_accounting_window(
        worker_instance_id=worker_id,
        window_kind=WorkerAccountingWindowKind.MONTHLY,
        at=next_month,
    )
    monthly_state_oct = accounting.get_window_state(window=monthly_window_oct)
    assert monthly_state_oct is not None
    assert monthly_state_oct.execution_count == 1


def test_postgresql_reservation_retry_is_idempotent(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(daily_execution_limit=1, max_concurrent_executions=2)
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    first_bundle = postgresql_autonomous_work_bundle
    first = first_bundle.worker_accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="same-key", policy=policy)
    )
    assert first.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    first_bundle.close()

    second_bundle = open_bundle(schema_name)
    try:
        second = second_bundle.worker_accounting.reserve(
            _reserve_request(worker_id=worker_id, source_ref="same-key", policy=policy)
        )
        assert second.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
        assert second.reservation == first.reservation
    finally:
        second_bundle.close()


def test_postgresql_conflicting_reservation_retry_returns_conflict(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(daily_execution_limit=5, max_concurrent_executions=5)
    accounting = postgresql_autonomous_work_bundle.worker_accounting
    first = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="same-key",
            policy=policy,
            profile_ref=_PROFILE_V1,
        )
    )
    assert first.disposition is WorkerBudgetAdmissionDisposition.ALLOWED
    conflict = accounting.reserve(
        _reserve_request(
            worker_id=worker_id,
            source_ref="same-key",
            policy=policy,
            profile_ref=_PROFILE_V2,
        )
    )
    assert conflict.disposition is WorkerBudgetAdmissionDisposition.CONFLICT


def test_postgresql_usage_replay_idempotent_and_conflict_rejected(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    worker_id = contract_suite.mint_worker_instance_id()
    policy = WorkerBudgetPolicy(max_concurrent_executions=2)
    accounting = postgresql_autonomous_work_bundle.worker_accounting
    accounting.reserve(
        _reserve_request(worker_id=worker_id, source_ref="usage", policy=policy)
    )
    execution_id = mint_execution_id()
    accounting.bind_execution(
        logical_dispatch=_logical_dispatch(worker_id=worker_id, source_ref="usage"),
        execution_id=execution_id,
        bound_at=_NOW,
    )
    usage_a = BudgetUsageTotals(total_tokens=10)
    usage_b = BudgetUsageTotals(total_tokens=20)
    accounting.record_consumption(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        usage=usage_a,
        recorded_at=_NOW,
    )
    accounting.record_consumption(
        worker_instance_id=worker_id,
        execution_id=execution_id,
        usage=usage_a,
        recorded_at=_NOW,
    )
    daily_window = worker_accounting_window(
        worker_instance_id=worker_id,
        window_kind=WorkerAccountingWindowKind.DAILY,
        at=_NOW,
    )
    state = accounting.get_window_state(window=daily_window)
    assert state is not None
    assert state.aggregate_usage.total_tokens == 10
    with pytest.raises(WorkerAccountingConflict):
        accounting.record_consumption(
            worker_instance_id=worker_id,
            execution_id=execution_id,
            usage=usage_b,
            recorded_at=_NOW,
        )
