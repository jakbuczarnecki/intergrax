# © Artur Czarnecki. All rights reserved.

"""AW-6B — PostgreSQL worker recovery episode qualification."""

from __future__ import annotations

import multiprocessing
from datetime import UTC, datetime

import pytest

import intergrax.autonomous_work.postgresql_repository as pg_repo_module
from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.autonomous_work.postgresql_repository import (
    AutonomousWorkSchemaVersionError,
    PostgreSQLAutonomousWorkStore,
)
from intergrax.autonomous_work.repository import WorkerRecoveryEpisodeClaimStatus
from intergrax.contracts.autonomous_work.revision import Revision
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
)
from tests.integration.autonomous_work.conftest import (
    materialization_options_for_schema,
    materialize_bundle,
    open_bundle,
    open_bundle_with_options,
    resolve_postgresql_config,
)
from tests.unit.autonomous_work import recovery_episode_repository_contracts as contracts
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = [pytest.mark.integration, pytest.mark.network]

_UTC = UTC
_NOW = datetime(2026, 9, 5, 12, 0, tzinfo=_UTC)
_CURRENT_SCHEMA_VERSION = pg_repo_module._SCHEMA_VERSION
_RECOVERY_TABLE = "aw_worker_recovery_episodes"
_ACCOUNTING_TABLE = "aw_worker_accounting_snapshots"


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
        5: [_RECOVERY_TABLE],
        4: [_RECOVERY_TABLE, _ACCOUNTING_TABLE],
        3: [_RECOVERY_TABLE, _ACCOUNTING_TABLE, "aw_goal_evaluation_cadence_states"],
        2: [
            _RECOVERY_TABLE,
            _ACCOUNTING_TABLE,
            "aw_goal_evaluation_cadence_states",
            "aw_worker_wake_up_receipts",
        ],
        1: [
            _RECOVERY_TABLE,
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


def _seed_v5_domain_data(bundle: AutonomousWorkRepositories) -> dict[str, object]:
    worker = bundle.worker_instance.create(contract_suite.worker_instance())
    return {"worker": worker}


def _multiprocess_recovery_claim_worker(
    dsn: str,
    schema_name: str,
    recovery_episode_id: str,
    expected_revision_value: int,
    claimed_at_iso: str,
    barrier: multiprocessing.Barrier,
    result_queue: multiprocessing.Queue[tuple[str, str | None]],
) -> None:
    bundle = open_bundle_with_options({"dsn": dsn, "schema_name": schema_name})
    try:
        barrier.wait(timeout=10)
        claim = bundle.worker_recovery_episode.claim_attempt(
            recovery_episode_id=recovery_episode_id,
            attempt_number=1,
            expected_revision=Revision(expected_revision_value),
            claimed_at=datetime.fromisoformat(claimed_at_iso),
        )
        result_queue.put(("ok", claim.status.name))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", type(exc).__name__))
    finally:
        bundle.close()


def test_postgresql_fresh_schema_v6_includes_recovery_table(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    assert _schema_version(store) == _CURRENT_SCHEMA_VERSION
    assert _table_exists(store, _RECOVERY_TABLE)
    seed = contracts.recovery_episode()
    created = postgresql_autonomous_work_bundle.worker_recovery_episode.create_or_get(seed)
    assert created.status.name in {"CREATED", "EXISTING"}


def test_postgresql_v5_to_v6_migration_preserves_existing_data(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    seeded = _seed_v5_domain_data(postgresql_autonomous_work_bundle)
    worker = seeded["worker"]
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    _downgrade_schema(store, target_version=5)
    assert not _table_exists(store, _RECOVERY_TABLE)

    migrated_bundle = open_bundle(schema_name)
    try:
        assert _schema_version(migrated_bundle.store) == _CURRENT_SCHEMA_VERSION  # type: ignore[arg-type]
        assert _table_exists(migrated_bundle.store, _RECOVERY_TABLE)  # type: ignore[arg-type]
        loaded = migrated_bundle.worker_instance.get(
            worker_instance_id=worker.worker_instance_id  # type: ignore[union-attr]
        )
        assert loaded == worker
        episode = contracts.recovery_episode()
        created = migrated_bundle.worker_recovery_episode.create_or_get(episode)
        assert created.status.name == "CREATED"
    finally:
        migrated_bundle.close()


def test_postgresql_migration_v5_to_v6_atomicity_on_failure(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded = _seed_v5_domain_data(postgresql_autonomous_work_bundle)
    worker = seeded["worker"]
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    store = postgresql_autonomous_work_bundle.store
    assert isinstance(store, PostgreSQLAutonomousWorkStore)
    _downgrade_schema(store, target_version=5)

    def failing_migration(self, session):  # type: ignore[no-untyped-def]
        session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_RECOVERY_TABLE} (
                recovery_episode_id TEXT NOT NULL PRIMARY KEY,
                worker_instance_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                revision INTEGER NOT NULL
            );
            """
        )
        raise AutonomousWorkSchemaVersionError("controlled migration failure")

    monkeypatch.setattr(PostgreSQLAutonomousWorkStore, "_migrate_v5_to_v6", failing_migration)
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
        assert int(row["schema_version"]) == 5
        assert (
            conn.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (schema_name, _RECOVERY_TABLE),
            ).fetchone()
            is None
        )
        worker_row = conn.execute(
            f"""
            SELECT record_json FROM {schema_name}.aw_worker_instances
            WHERE worker_instance_id = %s
            """,
            (worker.worker_instance_id.strip(),),  # type: ignore[union-attr]
        ).fetchone()
        assert worker_row is not None

    monkeypatch.undo()
    migrated_bundle = open_bundle(schema_name)
    try:
        loaded = migrated_bundle.worker_instance.get(
            worker_instance_id=worker.worker_instance_id  # type: ignore[union-attr]
        )
        assert loaded == worker
        assert _schema_version(migrated_bundle.store) == _CURRENT_SCHEMA_VERSION  # type: ignore[arg-type]
        assert _table_exists(migrated_bundle.store, _RECOVERY_TABLE)  # type: ignore[arg-type]
    finally:
        migrated_bundle.close()


@pytest.mark.parametrize("from_version", [1, 2, 3, 4, 5])
def test_postgresql_recovery_migration_chain_reaches_current_version(
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
        assert _schema_version(migrated_bundle.store) == _CURRENT_SCHEMA_VERSION  # type: ignore[arg-type]
        assert _table_exists(migrated_bundle.store, _RECOVERY_TABLE)  # type: ignore[arg-type]
        loaded = migrated_bundle.worker_instance.get(
            worker_instance_id=created.worker_instance_id
        )
        assert loaded == created
    finally:
        migrated_bundle.close()


def test_postgresql_recovery_repository_contracts(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    repo = postgresql_autonomous_work_bundle.worker_recovery_episode

    def factory():
        return repo

    contracts.run_recovery_episode_repository_contract_suite(factory)


def test_postgresql_recovery_restart_preserves_episode_state(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    repo = postgresql_autonomous_work_bundle.worker_recovery_episode
    seed = contracts.recovery_episode()
    created = repo.create_or_get(seed)
    claim = repo.claim_attempt(
        recovery_episode_id=seed.recovery_episode_id,
        attempt_number=1,
        expected_revision=created.episode.revision,
        claimed_at=_NOW,
    )
    postgresql_autonomous_work_bundle.close()

    reopened = open_bundle(schema_name)
    try:
        loaded = reopened.worker_recovery_episode.get(
            recovery_episode_id=seed.recovery_episode_id
        )
        assert loaded == claim.episode
    finally:
        reopened.close()


def test_postgresql_recovery_cross_connection_create_idempotency(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    options = materialization_options_for_schema(schema_name)
    seed = contracts.recovery_episode()
    first_bundle = materialize_bundle(options)
    second_bundle = materialize_bundle(options)
    try:
        first = first_bundle.worker_recovery_episode.create_or_get(seed)
        second = second_bundle.worker_recovery_episode.create_or_get(seed)
        assert first.status.name == "CREATED"
        assert second.status.name == "EXISTING"
        assert first.episode == second.episode
    finally:
        first_bundle.close()
        second_bundle.close()


def test_postgresql_recovery_conflicting_create(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    repo = postgresql_autonomous_work_bundle.worker_recovery_episode
    seed = contracts.recovery_episode()
    repo.create_or_get(seed)
    conflict = contracts.recovery_episode(
        recovery_decision_id=f"{seed.recovery_decision_id}:other",
    )
    result = repo.create_or_get(conflict)
    assert result.status.name == "CONFLICT"


def test_postgresql_recovery_multiprocess_attempt_claim(
    postgresql_autonomous_work_bundle: AutonomousWorkRepositories,
) -> None:
    config = resolve_postgresql_config()
    assert config is not None
    dsn = config.connection_string()
    assert dsn is not None
    schema_name = postgresql_autonomous_work_bundle.store.schema_name
    seed = contracts.recovery_episode()
    created = postgresql_autonomous_work_bundle.worker_recovery_episode.create_or_get(seed)
    postgresql_autonomous_work_bundle.close()

    barrier = multiprocessing.Barrier(2)
    result_queue: multiprocessing.Queue[tuple[str, str | None]] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_multiprocess_recovery_claim_worker,
            args=(
                dsn,
                schema_name,
                seed.recovery_episode_id,
                created.episode.revision.value,
                _NOW.isoformat(),
                barrier,
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    outcomes: list[str] = []
    while not result_queue.empty():
        status, value = result_queue.get_nowait()
        assert status == "ok"
        assert value is not None
        outcomes.append(value)

    assert outcomes.count(WorkerRecoveryEpisodeClaimStatus.CLAIMED.name) == 1
    assert len(outcomes) == 2
    assert (
        outcomes.count(WorkerRecoveryEpisodeClaimStatus.ALREADY_CLAIMED.name)
        + outcomes.count(WorkerRecoveryEpisodeClaimStatus.REVISION_CONFLICT.name)
    ) == 1
