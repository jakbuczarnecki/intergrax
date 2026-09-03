# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production PostgreSQL adapter for Autonomous Work durable state (AW-2C)."""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Generic, TypeVar

from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityConflict,
    AutonomousWorkEntityNotFound,
    AutonomousWorkRepositoryCapabilities,
    AutonomousWorkRevisionConflict,
    WorkerWakeUpReceiptClaim,
    WorkerWakeUpReceiptClaimStatus,
)
from intergrax.autonomous_work.wake_up_receipt_claim import resolve_wake_up_receipt_claim
from intergrax.autonomous_work.serialization import (
    responsibility_from_json,
    responsibility_to_json,
    worker_definition_from_json,
    worker_definition_to_json,
    worker_goal_from_json,
    worker_goal_to_json,
    worker_instance_from_json,
    worker_instance_to_json,
    worker_principal_binding_from_json,
    worker_principal_binding_to_json,
    worker_wake_up_receipt_from_json,
    worker_wake_up_receipt_to_json,
    work_continuity_state_from_json,
    work_continuity_state_to_json,
)
from intergrax.contracts.autonomous_work.continuity import WorkContinuityState
from intergrax.contracts.autonomous_work.goal import WorkerGoal
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WakeUpId,
    WorkerDefinitionId,
    WorkerGoalId,
    WorkerInstanceId,
)
from intergrax.contracts.autonomous_work.principal_binding import WorkerPrincipalBinding
from intergrax.contracts.autonomous_work.responsibility import Responsibility
from intergrax.contracts.autonomous_work.revision import (
    DefinitionRevision,
    Revision,
    initial_revision,
)
from intergrax.contracts.autonomous_work.worker import WorkerDefinition, WorkerInstance
from intergrax.contracts.autonomous_work.wake_up import WorkerWakeUpReceipt
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import (
    validate_schema_identifier,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLIsolationLevel,
    PostgreSQLSession,
    is_postgresql_unique_violation,
)

_CLOSED_ERROR = "Autonomous Work repository store is closed"
_BACKEND_ID = "autonomous_work.postgresql"
_CAPABILITIES = AutonomousWorkRepositoryCapabilities(
    backend_id=_BACKEND_ID,
    durable=True,
    reference_only=False,
)
_ISOLATION_LEVEL = PostgreSQLIsolationLevel.READ_COMMITTED
# v2 introduced ``aw_worker_principal_bindings`` before AW-3A qualification closed.
# v3 introduced ``aw_worker_wake_up_receipts`` for AW-4A durable wake-up idempotency.
_SCHEMA_VERSION_V1 = 1
_SCHEMA_VERSION_V2 = 2
_SCHEMA_VERSION_V3 = 3
_SCHEMA_VERSION = _SCHEMA_VERSION_V3
_SCHEMA_META_TABLE = "autonomous_work_schema_meta"
_SCHEMA_LOCK_KEY = "autonomous_work_schema_init"

_EntityT = TypeVar("_EntityT")


class AutonomousWorkSchemaVersionError(IntegrationConfigurationError):
    """Raised when persisted Autonomous Work schema version is unsupported."""


def _require_revision_argument(
    value: object,
    *,
    param_name: str = "expected_revision",
) -> Revision:
    if not isinstance(value, Revision):
        raise TypeError(f"{param_name} must be Revision, got {type(value).__name__}")
    return value


def _unique_violation(exc: BaseException) -> bool:
    return is_postgresql_unique_violation(exc)


class PostgreSQLAutonomousWorkStore:
    """Shared PostgreSQL schema and connection factory for Autonomous Work repositories."""

    def __init__(
        self,
        connection_provider: PostgreSQLConnectionProvider,
        *,
        schema_name: str,
    ) -> None:
        resolved_schema = validate_schema_identifier(schema_name.strip())
        self._schema_name = resolved_schema
        self._provider = connection_provider
        self._closed = False
        self._init_lock = threading.Lock()
        self._schema_ready = False
        try:
            with self._provider.connection() as session:
                self._initialize_schema(session)
                session.commit()
        except IntegrationConfigurationError:
            self._closed = True
            raise
        except Exception as exc:
            self._closed = True
            raise IntegrationConfigurationError(
                "PostgreSQL Autonomous Work store could not be opened"
            ) from exc

    @property
    def schema_name(self) -> str:
        return self._schema_name

    @property
    def connection_provider(self) -> PostgreSQLConnectionProvider:
        return self._provider

    def close(self) -> None:
        with self._init_lock:
            if self._closed:
                return
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(_CLOSED_ERROR)

    @contextmanager
    def transaction(self) -> Generator[PostgreSQLSession, None, None]:
        self._ensure_open()
        with self._provider.transaction(isolation_level=_ISOLATION_LEVEL) as session:
            yield session

    def _initialize_schema(self, session: PostgreSQLSession) -> None:
        with self._init_lock:
            if self._schema_ready:
                return
            self._provider.ensure_schema_exists(session, self._schema_name)
            session.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_SCHEMA_LOCK_KEY,))
            session.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_SCHEMA_META_TABLE} (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    schema_version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS aw_worker_definitions (
                    worker_definition_id TEXT NOT NULL,
                    definition_revision INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (worker_definition_id, definition_revision)
                );

                CREATE TABLE IF NOT EXISTS aw_worker_instances (
                    worker_instance_id TEXT NOT NULL PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS aw_responsibilities (
                    responsibility_id TEXT NOT NULL PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS aw_worker_goals (
                    goal_id TEXT NOT NULL PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS aw_work_continuity_states (
                    worker_instance_id TEXT NOT NULL PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS aw_worker_principal_bindings (
                    worker_instance_id TEXT NOT NULL PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS aw_worker_wake_up_receipts (
                    worker_instance_id TEXT NOT NULL,
                    wake_up_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    accepted_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (worker_instance_id, wake_up_id)
                );
                """
            )
            session.execute(
                f"""
                INSERT INTO {_SCHEMA_META_TABLE} (id, schema_version)
                VALUES (1, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (_SCHEMA_VERSION,),
            )
            row = session.execute(
                f"SELECT schema_version FROM {_SCHEMA_META_TABLE} WHERE id = 1"
            ).fetchone()
            if row is None:
                raise IntegrationConfigurationError(
                    "Autonomous Work schema metadata is missing after bootstrap"
                )
            persisted_version = int(row["schema_version"])
            if persisted_version > _SCHEMA_VERSION:
                raise AutonomousWorkSchemaVersionError(
                    "Autonomous Work schema version is newer than supported adapter"
                )
            if persisted_version < _SCHEMA_VERSION:
                self._migrate_schema(session, from_version=persisted_version)
                row = session.execute(
                    f"SELECT schema_version FROM {_SCHEMA_META_TABLE} WHERE id = 1"
                ).fetchone()
                if row is None:
                    raise IntegrationConfigurationError(
                        "Autonomous Work schema metadata is missing after migration"
                    )
                persisted_version = int(row["schema_version"])
            if persisted_version != _SCHEMA_VERSION:
                raise AutonomousWorkSchemaVersionError(
                    "Autonomous Work schema requires migration to a newer version"
                )
            self._schema_ready = True

    def _migrate_schema(
        self,
        session: PostgreSQLSession,
        *,
        from_version: int,
    ) -> None:
        if from_version == _SCHEMA_VERSION_V1:
            self._migrate_v1_to_v2(session)
            from_version = _SCHEMA_VERSION_V2
        if from_version == _SCHEMA_VERSION_V2:
            self._migrate_v2_to_v3(session)
            return
        raise AutonomousWorkSchemaVersionError(
            f"unsupported Autonomous Work schema migration from version {from_version}"
        )

    def _complete_migration_step(
        self,
        session: PostgreSQLSession,
        *,
        expected_from: int,
        expected_to: int,
        updated_rows: int,
    ) -> None:
        if updated_rows != 1:
            raise AutonomousWorkSchemaVersionError(
                (
                    "Autonomous Work schema migration "
                    f"{expected_from}→{expected_to} did not update metadata "
                    f"(expected source version {expected_from})"
                )
            )
        row = session.execute(
            f"SELECT schema_version FROM {_SCHEMA_META_TABLE} WHERE id = 1"
        ).fetchone()
        if row is None:
            raise IntegrationConfigurationError(
                "Autonomous Work schema metadata is missing after migration step"
            )
        persisted_version = int(row["schema_version"])
        if persisted_version != expected_to:
            raise AutonomousWorkSchemaVersionError(
                (
                    "Autonomous Work schema migration "
                    f"{expected_from}→{expected_to} left metadata at version "
                    f"{persisted_version}"
                )
            )

    def _migrate_v1_to_v2(self, session: PostgreSQLSession) -> None:
        session.execute(
            """
            CREATE TABLE IF NOT EXISTS aw_worker_principal_bindings (
                worker_instance_id TEXT NOT NULL PRIMARY KEY,
                record_json TEXT NOT NULL,
                revision INTEGER NOT NULL
            );
            """
        )
        updated = session.execute(
            f"""
            UPDATE {_SCHEMA_META_TABLE}
            SET schema_version = %s
            WHERE id = 1 AND schema_version = %s
            """,
            (_SCHEMA_VERSION_V2, _SCHEMA_VERSION_V1),
        )
        self._complete_migration_step(
            session,
            expected_from=_SCHEMA_VERSION_V1,
            expected_to=_SCHEMA_VERSION_V2,
            updated_rows=updated.rowcount,
        )

    def _migrate_v2_to_v3(self, session: PostgreSQLSession) -> None:
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
        updated = session.execute(
            f"""
            UPDATE {_SCHEMA_META_TABLE}
            SET schema_version = %s
            WHERE id = 1 AND schema_version = %s
            """,
            (_SCHEMA_VERSION_V3, _SCHEMA_VERSION_V2),
        )
        self._complete_migration_step(
            session,
            expected_from=_SCHEMA_VERSION_V2,
            expected_to=_SCHEMA_VERSION_V3,
            updated_rows=updated.rowcount,
        )


class _ImmutableCreateRepository(Generic[_EntityT]):
    def __init__(
        self,
        store: PostgreSQLAutonomousWorkStore,
        *,
        table_name: str,
        entity_kind: str,
        to_json: Callable[[_EntityT], str],
        from_json: Callable[[str], _EntityT],
        identity_for_conflict: Callable[[_EntityT], str],
        insert_sql: str,
        select_sql: str,
        insert_params: Callable[[_EntityT, str], tuple[Any, ...]],
        select_params: Callable[[_EntityT], tuple[Any, ...]],
    ) -> None:
        self._store = store
        self._table_name = table_name
        self._entity_kind = entity_kind
        self._to_json = to_json
        self._from_json = from_json
        self._identity_for_conflict = identity_for_conflict
        self._insert_sql = insert_sql
        self._select_sql = select_sql
        self._insert_params = insert_params
        self._select_params = select_params

    def create(self, entity: _EntityT) -> _EntityT:
        record_json = self._to_json(entity)
        entity_id = self._identity_for_conflict(entity)
        try:
            with self._store.transaction() as conn:
                conn.execute(self._insert_sql, self._insert_params(entity, record_json))
                return entity
        except Exception as exc:
            if not _unique_violation(exc):
                raise
            with self._store.transaction() as conn:
                existing = self._get_in_transaction(conn, entity)
                if existing is None:
                    raise AutonomousWorkEntityConflict(
                        f"{self._entity_kind} already exists with different content "
                        f"for {entity_id}"
                    ) from exc
                if existing == entity:
                    return existing
                raise AutonomousWorkEntityConflict(
                    f"{self._entity_kind} already exists with different content "
                    f"for {entity_id}"
                ) from exc

    def _get_in_transaction(self, conn: PostgreSQLSession, entity: _EntityT) -> _EntityT | None:
        row = conn.execute(self._select_sql, self._select_params(entity)).fetchone()
        if row is None:
            return None
        return self._from_json(row["record_json"])


class _RevisionedRepository(Generic[_EntityT]):
    def __init__(
        self,
        store: PostgreSQLAutonomousWorkStore,
        *,
        table_name: str,
        entity_kind: str,
        to_json: Callable[[_EntityT], str],
        from_json: Callable[[str], _EntityT],
        identity_for_conflict: Callable[[_EntityT], str],
        entity_id_column: str,
        read_revision: Callable[[_EntityT], Revision],
        write_revision: Callable[[_EntityT, Revision], _EntityT],
        insert_sql: str,
        select_sql: str,
        update_sql: str,
        insert_params: Callable[[_EntityT, str], tuple[Any, ...]],
        select_params: Callable[[str], tuple[Any, ...]],
        update_params: Callable[[_EntityT, str, Revision, str], tuple[Any, ...]],
    ) -> None:
        self._store = store
        self._table_name = table_name
        self._entity_kind = entity_kind
        self._to_json = to_json
        self._from_json = from_json
        self._identity_for_conflict = identity_for_conflict
        self._entity_id_column = entity_id_column
        self._read_revision = read_revision
        self._write_revision = write_revision
        self._insert_sql = insert_sql
        self._select_sql = select_sql
        self._update_sql = update_sql
        self._insert_params = insert_params
        self._select_params = select_params
        self._update_params = update_params

    def create(self, entity: _EntityT) -> _EntityT:
        if self._read_revision(entity) != initial_revision():
            raise ValueError(
                f"{self._entity_kind} create requires revision {initial_revision().value}"
            )
        record_json = self._to_json(entity)
        entity_id = self._identity_for_conflict(entity)
        try:
            with self._store.transaction() as conn:
                conn.execute(self._insert_sql, self._insert_params(entity, record_json))
                return entity
        except Exception as exc:
            if not _unique_violation(exc):
                raise
            with self._store.transaction() as conn:
                existing = self._get_in_transaction(conn, entity_id)
                if existing is None:
                    raise AutonomousWorkEntityConflict(
                        f"{self._entity_kind} already exists with different content "
                        f"for {entity_id}"
                    ) from exc
                if existing == entity:
                    return existing
                raise AutonomousWorkEntityConflict(
                    f"{self._entity_kind} already exists with different content "
                    f"for {entity_id}"
                ) from exc

    def get(self, entity_id: str) -> _EntityT | None:
        with self._store.transaction() as conn:
            return self._get_in_transaction(conn, entity_id)

    def replace(
        self,
        entity: _EntityT,
        *,
        expected_revision: Revision,
    ) -> _EntityT:
        expected_revision = _require_revision_argument(expected_revision)
        entity_id = self._identity_for_conflict(entity)
        next_revision = Revision(expected_revision.value + 1)
        persisted = self._write_revision(entity, next_revision)
        record_json = self._to_json(persisted)
        with self._store.transaction() as conn:
            current = self._get_in_transaction(conn, entity_id)
            if current is None:
                raise AutonomousWorkEntityNotFound(
                    f"{self._entity_kind} not found for {entity_id}"
                )
            current_revision = self._read_revision(current)
            if current_revision != expected_revision:
                raise AutonomousWorkRevisionConflict(
                    (
                        f"{self._entity_kind} revision conflict for {entity_id}: "
                        f"expected {expected_revision.value}, actual {current_revision.value}"
                    ),
                    entity_kind=self._entity_kind,
                    entity_id=entity_id,
                    expected_revision=expected_revision,
                    actual_revision=current_revision,
                )
            candidate_revision = self._read_revision(entity)
            if candidate_revision != expected_revision:
                raise ValueError(
                    (
                        f"{self._entity_kind} replacement candidate revision "
                        f"{candidate_revision.value} does not match "
                        f"expected_revision {expected_revision.value}"
                    )
                )
            updated = conn.execute(
                self._update_sql,
                self._update_params(persisted, record_json, expected_revision, entity_id),
            )
            if updated.rowcount == 1:
                return persisted
            raced = self._get_in_transaction(conn, entity_id)
            if raced is None:
                raise AutonomousWorkEntityNotFound(
                    f"{self._entity_kind} not found for {entity_id}"
                )
            actual_revision = self._read_revision(raced)
            raise AutonomousWorkRevisionConflict(
                (
                    f"{self._entity_kind} revision conflict for {entity_id}: "
                    f"expected {expected_revision.value}, actual {actual_revision.value}"
                ),
                entity_kind=self._entity_kind,
                entity_id=entity_id,
                expected_revision=expected_revision,
                actual_revision=actual_revision,
            )

    def _get_in_transaction(
        self,
        conn: PostgreSQLSession,
        entity_id: str,
    ) -> _EntityT | None:
        row = conn.execute(self._select_sql, self._select_params(entity_id)).fetchone()
        if row is None:
            return None
        return self._from_json(row["record_json"])


class PostgreSQLWorkerDefinitionRepository:
    """Production repository for immutable WorkerDefinition versions."""

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._store = store
        self._delegate = _ImmutableCreateRepository(
            store,
            table_name="aw_worker_definitions",
            entity_kind="WorkerDefinition",
            to_json=worker_definition_to_json,
            from_json=worker_definition_from_json,
            identity_for_conflict=lambda definition: (
                f"{definition.worker_definition_id}@{definition.revision.value}"
            ),
            insert_sql="""
                INSERT INTO aw_worker_definitions (
                    worker_definition_id, definition_revision, record_json
                ) VALUES (%s, %s, %s)
            """,
            select_sql="""
                SELECT record_json FROM aw_worker_definitions
                WHERE worker_definition_id = %s AND definition_revision = %s
            """,
            insert_params=lambda definition, record_json: (
                definition.worker_definition_id.strip(),
                definition.revision.value,
                record_json,
            ),
            select_params=lambda definition: (
                definition.worker_definition_id.strip(),
                definition.revision.value,
            ),
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, definition: WorkerDefinition) -> WorkerDefinition:
        return self._delegate.create(definition)

    def get(
        self,
        *,
        worker_definition_id: WorkerDefinitionId,
        definition_revision: DefinitionRevision,
    ) -> WorkerDefinition | None:
        with self._store.transaction() as conn:
            row = conn.execute(
                """
                SELECT record_json FROM aw_worker_definitions
                WHERE worker_definition_id = %s AND definition_revision = %s
                """,
                (worker_definition_id.strip(), definition_revision.value),
            ).fetchone()
            if row is None:
                return None
            return worker_definition_from_json(row["record_json"])


class PostgreSQLWorkerInstanceRepository:
    """Production repository for durable WorkerInstance records."""

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._delegate = _RevisionedRepository(
            store,
            table_name="aw_worker_instances",
            entity_kind="WorkerInstance",
            to_json=worker_instance_to_json,
            from_json=worker_instance_from_json,
            identity_for_conflict=lambda instance: instance.worker_instance_id,
            entity_id_column="worker_instance_id",
            read_revision=lambda instance: instance.revision,
            write_revision=lambda instance, revision: replace(instance, revision=revision),
            insert_sql="""
                INSERT INTO aw_worker_instances (
                    worker_instance_id, record_json, revision
                ) VALUES (%s, %s, %s)
            """,
            select_sql="""
                SELECT record_json FROM aw_worker_instances
                WHERE worker_instance_id = %s
            """,
            update_sql="""
                UPDATE aw_worker_instances
                SET record_json = %s, revision = %s
                WHERE worker_instance_id = %s AND revision = %s
            """,
            insert_params=lambda instance, record_json: (
                instance.worker_instance_id.strip(),
                record_json,
                instance.revision.value,
            ),
            select_params=lambda entity_id: (entity_id.strip(),),
            update_params=lambda persisted, record_json, expected_revision, entity_id: (
                record_json,
                persisted.revision.value,
                entity_id.strip(),
                expected_revision.value,
            ),
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, instance: WorkerInstance) -> WorkerInstance:
        return self._delegate.create(instance)

    def get(self, *, worker_instance_id: WorkerInstanceId) -> WorkerInstance | None:
        return self._delegate.get(worker_instance_id)

    def replace(
        self,
        instance: WorkerInstance,
        *,
        expected_revision: Revision,
    ) -> WorkerInstance:
        return self._delegate.replace(instance, expected_revision=expected_revision)


class PostgreSQLResponsibilityRepository:
    """Production repository for Responsibility records."""

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._delegate = _RevisionedRepository(
            store,
            table_name="aw_responsibilities",
            entity_kind="Responsibility",
            to_json=responsibility_to_json,
            from_json=responsibility_from_json,
            identity_for_conflict=lambda responsibility: responsibility.responsibility_id,
            entity_id_column="responsibility_id",
            read_revision=lambda responsibility: responsibility.revision,
            write_revision=lambda responsibility, revision: replace(
                responsibility, revision=revision
            ),
            insert_sql="""
                INSERT INTO aw_responsibilities (
                    responsibility_id, record_json, revision
                ) VALUES (%s, %s, %s)
            """,
            select_sql="""
                SELECT record_json FROM aw_responsibilities
                WHERE responsibility_id = %s
            """,
            update_sql="""
                UPDATE aw_responsibilities
                SET record_json = %s, revision = %s
                WHERE responsibility_id = %s AND revision = %s
            """,
            insert_params=lambda responsibility, record_json: (
                responsibility.responsibility_id.strip(),
                record_json,
                responsibility.revision.value,
            ),
            select_params=lambda entity_id: (entity_id.strip(),),
            update_params=lambda persisted, record_json, expected_revision, entity_id: (
                record_json,
                persisted.revision.value,
                entity_id.strip(),
                expected_revision.value,
            ),
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, responsibility: Responsibility) -> Responsibility:
        return self._delegate.create(responsibility)

    def get(self, *, responsibility_id: ResponsibilityId) -> Responsibility | None:
        return self._delegate.get(responsibility_id)

    def replace(
        self,
        responsibility: Responsibility,
        *,
        expected_revision: Revision,
    ) -> Responsibility:
        return self._delegate.replace(responsibility, expected_revision=expected_revision)

    def list_for_worker_instance(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
    ) -> tuple[Responsibility, ...]:
        with self._delegate._store.transaction() as conn:
            rows = conn.execute(
                """
                SELECT record_json
                FROM aw_responsibilities
                WHERE record_json::json->>'worker_instance_id' = %s
                ORDER BY responsibility_id
                """,
                (worker_instance_id.strip(),),
            ).fetchall()
        return tuple(
            responsibility_from_json(row["record_json"]) for row in rows
        )


class PostgreSQLWorkerGoalRepository:
    """Production repository for WorkerGoal records."""

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._delegate = _RevisionedRepository(
            store,
            table_name="aw_worker_goals",
            entity_kind="WorkerGoal",
            to_json=worker_goal_to_json,
            from_json=worker_goal_from_json,
            identity_for_conflict=lambda goal: goal.goal_id,
            entity_id_column="goal_id",
            read_revision=lambda goal: goal.revision,
            write_revision=lambda goal, revision: replace(goal, revision=revision),
            insert_sql="""
                INSERT INTO aw_worker_goals (goal_id, record_json, revision)
                VALUES (%s, %s, %s)
            """,
            select_sql="""
                SELECT record_json FROM aw_worker_goals WHERE goal_id = %s
            """,
            update_sql="""
                UPDATE aw_worker_goals
                SET record_json = %s, revision = %s
                WHERE goal_id = %s AND revision = %s
            """,
            insert_params=lambda goal, record_json: (
                goal.goal_id.strip(),
                record_json,
                goal.revision.value,
            ),
            select_params=lambda entity_id: (entity_id.strip(),),
            update_params=lambda persisted, record_json, expected_revision, entity_id: (
                record_json,
                persisted.revision.value,
                entity_id.strip(),
                expected_revision.value,
            ),
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, goal: WorkerGoal) -> WorkerGoal:
        return self._delegate.create(goal)

    def get(self, *, goal_id: WorkerGoalId) -> WorkerGoal | None:
        return self._delegate.get(goal_id)

    def replace(
        self,
        goal: WorkerGoal,
        *,
        expected_revision: Revision,
    ) -> WorkerGoal:
        return self._delegate.replace(goal, expected_revision=expected_revision)

    def list_for_responsibility(
        self,
        *,
        responsibility_id: ResponsibilityId,
    ) -> tuple[WorkerGoal, ...]:
        with self._delegate._store.transaction() as conn:
            rows = conn.execute(
                """
                SELECT record_json
                FROM aw_worker_goals
                WHERE record_json::json->>'responsibility_id' = %s
                ORDER BY goal_id
                """,
                (responsibility_id.strip(),),
            ).fetchall()
        return tuple(worker_goal_from_json(row["record_json"]) for row in rows)


class PostgreSQLWorkContinuityStateRepository:
    """Production repository for WorkContinuityState checkpoints."""

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._delegate = _RevisionedRepository(
            store,
            table_name="aw_work_continuity_states",
            entity_kind="WorkContinuityState",
            to_json=work_continuity_state_to_json,
            from_json=work_continuity_state_from_json,
            identity_for_conflict=lambda state: state.worker_instance_ref,
            entity_id_column="worker_instance_id",
            read_revision=lambda state: state.revision,
            write_revision=lambda state, revision: replace(state, revision=revision),
            insert_sql="""
                INSERT INTO aw_work_continuity_states (
                    worker_instance_id, record_json, revision
                ) VALUES (%s, %s, %s)
            """,
            select_sql="""
                SELECT record_json FROM aw_work_continuity_states
                WHERE worker_instance_id = %s
            """,
            update_sql="""
                UPDATE aw_work_continuity_states
                SET record_json = %s, revision = %s
                WHERE worker_instance_id = %s AND revision = %s
            """,
            insert_params=lambda state, record_json: (
                state.worker_instance_ref.strip(),
                record_json,
                state.revision.value,
            ),
            select_params=lambda entity_id: (entity_id.strip(),),
            update_params=lambda persisted, record_json, expected_revision, entity_id: (
                record_json,
                persisted.revision.value,
                entity_id.strip(),
                expected_revision.value,
            ),
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, state: WorkContinuityState) -> WorkContinuityState:
        return self._delegate.create(state)

    def get(self, *, worker_instance_id: WorkerInstanceId) -> WorkContinuityState | None:
        return self._delegate.get(worker_instance_id)

    def replace(
        self,
        state: WorkContinuityState,
        *,
        expected_revision: Revision,
    ) -> WorkContinuityState:
        return self._delegate.replace(state, expected_revision=expected_revision)


class PostgreSQLWorkerPrincipalBindingRepository:
    """Production repository for immutable WorkerPrincipalBinding records."""

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._store = store
        self._delegate = _ImmutableCreateRepository(
            store,
            table_name="aw_worker_principal_bindings",
            entity_kind="WorkerPrincipalBinding",
            to_json=worker_principal_binding_to_json,
            from_json=worker_principal_binding_from_json,
            identity_for_conflict=lambda binding: binding.worker_instance_id,
            insert_sql="""
                INSERT INTO aw_worker_principal_bindings (
                    worker_instance_id, record_json, revision
                ) VALUES (%s, %s, %s)
            """,
            select_sql="""
                SELECT record_json FROM aw_worker_principal_bindings
                WHERE worker_instance_id = %s
            """,
            insert_params=lambda binding, record_json: (
                binding.worker_instance_id.strip(),
                record_json,
                binding.revision.value,
            ),
            select_params=lambda binding: (binding.worker_instance_id.strip(),),
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, binding: WorkerPrincipalBinding) -> WorkerPrincipalBinding:
        if binding.revision != initial_revision():
            raise ValueError(
                f"WorkerPrincipalBinding create requires revision {initial_revision().value}"
            )
        return self._delegate.create(binding)

    def get(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
    ) -> WorkerPrincipalBinding | None:
        with self._store.transaction() as conn:
            row = conn.execute(
                """
                SELECT record_json FROM aw_worker_principal_bindings
                WHERE worker_instance_id = %s
                """,
                (worker_instance_id.strip(),),
            ).fetchone()
            if row is None:
                return None
            return worker_principal_binding_from_json(row["record_json"])


class PostgreSQLWorkerWakeUpReceiptRepository:
    """Production repository for durable wake-up admission receipts."""

    def __init__(self, store: PostgreSQLAutonomousWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return _CAPABILITIES

    def claim(self, receipt: WorkerWakeUpReceipt) -> WorkerWakeUpReceiptClaim:
        record_json = worker_wake_up_receipt_to_json(receipt)
        with self._store.transaction() as conn:
            inserted = conn.execute(
                """
                INSERT INTO aw_worker_wake_up_receipts (
                    worker_instance_id, wake_up_id, record_json, accepted_at
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (worker_instance_id, wake_up_id) DO NOTHING
                RETURNING record_json
                """,
                (
                    receipt.worker_instance_id.strip(),
                    receipt.wake_up_id.strip(),
                    record_json,
                    receipt.accepted_at,
                ),
            ).fetchone()
            if inserted is not None:
                return WorkerWakeUpReceiptClaim(
                    status=WorkerWakeUpReceiptClaimStatus.CLAIMED,
                    receipt=worker_wake_up_receipt_from_json(inserted["record_json"]),
                )
            row = conn.execute(
                """
                SELECT record_json FROM aw_worker_wake_up_receipts
                WHERE worker_instance_id = %s AND wake_up_id = %s
                """,
                (
                    receipt.worker_instance_id.strip(),
                    receipt.wake_up_id.strip(),
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    "wake-up receipt conflict without stored canonical receipt"
                )
            stored = worker_wake_up_receipt_from_json(row["record_json"])
            return resolve_wake_up_receipt_claim(receipt, stored)

    def get(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        wake_up_id: WakeUpId,
    ) -> WorkerWakeUpReceipt | None:
        with self._store.transaction() as conn:
            row = conn.execute(
                """
                SELECT record_json FROM aw_worker_wake_up_receipts
                WHERE worker_instance_id = %s AND wake_up_id = %s
                """,
                (worker_instance_id.strip(), wake_up_id.strip()),
            ).fetchone()
            if row is None:
                return None
            return worker_wake_up_receipt_from_json(row["record_json"])
