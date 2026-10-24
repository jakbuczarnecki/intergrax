# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
PostgreSQL connection/session provider — canonical driver and transaction ownership.

Only this module may import ``psycopg`` for relational-store PostgreSQL integration.
Domain adapters consume ``PostgreSQLConnectionProvider`` via typed session APIs.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from typing import Any

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationError,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
    validate_schema_identifier,
)

_PSYCOPG_INSTALL_GUIDANCE = (
    "PostgreSQL integration requires psycopg. "
    "Install with: uv sync --extra integrations-postgresql"
)


class PostgreSQLIsolationLevel(str, Enum):
    """Bounded PostgreSQL transaction isolation levels."""

    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


def import_psycopg() -> tuple[Any, Any, Any, Any]:
    """Canonical psycopg import boundary for PostgreSQL platform infrastructure."""
    try:
        import psycopg
        from psycopg import errors, sql
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise IntegrationConfigurationError(_PSYCOPG_INSTALL_GUIDANCE) from exc
    return psycopg, errors, dict_row, sql


def is_postgresql_unique_violation(exc: BaseException) -> bool:
    """Detect PostgreSQL unique-constraint violations without leaking driver types to callers."""
    _, errors, _, _ = import_psycopg()
    return isinstance(exc, errors.UniqueViolation)


def is_postgresql_undefined_table(exc: BaseException) -> bool:
    """Detect missing-relation errors without leaking driver types to callers."""
    _, errors, _, _ = import_psycopg()
    return isinstance(exc, errors.UndefinedTable)


class PostgreSQLExecutionResult:
    """Typed execution result for domain SQL repositories."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def fetchone(self) -> Mapping[str, Any] | None:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def fetchall(self) -> Sequence[Mapping[str, Any]]:
        return [dict(row) for row in self._cursor.fetchall()]

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class PostgreSQLBoundedFetchTransactionScope(str, Enum):
    """How bounded fetch may use the connection transaction boundary."""

    ADAPTER_OWNED_READONLY = "adapter_owned_readonly"
    CALLER_OWNED = "caller_owned"


class PostgreSQLBoundedFetchSafetyError(IntegrationError):
    """Bounded fetch rejected: connection state is unsafe or ambiguous."""

    def __init__(self, message: str, *, transaction_status: str = "") -> None:
        super().__init__(message)
        self.transaction_status = transaction_status


def classify_postgresql_bounded_fetch_transaction_scope(
    connection: Any,
) -> PostgreSQLBoundedFetchTransactionScope:
    """
    Classify whether bounded fetch may open its own READ ONLY transaction.

    IDLE → adapter-owned READ ONLY may begin. INTRANS → caller-owned; no BEGIN/commit/rollback.
    INERROR, ACTIVE, UNKNOWN, and closed connections fail closed.
    """
    from psycopg import pq

    if getattr(connection, "closed", False):
        raise PostgreSQLBoundedFetchSafetyError(
            "PostgreSQL connection is closed; bounded fetch cannot proceed.",
            transaction_status="closed",
        )

    status = connection.info.transaction_status
    if status == pq.TransactionStatus.IDLE:
        return PostgreSQLBoundedFetchTransactionScope.ADAPTER_OWNED_READONLY
    if status == pq.TransactionStatus.INTRANS:
        return PostgreSQLBoundedFetchTransactionScope.CALLER_OWNED
    if status == pq.TransactionStatus.INERROR:
        raise PostgreSQLBoundedFetchSafetyError(
            "PostgreSQL connection is in failed transaction state (INERROR); "
            "caller must roll back before bounded fetch.",
            transaction_status="INERROR",
        )
    if status == pq.TransactionStatus.ACTIVE:
        raise PostgreSQLBoundedFetchSafetyError(
            "PostgreSQL connection has an active query; bounded fetch rejected.",
            transaction_status="ACTIVE",
        )
    raise PostgreSQLBoundedFetchSafetyError(
        "PostgreSQL transaction status is UNKNOWN; bounded fetch fails closed.",
        transaction_status="UNKNOWN",
    )


def set_local_config(
    session: PostgreSQLSession,
    setting: str,
    value: str,
) -> None:
    """Apply a transaction-local PostgreSQL configuration value via parameterized set_config."""
    session.execute(
        "SELECT set_config(%s, %s, true)",
        (setting, value),
    )


class PostgreSQLSession:
    """Parameterized SQL session over a single PostgreSQL connection."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def execute(self, sql: str, params: Sequence[Any] = ()) -> PostgreSQLExecutionResult:
        cursor = self._connection.execute(sql, params)
        return PostgreSQLExecutionResult(cursor)

    def execute_statement(self, statement: Any) -> PostgreSQLExecutionResult:
        """Execute a driver-composed SQL statement via the platform session boundary."""
        cursor = self._connection.execute(statement)
        return PostgreSQLExecutionResult(cursor)

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()


class PostgreSQLConnectionProvider:
    """
    Platform-owned PostgreSQL connection lifecycle for transactional domain repositories.

    Owns driver loading, connection acquisition, search_path setup, schema identifier safety,
    and explicit transaction boundaries.
    """

    def __init__(
        self,
        config: PostgreSQLIntegrationConfig,
        *,
        connection_factory: Callable[[], Any] | None = None,
        tenant_schema: str | None = None,
    ) -> None:
        self._config = config
        self._connection_factory = connection_factory
        resolved_schema = tenant_schema or config.tenant_schema or "public"
        self._tenant_schema = validate_schema_identifier(resolved_schema)

    @property
    def config(self) -> PostgreSQLIntegrationConfig:
        return self._config

    @property
    def tenant_schema(self) -> str:
        return self._tenant_schema

    def open_configured_connection(self) -> Any:
        """Open a connection with search_path applied and committed. Caller owns lifecycle."""
        connection = self._open_raw_connection()
        self._apply_search_path_on_connection(connection)
        connection.commit()
        return connection

    @contextmanager
    def connection(self) -> Generator[PostgreSQLSession, None, None]:
        connection = self._open_raw_connection()
        session = PostgreSQLSession(connection)
        try:
            self._apply_search_path_on_connection(connection)
            yield session
        finally:
            connection.close()

    @contextmanager
    def transaction(
        self,
        *,
        isolation_level: PostgreSQLIsolationLevel = PostgreSQLIsolationLevel.READ_COMMITTED,
    ) -> Generator[PostgreSQLSession, None, None]:
        with self.connection() as session:
            session.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation_level.value}")
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    def ensure_schema_exists(self, session: PostgreSQLSession, schema_name: str | None = None) -> None:
        schema = validate_schema_identifier(schema_name or self._tenant_schema)
        if schema == "public":
            return
        _, _, _, sql = import_psycopg()
        statement = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        session.execute_statement(statement)

    def _open_raw_connection(self) -> Any:
        if self._connection_factory is not None:
            return self._connection_factory()
        psycopg, _, dict_row, _ = import_psycopg()
        return psycopg.connect(
            self._config.connection_string(),
            row_factory=dict_row,
        )

    def _apply_search_path_on_connection(self, connection: Any) -> None:
        schema = validate_schema_identifier(self._tenant_schema)
        _, _, _, sql = import_psycopg()
        statement = sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema))
        connection.execute(statement)
