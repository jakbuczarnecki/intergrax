"""PgVector connection lifecycle — reuses platform psycopg session primitives."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from intergrax.integrations._shared.p2.configs import SqlIntegrationConfig
from intergrax.integrations.providers.relational_store.postgresql.config import (
    validate_schema_identifier,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLIsolationLevel,
    PostgreSQLSession,
    import_psycopg,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.errors import (
    PgVectorBootstrapOperationError,
    PgVectorBootstrapSchemaError,
)


class PgVectorDriverConnection(Protocol):
    def execute(self, statement: str | object, params: tuple[object, ...] = ()): ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


def register_pgvector_types(connection: PgVectorDriverConnection) -> None:
    try:
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise PgVectorBootstrapSchemaError(
            "pgvector python adapter is unavailable"
        ) from exc
    register_vector(connection)


@dataclass(slots=True)
class PgVectorConnectionProvider:
    """Platform-owned pgvector connection lifecycle for bootstrap adapter writes."""

    sql_integration: SqlIntegrationConfig
    schema_name: str
    connection_factory: Callable[[], PgVectorDriverConnection] | None = None

    @contextmanager
    def connection(self) -> Generator[PostgreSQLSession, None, None]:
        connection = self._open_raw_connection()
        session = PostgreSQLSession(connection)
        try:
            self._apply_search_path(connection)
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
        schema = validate_schema_identifier(schema_name or self.schema_name)
        if schema == "public":
            return
        _, _, _, sql = import_psycopg()
        statement = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        session.execute_statement(statement)

    def _open_raw_connection(self) -> PgVectorDriverConnection:
        if self.connection_factory is not None:
            return self.connection_factory()
        psycopg, _, dict_row, _ = import_psycopg()
        try:
            connection = psycopg.connect(
                self.sql_integration.connection_dsn(),
                row_factory=dict_row,
            )
        except (OSError, ValueError) as exc:
            raise PgVectorBootstrapOperationError("pgvector connection failed") from exc
        try:
            register_pgvector_types(connection)
        except PgVectorBootstrapSchemaError:
            connection.close()
            raise
        return connection

    def _apply_search_path(self, connection: PgVectorDriverConnection) -> None:
        schema = validate_schema_identifier(self.schema_name)
        _, _, _, sql = import_psycopg()
        statement = sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema))
        connection.execute(statement)
