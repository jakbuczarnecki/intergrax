"""PgVector vector storage bootstrap adapter for Data Pack load."""

from __future__ import annotations

import math
from collections.abc import Sequence

from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLIsolationLevel,
    PostgreSQLSession,
    import_psycopg,
    is_postgresql_unique_violation,
    set_local_config,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.configuration import (
    ExpectedVectorIdentity,
    PgVectorBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.connection import (
    PgVectorConnectionProvider,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.errors import (
    PgVectorBootstrapConfigurationError,
    PgVectorBootstrapIdentityConflictError,
    PgVectorBootstrapOperationError,
    PgVectorBootstrapSchemaError,
    PgVectorBootstrapVectorValidationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.schema import (
    PgVectorTableSpec,
    create_table_ddl,
    ensure_pgvector_extension,
    verify_table_compatible,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.stored_row import (
    StoredPgVectorRow,
    embedding_identity_matches,
    normalize_vector_float32,
    record_matches_stored,
    stored_pgvector_row_from_fetched_row,
    stored_row_from_record,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.target_mapping import (
    PhysicalPgVectorTarget,
    resolve_physical_target,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    StorageLoadBatchResult,
    VectorBatch,
    VectorLoadRecord,
    VectorTargetId,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapIntegrityError,
    StorageBootstrapWriteError,
)

_InsertSqlParams = tuple[
    str,
    str,
    str,
    str,
    str | None,
    str,
    str,
    str,
    str | None,
    int,
    str,
    list[float],
]


def _sanitize_provider_error(exc: BaseException) -> str:
    return f"pgvector operation failed: {type(exc).__name__}"


def _validate_vector_record(
    record: VectorLoadRecord,
    expected_identity: ExpectedVectorIdentity,
) -> None:
    if not embedding_identity_matches(record, expected_identity):
        raise PgVectorBootstrapVectorValidationError("VECTOR_IDENTITY_INCOMPATIBLE")
    if len(record.dense_embedding) != expected_identity.dimension:
        raise PgVectorBootstrapVectorValidationError("vector dimension mismatch")
    for value in record.dense_embedding:
        if not math.isfinite(value):
            raise PgVectorBootstrapVectorValidationError("non-finite vector value")
    norm = math.sqrt(sum(value * value for value in record.dense_embedding))
    if norm <= 0.0:
        raise PgVectorBootstrapVectorValidationError("zero vector rejected")


class PgVectorStorageAdapter:
    """``VectorStorageLoadPort`` implementation over platform pgvector primitives."""

    def __init__(
        self,
        provider: PgVectorConnectionProvider,
        configuration: PgVectorBootstrapConfiguration,
        *,
        prepared_targets: set[str] | None = None,
    ) -> None:
        self._provider = provider
        self._configuration = configuration
        self._prepared_targets = prepared_targets if prepared_targets is not None else set()

    @classmethod
    def from_env(
        cls,
        *,
        schema_name: str,
        table_name: str | None = None,
        allow_create_extension: bool = True,
    ) -> PgVectorStorageAdapter:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.configuration import (
            DEFAULT_LOGICAL_TABLE_NAME,
        )

        resolved_table = table_name or DEFAULT_LOGICAL_TABLE_NAME
        configuration = PgVectorBootstrapConfiguration.from_env(
            schema_name=schema_name,
            table_name=resolved_table,
            allow_create_extension=allow_create_extension,
        )
        provider = PgVectorConnectionProvider(
            sql_integration=configuration.sql_integration,
            schema_name=configuration.schema_name,
        )
        return cls(provider=provider, configuration=configuration)

    def prepare_target(self, logical_target: VectorTargetId) -> None:
        physical = resolve_physical_target(logical_target, self._configuration)
        spec = PgVectorTableSpec(
            schema_name=physical.schema_name,
            table_name=physical.table_name,
            dimension=self._configuration.expected_vector_identity.dimension,
        )
        try:
            with self._provider.transaction(
                isolation_level=PostgreSQLIsolationLevel.READ_COMMITTED,
            ) as session:
                self._apply_session_limits(session)
                self._provider.ensure_schema_exists(session, physical.schema_name)
                ensure_pgvector_extension(
                    session,
                    allow_create=self._configuration.allow_create_extension,
                )
                session.execute_statement(create_table_ddl(spec))
                verify_table_compatible(session, spec)
        except PgVectorBootstrapSchemaError:
            raise
        except PgVectorBootstrapConfigurationError:
            raise
        except (OSError, ValueError) as exc:
            raise PgVectorBootstrapOperationError("pgvector prepare_target failed") from exc
        self._prepared_targets.add(str(logical_target))

    def write_batch(self, batch: VectorBatch) -> StorageLoadBatchResult:
        physical = resolve_physical_target(batch.target, self._configuration)
        requested = len(batch.records)
        if requested == 0:
            return StorageLoadBatchResult(
                requested_count=0,
                written_count=0,
                updated_count=0,
                skipped_count=0,
                failed_count=0,
            )
        if str(batch.target) not in self._prepared_targets:
            self.prepare_target(batch.target)

        tolerance = self._configuration.vector_transport_tolerance
        expected_identity = self._configuration.expected_vector_identity

        written_count = 0
        skipped_count = 0
        first_failed_identity: str | None = None

        try:
            with self._provider.transaction(
                isolation_level=PostgreSQLIsolationLevel.READ_COMMITTED,
            ) as session:
                self._apply_session_limits(session)
                for record in batch.records:
                    _validate_vector_record(record, expected_identity)

                existing_by_logical = self._fetch_existing_rows(
                    session,
                    physical,
                    [record.logical_point_id for record in batch.records],
                )

                for record in batch.records:
                    stored = existing_by_logical.get(record.logical_point_id)
                    if stored is None:
                        self._insert_record(session, physical, record)
                        written_count += 1
                        continue
                    if record_matches_stored(record, stored, tolerance=tolerance):
                        skipped_count += 1
                        continue
                    first_failed_identity = record.logical_point_id
                    raise PgVectorBootstrapIdentityConflictError(
                        f"VECTOR_CONTENT_CONFLICT: {record.logical_point_id}"
                    )

                verify_result = self._verify_records_in_session(
                    session,
                    physical,
                    batch.records,
                    tolerance=tolerance,
                )
                if verify_result.failed_count > 0:
                    raise StorageBootstrapIntegrityError(
                        "post-write vector batch verification failed"
                    )
        except PgVectorBootstrapIdentityConflictError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except PgVectorBootstrapVectorValidationError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except PgVectorBootstrapConfigurationError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except PgVectorBootstrapSchemaError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except PgVectorBootstrapOperationError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except StorageBootstrapIntegrityError:
            raise
        except (OSError, ValueError) as exc:
            raise StorageBootstrapWriteError("pgvector vector batch write failed") from exc

        return StorageLoadBatchResult(
            requested_count=requested,
            written_count=written_count,
            updated_count=0,
            skipped_count=skipped_count,
            failed_count=0,
            first_failed_identity=first_failed_identity,
        )

    def verify_batch(self, batch: VectorBatch) -> StorageLoadBatchResult:
        physical = resolve_physical_target(batch.target, self._configuration)
        requested = len(batch.records)
        if requested == 0:
            return StorageLoadBatchResult(
                requested_count=0,
                written_count=0,
                updated_count=0,
                skipped_count=0,
                failed_count=0,
            )
        tolerance = self._configuration.vector_transport_tolerance
        try:
            with self._provider.connection() as session:
                self._apply_session_limits(session)
                return self._verify_records_in_session(
                    session,
                    physical,
                    batch.records,
                    tolerance=tolerance,
                )
        except PgVectorBootstrapOperationError as exc:
            raise StorageBootstrapIntegrityError(str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise StorageBootstrapIntegrityError(
                "pgvector vector batch verification failed"
            ) from exc

    def _apply_session_limits(self, session: PostgreSQLSession) -> None:
        if self._configuration.application_name:
            set_local_config(
                session,
                "application_name",
                self._configuration.application_name,
            )

    def _verify_records_in_session(
        self,
        session: PostgreSQLSession,
        physical: PhysicalPgVectorTarget,
        records: Sequence[VectorLoadRecord],
        *,
        tolerance: float,
    ) -> StorageLoadBatchResult:
        existing_by_logical = self._fetch_existing_rows(
            session,
            physical,
            [record.logical_point_id for record in records],
        )
        verified = 0
        first_failed_identity: str | None = None
        for record in records:
            stored = existing_by_logical.get(record.logical_point_id)
            if stored is not None and record_matches_stored(
                record,
                stored,
                tolerance=tolerance,
            ):
                verified += 1
            elif first_failed_identity is None:
                first_failed_identity = record.logical_point_id
        failed = len(records) - verified
        return StorageLoadBatchResult(
            requested_count=len(records),
            written_count=verified,
            updated_count=0,
            skipped_count=0,
            failed_count=failed,
            first_failed_identity=first_failed_identity,
        )

    def _fetch_existing_rows(
        self,
        session: PostgreSQLSession,
        physical: PhysicalPgVectorTarget,
        logical_point_ids: Sequence[str],
    ) -> dict[str, StoredPgVectorRow]:
        if not logical_point_ids:
            return {}
        rows = session.execute(
            f"""
            SELECT logical_point_id, catalog_id, offer_id, source_revision_norm,
                   source_revision, semantic_text_hash, embedding_provider,
                   embedding_model, embedding_revision, embedding_dimension,
                   derivation_version, dense_embedding
            FROM {physical.table_name}
            WHERE logical_point_id = ANY(%s)
            """,
            (list(logical_point_ids),),
        ).fetchall()
        stored: dict[str, StoredPgVectorRow] = {}
        for row in rows:
            converted = stored_pgvector_row_from_fetched_row(row)
            stored[converted.logical_point_id] = converted
        return stored

    def _insert_record(
        self,
        session: PostgreSQLSession,
        physical: PhysicalPgVectorTarget,
        record: VectorLoadRecord,
    ) -> None:
        row = stored_row_from_record(record)
        params: _InsertSqlParams = (
            row.logical_point_id,
            row.catalog_id,
            row.offer_id,
            row.source_revision_norm,
            row.source_revision,
            row.semantic_text_hash,
            row.embedding_provider,
            row.embedding_model,
            row.embedding_revision,
            row.embedding_dimension,
            row.derivation_version,
            list(normalize_vector_float32(record.dense_embedding)),
        )
        insert_sql = (
            f"INSERT INTO {physical.table_name} ("
            "logical_point_id, catalog_id, offer_id, source_revision_norm, "
            "source_revision, semantic_text_hash, embedding_provider, "
            "embedding_model, embedding_revision, embedding_dimension, "
            "derivation_version, dense_embedding"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        _, pg_errors, _, _ = import_psycopg()
        try:
            session.execute(insert_sql, params)
        except pg_errors.Error as exc:
            if is_postgresql_unique_violation(exc):
                existing = self._fetch_existing_rows(
                    session,
                    physical,
                    [record.logical_point_id],
                ).get(record.logical_point_id)
                if existing is not None and record_matches_stored(
                    record,
                    existing,
                    tolerance=self._configuration.vector_transport_tolerance,
                ):
                    return
                raise PgVectorBootstrapIdentityConflictError(
                    f"VECTOR_CONTENT_CONFLICT: {record.logical_point_id}"
                ) from exc
            raise PgVectorBootstrapOperationError(
                _sanitize_provider_error(exc)
            ) from exc
