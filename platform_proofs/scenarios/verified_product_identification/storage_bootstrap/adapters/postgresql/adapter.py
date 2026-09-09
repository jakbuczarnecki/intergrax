"""PostgreSQL relational storage bootstrap adapter for Data Pack load."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLIsolationLevel,
    PostgreSQLSession,
    import_psycopg,
    is_postgresql_unique_violation,
)

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.errors import (
    PostgreSqlBootstrapConfigurationError,
    PostgreSqlBootstrapIdentityConflictError,
    PostgreSqlBootstrapOperationError,
    PostgreSqlBootstrapSchemaError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    RelationalTableSpec,
    create_table_ddl,
    verify_table_compatible,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.stored_row import (
    StoredRelationalRow,
    stored_relational_row_from_fetched_row,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.target_mapping import (
    PhysicalRelationalTarget,
    resolve_physical_target,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalBatch,
    RelationalLoadRecord,
    RelationalTargetId,
    StorageLoadBatchResult,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapIntegrityError,
    StorageBootstrapWriteError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.mapping import (
    identity_key,
)

_InsertSqlParams = tuple[str, str, str, str | None, int, str, str, str, str]


def _source_revision_norm(source_revision: str | None) -> str:
    return source_revision or ""


def _identity_params(record: RelationalLoadRecord) -> tuple[str, str, str]:
    source_ref = record.source_ref
    return (
        source_ref.catalog_id,
        source_ref.offer_id.value,
        _source_revision_norm(source_ref.source_revision),
    )


def _record_payload_matches(
    existing: StoredRelationalRow,
    record: RelationalLoadRecord,
) -> bool:
    return (
        existing.global_row_index == record.global_row_index
        and existing.semantic_text_hash == record.semantic_text_hash
        and existing.derivation_version == record.derivation_version
        and existing.record_json == record.record_json
        and existing.semantic_text == record.semantic_text
    )


@dataclass(slots=True)
class PostgreSqlRelationalStorageAdapter:
    """``RelationalStorageLoadPort`` implementation over platform PostgreSQL sessions."""

    _provider: PostgreSQLConnectionProvider
    _configuration: PostgreSqlBootstrapConfiguration
    _prepared_targets: set[str]

    @classmethod
    def from_env(
        cls,
        *,
        schema_name: str,
        table_name: str | None = None,
    ) -> PostgreSqlRelationalStorageAdapter:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
            DEFAULT_RELATIONAL_TABLE_NAME,
        )

        resolved_table = table_name or DEFAULT_RELATIONAL_TABLE_NAME
        configuration = PostgreSqlBootstrapConfiguration.from_env(
            schema_name=schema_name,
            table_name=resolved_table,
        )
        provider = PostgreSQLConnectionProvider(
            configuration.integration,
            tenant_schema=configuration.schema_name,
        )
        return cls(
            _provider=provider,
            _configuration=configuration,
            _prepared_targets=set(),
        )

    def prepare_target(self, logical_target: RelationalTargetId) -> None:
        physical = resolve_physical_target(logical_target, self._configuration)
        spec = RelationalTableSpec(
            schema_name=physical.schema_name,
            table_name=physical.table_name,
        )
        try:
            with self._provider.transaction(
                isolation_level=PostgreSQLIsolationLevel.READ_COMMITTED,
            ) as session:
                self._apply_session_limits(session)
                self._provider.ensure_schema_exists(session, physical.schema_name)
                session.execute_statement(create_table_ddl(spec))
                verify_table_compatible(session, spec)
        except PostgreSqlBootstrapSchemaError:
            raise
        except PostgreSqlBootstrapConfigurationError:
            raise
        except (OSError, ValueError) as exc:
            raise PostgreSqlBootstrapOperationError("PostgreSQL prepare_target failed") from exc
        self._prepared_targets.add(str(logical_target))

    def write_batch(self, batch: RelationalBatch) -> StorageLoadBatchResult:
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

        written_count = 0
        skipped_count = 0
        failed_count = 0
        first_failed_identity: str | None = None

        try:
            with self._provider.transaction(
                isolation_level=PostgreSQLIsolationLevel.READ_COMMITTED,
            ) as session:
                self._apply_session_limits(session)
                for record in batch.records:
                    try:
                        outcome = self._write_record(session, physical, record)
                    except PostgreSqlBootstrapIdentityConflictError as exc:
                        failed_count += 1
                        first_failed_identity = identity_key(record.source_ref)
                        raise StorageBootstrapWriteError(
                            f"IDENTITY_CONTENT_CONFLICT: {exc}"
                        ) from exc
                    if outcome == "written":
                        written_count += 1
                    else:
                        skipped_count += 1
        except StorageBootstrapWriteError:
            raise
        except PostgreSqlBootstrapConfigurationError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except PostgreSqlBootstrapOperationError as exc:
            raise StorageBootstrapWriteError(str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise StorageBootstrapWriteError("PostgreSQL relational batch write failed") from exc

        return StorageLoadBatchResult(
            requested_count=requested,
            written_count=written_count,
            updated_count=0,
            skipped_count=skipped_count,
            failed_count=failed_count,
            first_failed_identity=first_failed_identity,
        )

    def verify_batch(self, batch: RelationalBatch) -> StorageLoadBatchResult:
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

        verified = 0
        first_failed_identity: str | None = None
        try:
            with self._provider.connection() as session:
                self._apply_session_limits(session)
                for record in batch.records:
                    if self._verify_record(session, physical, record):
                        verified += 1
                    elif first_failed_identity is None:
                        first_failed_identity = identity_key(record.source_ref)
        except (OSError, ValueError) as exc:
            raise StorageBootstrapIntegrityError(
                "PostgreSQL relational batch verification failed"
            ) from exc

        failed = requested - verified
        return StorageLoadBatchResult(
            requested_count=requested,
            written_count=verified,
            updated_count=0,
            skipped_count=0,
            failed_count=failed,
            first_failed_identity=first_failed_identity,
        )

    def _apply_session_limits(self, session: PostgreSQLSession) -> None:
        if self._configuration.statement_timeout_ms is not None:
            session.execute(
                "SET LOCAL statement_timeout = %s",
                (str(self._configuration.statement_timeout_ms),),
            )
        if self._configuration.application_name:
            session.execute(
                "SET LOCAL application_name = %s",
                (self._configuration.application_name,),
            )

    def _write_record(
        self,
        session: PostgreSQLSession,
        physical: PhysicalRelationalTarget,
        record: RelationalLoadRecord,
    ) -> str:
        catalog_id, offer_id, revision_norm = _identity_params(record)
        insert_sql = (
            f"INSERT INTO {physical.table_name} ("
            "catalog_id, offer_id, source_revision_norm, source_revision, "
            "global_row_index, record_json, semantic_text, semantic_text_hash, "
            "derivation_version"
            ") VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s) "
            "ON CONFLICT (catalog_id, offer_id, source_revision_norm) DO NOTHING"
        )
        params: _InsertSqlParams = (
            catalog_id,
            offer_id,
            revision_norm,
            record.source_ref.source_revision,
            record.global_row_index,
            record.record_json,
            record.semantic_text,
            record.semantic_text_hash,
            record.derivation_version,
        )
        if self._execute_insert(session, insert_sql, params, physical, record) > 0:
            return "written"

        existing = self._fetch_by_source_identity(session, physical, record.source_ref)
        if existing is None:
            raise PostgreSqlBootstrapOperationError(
                "insert conflict without existing row"
            )
        if _record_payload_matches(existing, record):
            return "skipped"
        raise PostgreSqlBootstrapIdentityConflictError(
            f"identity {identity_key(record.source_ref)} has incompatible stored content"
        )

    def _execute_insert(
        self,
        session: PostgreSQLSession,
        insert_sql: str,
        params: _InsertSqlParams,
        physical: PhysicalRelationalTarget,
        record: RelationalLoadRecord,
    ) -> int:
        _, pg_errors, _, _ = import_psycopg()
        try:
            return session.execute(insert_sql, params).rowcount
        except pg_errors.Error as exc:
            if is_postgresql_unique_violation(exc):
                self._raise_identity_conflict_for_unique_violation(
                    session,
                    physical,
                    record,
                )
            raise PostgreSqlBootstrapOperationError(
                "PostgreSQL insert failed"
            ) from exc

    def _raise_identity_conflict_for_unique_violation(
        self,
        session: PostgreSQLSession,
        physical: PhysicalRelationalTarget,
        record: RelationalLoadRecord,
    ) -> None:
        by_identity = self._fetch_by_source_identity(session, physical, record.source_ref)
        if by_identity is not None and not _record_payload_matches(by_identity, record):
            raise PostgreSqlBootstrapIdentityConflictError(
                f"identity {identity_key(record.source_ref)} has incompatible stored content"
            )
        by_row_index = self._fetch_by_global_row_index(
            session,
            physical,
            record.global_row_index,
        )
        if by_row_index is not None:
            existing_identity = (
                by_row_index.catalog_id,
                by_row_index.offer_id,
                by_row_index.source_revision_norm,
            )
            incoming_identity = _identity_params(record)
            if existing_identity != incoming_identity:
                raise PostgreSqlBootstrapIdentityConflictError(
                    f"global_row_index {record.global_row_index} already bound to "
                    f"different source identity"
                )
            if not _record_payload_matches(by_row_index, record):
                raise PostgreSqlBootstrapIdentityConflictError(
                    f"global_row_index {record.global_row_index} has incompatible stored content"
                )
        raise PostgreSqlBootstrapIdentityConflictError(
            f"identity {identity_key(record.source_ref)} conflicts with stored row"
        )

    def _verify_record(
        self,
        session: PostgreSQLSession,
        physical: PhysicalRelationalTarget,
        record: RelationalLoadRecord,
    ) -> bool:
        existing = self._fetch_by_source_identity(session, physical, record.source_ref)
        if existing is None:
            return False
        return (
            existing.global_row_index == record.global_row_index
            and existing.semantic_text_hash == record.semantic_text_hash
        )

    def _fetch_by_source_identity(
        self,
        session: PostgreSQLSession,
        physical: PhysicalRelationalTarget,
        source_ref: SourceRecordRef,
    ) -> StoredRelationalRow | None:
        revision_norm = _source_revision_norm(source_ref.source_revision)
        row = session.execute(
            f"""
            SELECT catalog_id, offer_id, source_revision_norm, source_revision,
                   global_row_index, record_json::text AS record_json,
                   semantic_text, semantic_text_hash, derivation_version
            FROM {physical.table_name}
            WHERE catalog_id = %s AND offer_id = %s AND source_revision_norm = %s
            """,
            (
                source_ref.catalog_id,
                source_ref.offer_id.value,
                revision_norm,
            ),
        ).fetchone()
        if row is None:
            return None
        return stored_relational_row_from_fetched_row(row)

    def _fetch_by_global_row_index(
        self,
        session: PostgreSQLSession,
        physical: PhysicalRelationalTarget,
        global_row_index: int,
    ) -> StoredRelationalRow | None:
        row = session.execute(
            f"""
            SELECT catalog_id, offer_id, source_revision_norm, source_revision,
                   global_row_index, record_json::text AS record_json,
                   semantic_text, semantic_text_hash, derivation_version
            FROM {physical.table_name}
            WHERE global_row_index = %s
            """,
            (global_row_index,),
        ).fetchone()
        if row is None:
            return None
        return stored_relational_row_from_fetched_row(row)
