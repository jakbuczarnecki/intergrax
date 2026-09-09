"""PostgreSQL relational storage bootstrap adapter for Data Pack load."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLIsolationLevel,
    PostgreSQLSession,
    set_local_config,
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
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.identifier_projection import (
    ProjectedIdentifierRow,
    project_identifiers_from_load_record,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.lexical_projection import (
    project_lexical_from_load_record,
    project_lexical_postings,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    IdentifierTableSpec,
    LEXICAL_CORPUS_STATS_SINGLETON_KEY,
    LEXICAL_STATISTICS_VERSION,
    LexicalCorpusStatsTableSpec,
    LexicalDocumentTableSpec,
    LexicalPostingTableSpec,
    LexicalTermStatsTableSpec,
    RelationalTableSpec,
    create_identifier_lookup_index_ddl,
    create_identifier_table_ddl,
    create_lexical_corpus_stats_table_ddl,
    create_lexical_document_table_ddl,
    create_lexical_posting_lookup_index_ddl,
    create_lexical_posting_table_ddl,
    create_lexical_term_stats_table_ddl,
    create_structured_attribute_table_ddl,
    create_structured_canonical_equals_index_ddl,
    create_structured_contains_index_ddl,
    create_structured_source_equals_index_ddl,
    create_table_ddl,
    identifier_insert_dml,
    lexical_corpus_stats_increment_dml,
    lexical_document_insert_dml,
    lexical_posting_insert_dml,
    lexical_term_stats_increment_dml,
    pg_trgm_extension_available,
    rebuild_lexical_statistics,
    structured_attribute_insert_dml,
    StructuredAttributeTableSpec,
    verify_identifier_table_compatible,
    verify_lexical_corpus_stats_table_compatible,
    verify_lexical_document_table_compatible,
    verify_lexical_posting_table_compatible,
    verify_lexical_term_stats_table_compatible,
    verify_structured_attribute_table_compatible,
    verify_table_compatible,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.stored_row import (
    StoredRelationalRow,
    stored_relational_row_from_fetched_row,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.structured_projection import (
    project_structured_from_load_record,
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
_IdentifierInsertParams = tuple[str, str, str, str | None, str, str, str, str]
_StructuredInsertParams = tuple[str, str, str, str | None, str, str | None, str, str, str, str | None, str]
_LexicalDocumentInsertParams = tuple[str, str, str, str | None, str, str, int, str]
_LexicalPostingInsertParams = tuple[str, str, str, str, int]


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
        identifier_spec = IdentifierTableSpec(
            schema_name=physical.schema_name,
            table_name=self._configuration.identifier_table_name,
        )
        lexical_document_spec = LexicalDocumentTableSpec(
            schema_name=physical.schema_name,
            table_name=self._configuration.lexical_document_table_name,
        )
        lexical_posting_spec = LexicalPostingTableSpec(
            schema_name=physical.schema_name,
            table_name=self._configuration.lexical_posting_table_name,
        )
        corpus_stats_spec = LexicalCorpusStatsTableSpec(
            schema_name=physical.schema_name,
            table_name=self._configuration.lexical_corpus_stats_table_name,
        )
        term_stats_spec = LexicalTermStatsTableSpec(
            schema_name=physical.schema_name,
            table_name=self._configuration.lexical_term_stats_table_name,
        )
        structured_spec = StructuredAttributeTableSpec(
            schema_name=physical.schema_name,
            table_name=self._configuration.structured_attribute_table_name,
        )
        try:
            with self._provider.transaction(
                isolation_level=PostgreSQLIsolationLevel.READ_COMMITTED,
            ) as session:
                self._apply_session_limits(session)
                self._provider.ensure_schema_exists(session, physical.schema_name)
                session.execute_statement(create_table_ddl(spec))
                verify_table_compatible(session, spec)
                session.execute_statement(create_identifier_table_ddl(identifier_spec))
                session.execute_statement(create_identifier_lookup_index_ddl(identifier_spec))
                verify_identifier_table_compatible(session, identifier_spec)
                session.execute_statement(create_lexical_document_table_ddl(lexical_document_spec))
                verify_lexical_document_table_compatible(session, lexical_document_spec)
                session.execute_statement(create_lexical_posting_table_ddl(lexical_posting_spec))
                session.execute_statement(
                    create_lexical_posting_lookup_index_ddl(lexical_posting_spec)
                )
                verify_lexical_posting_table_compatible(session, lexical_posting_spec)
                session.execute_statement(
                    create_lexical_corpus_stats_table_ddl(corpus_stats_spec)
                )
                verify_lexical_corpus_stats_table_compatible(session, corpus_stats_spec)
                session.execute_statement(
                    create_lexical_term_stats_table_ddl(term_stats_spec)
                )
                verify_lexical_term_stats_table_compatible(session, term_stats_spec)
                session.execute_statement(create_structured_attribute_table_ddl(structured_spec))
                session.execute_statement(
                    create_structured_canonical_equals_index_ddl(structured_spec)
                )
                session.execute_statement(create_structured_source_equals_index_ddl(structured_spec))
                contains_available = pg_trgm_extension_available(session)
                if contains_available:
                    session.execute_statement(create_structured_contains_index_ddl(structured_spec))
                verify_structured_attribute_table_compatible(
                    session,
                    structured_spec,
                    require_contains_index=contains_available,
                )
                rebuild_lexical_statistics(
                    session,
                    document_spec=lexical_document_spec,
                    posting_spec=lexical_posting_spec,
                    corpus_stats_spec=corpus_stats_spec,
                    term_stats_spec=term_stats_spec,
                )
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
            set_local_config(
                session,
                "statement_timeout",
                str(self._configuration.statement_timeout_ms),
            )
        if self._configuration.application_name:
            set_local_config(
                session,
                "application_name",
                self._configuration.application_name,
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
            "ON CONFLICT DO NOTHING"
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
            self._write_identifier_rows(
                session,
                physical,
                project_identifiers_from_load_record(record),
            )
            self._write_structured_rows(session, record)
            self._write_lexical_rows(session, record)
            return "written"

        existing = self._fetch_by_source_identity(session, physical, record.source_ref)
        if existing is not None:
            if _record_payload_matches(existing, record):
                return "skipped"
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
            return "skipped"

        raise PostgreSqlBootstrapOperationError("insert conflict without existing row")

    def _execute_insert(
        self,
        session: PostgreSQLSession,
        insert_sql: str,
        params: _InsertSqlParams,
        physical: PhysicalRelationalTarget,
        record: RelationalLoadRecord,
    ) -> int:
        return session.execute(insert_sql, params).rowcount

    def _write_identifier_rows(
        self,
        session: PostgreSQLSession,
        physical: PhysicalRelationalTarget,
        rows: tuple[ProjectedIdentifierRow, ...],
    ) -> None:
        if not rows:
            return
        insert_sql = identifier_insert_dml(
            IdentifierTableSpec(
                schema_name=self._configuration.schema_name,
                table_name=self._configuration.identifier_table_name,
            )
        )
        for row in rows:
            params: _IdentifierInsertParams = (
                row.catalog_id,
                row.offer_id,
                row.source_revision_norm,
                row.source_revision,
                row.identifier_type.value,
                row.source_value,
                row.normalized_value,
                row.source_field,
            )
            session.execute(insert_sql, params)

    def _write_structured_rows(
        self,
        session: PostgreSQLSession,
        record: RelationalLoadRecord,
    ) -> None:
        rows = project_structured_from_load_record(record)
        if not rows:
            return
        insert_sql = structured_attribute_insert_dml(
            StructuredAttributeTableSpec(
                schema_name=self._configuration.schema_name,
                table_name=self._configuration.structured_attribute_table_name,
            )
        )
        for row in rows:
            params: _StructuredInsertParams = (
                row.catalog_id,
                row.offer_id,
                row.source_revision_norm,
                row.source_revision,
                row.attr_identity,
                row.canonical_key,
                row.source_key,
                row.source_value,
                row.normalized_text_value,
                row.typed_value_text,
                row.source_field,
            )
            session.execute(insert_sql, params)

    def _write_lexical_rows(
        self,
        session: PostgreSQLSession,
        record: RelationalLoadRecord,
    ) -> None:
        projection = project_lexical_from_load_record(record)
        if projection is None:
            return

        document_spec = LexicalDocumentTableSpec(
            schema_name=self._configuration.schema_name,
            table_name=self._configuration.lexical_document_table_name,
        )
        posting_spec = LexicalPostingTableSpec(
            schema_name=self._configuration.schema_name,
            table_name=self._configuration.lexical_posting_table_name,
        )
        corpus_stats_spec = LexicalCorpusStatsTableSpec(
            schema_name=self._configuration.schema_name,
            table_name=self._configuration.lexical_corpus_stats_table_name,
        )
        term_stats_spec = LexicalTermStatsTableSpec(
            schema_name=self._configuration.schema_name,
            table_name=self._configuration.lexical_term_stats_table_name,
        )
        revision_norm = _source_revision_norm(record.source_ref.source_revision)
        document_params: _LexicalDocumentInsertParams = (
            record.source_ref.catalog_id,
            record.source_ref.offer_id.value,
            revision_norm,
            record.source_ref.source_revision,
            projection.lexical_document,
            projection.document_hash,
            projection.document_length,
            projection.derivation_version,
        )
        document_inserted = (
            session.execute(lexical_document_insert_dml(document_spec), document_params).rowcount
            > 0
        )
        if document_inserted:
            session.execute(
                lexical_corpus_stats_increment_dml(corpus_stats_spec),
                (
                    LEXICAL_CORPUS_STATS_SINGLETON_KEY,
                    LEXICAL_STATISTICS_VERSION,
                    projection.document_length,
                    float(projection.document_length),
                ),
            )
        posting_insert_sql = lexical_posting_insert_dml(posting_spec)
        term_stats_increment_sql = lexical_term_stats_increment_dml(term_stats_spec)
        for posting in project_lexical_postings(projection):
            posting_params: _LexicalPostingInsertParams = (
                posting.term,
                posting.catalog_id,
                posting.offer_id,
                posting.source_revision_norm,
                posting.term_frequency,
            )
            if session.execute(posting_insert_sql, posting_params).rowcount > 0:
                session.execute(term_stats_increment_sql, (posting.term,))

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
