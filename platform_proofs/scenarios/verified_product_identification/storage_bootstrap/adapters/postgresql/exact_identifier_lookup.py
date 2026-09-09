"""PostgreSQL exact identifier lookup adapter for indexed Data Pack retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLSession,
    import_psycopg,
    set_local_config,
)

from platform_proofs.scenarios.verified_product_identification.application.catalog.identifier_normalization import (
    normalize_exact_lookup_value,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    ExactIdentifierLookupResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ExactChannelScore,
    ProductCandidate,
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    DEFAULT_APPLICATION_NAME,
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.stored_identifier_row import (
    StoredIdentifierRow,
    stored_identifier_row_from_fetched_row,
)


def _map_provider_failure(exc: BaseException) -> CatalogSearchFailure | None:
    _, errors, _, _ = import_psycopg()
    if isinstance(exc, errors.QueryCanceled):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.TIMEOUT,
            message="exact identifier lookup timed out",
        )
    if isinstance(exc, errors.OperationalError):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="exact identifier lookup unavailable",
        )
    if isinstance(exc, (OSError, ConnectionError)):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="exact identifier lookup unavailable",
        )
    return None


def _candidate_from_row(
    row: StoredIdentifierRow,
    *,
    rank: int,
) -> ProductCandidate:
    matched_identifier = ProductIdentifier(
        identifier_type=row.identifier_type,
        value=row.source_value,
        source_field=row.source_field,
    )
    source_revision = row.source_revision if row.source_revision is not None else None
    return ProductCandidate(
        offer_id=ProductOfferId(row.offer_id),
        channel=RetrievalChannel.EXACT,
        rank=rank,
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(row.offer_id),
            catalog_id=row.catalog_id,
            source_revision=source_revision,
        ),
        channel_score=ExactChannelScore(matched_identifier=matched_identifier),
    )


@dataclass(slots=True)
class PostgreSqlExactIdentifierLookupAdapter:
    """``ExactIdentifierLookupPort`` over indexed ``vpi_product_identifiers`` rows."""

    _provider: PostgreSQLConnectionProvider
    _configuration: PostgreSqlBootstrapConfiguration

    @classmethod
    def from_env(
        cls,
        *,
        schema_name: str,
        identifier_table_name: str | None = None,
        statement_timeout_ms: int | None = None,
        application_name: str = DEFAULT_APPLICATION_NAME,
    ) -> PostgreSqlExactIdentifierLookupAdapter:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
            DEFAULT_IDENTIFIER_TABLE_NAME,
        )

        configuration = PostgreSqlBootstrapConfiguration.from_env(
            schema_name=schema_name,
            identifier_table_name=identifier_table_name or DEFAULT_IDENTIFIER_TABLE_NAME,
            statement_timeout_ms=statement_timeout_ms,
            application_name=application_name,
        )
        provider = PostgreSQLConnectionProvider(
            configuration.integration,
            tenant_schema=configuration.schema_name,
        )
        return cls(_provider=provider, _configuration=configuration)

    @classmethod
    def from_configuration(
        cls,
        configuration: PostgreSqlBootstrapConfiguration,
    ) -> PostgreSqlExactIdentifierLookupAdapter:
        provider = PostgreSQLConnectionProvider(
            configuration.integration,
            tenant_schema=configuration.schema_name,
        )
        return cls(_provider=provider, _configuration=configuration)

    def lookup(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
        identifier = query.identifier
        normalized_value = normalize_exact_lookup_value(
            identifier.identifier_type,
            identifier.value,
        )
        if not normalized_value:
            return ExactIdentifierLookupResult(
                candidates=(),
                failure=CatalogSearchFailure(
                    kind=CatalogSearchFailureKind.INVALID_QUERY,
                    message="identifier value cannot be normalized for exact lookup",
                ),
            )

        lookup_sql = (
            f"SELECT catalog_id, offer_id, source_revision_norm, source_revision, "
            f"identifier_type, source_value, normalized_value, source_field "
            f"FROM {self._configuration.identifier_table_name} "
            f"WHERE identifier_type = %s AND normalized_value = %s "
            f"ORDER BY catalog_id ASC, offer_id ASC, source_revision_norm ASC "
            f"LIMIT %s"
        )
        params = (
            identifier.identifier_type.value,
            normalized_value,
            query.limit,
        )
        _, errors, _, _ = import_psycopg()
        try:
            with self._provider.connection() as session:
                self._apply_session_limits(session)
                rows = session.execute(lookup_sql, params).fetchall()
        except (OSError, ConnectionError, errors.QueryCanceled, errors.OperationalError) as exc:
            mapped = _map_provider_failure(exc)
            if mapped is not None:
                return ExactIdentifierLookupResult(candidates=(), failure=mapped)
            raise

        candidates = tuple(
            _candidate_from_row(
                stored_identifier_row_from_fetched_row(row),
                rank=index,
            )
            for index, row in enumerate(rows)
        )
        return ExactIdentifierLookupResult(candidates=candidates)

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
