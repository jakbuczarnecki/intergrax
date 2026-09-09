"""PostgreSQL structured attribute search adapter for indexed Data Pack retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLSession,
    import_psycopg,
    set_local_config,
)

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    StructuredSearchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ProductCandidate,
    RetrievalChannel,
    StructuredChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    DEFAULT_APPLICATION_NAME,
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    StructuredAttributeTableSpec,
    structured_contains_capability_available,
    structured_constraint_search_dml,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.stored_structured_row import (
    StoredStructuredCandidateRow,
    stored_structured_candidate_row_from_fetched_row,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.structured_query import (
    PreparedStructuredSearchQuery,
    prepare_structured_search_query,
)


def _map_provider_failure(exc: BaseException) -> CatalogSearchFailure | None:
    _, errors, _, _ = import_psycopg()
    if isinstance(exc, errors.QueryCanceled):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.TIMEOUT,
            message="structured search timed out",
        )
    if isinstance(exc, errors.OperationalError):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="structured search unavailable",
        )
    if isinstance(exc, (OSError, ConnectionError)):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="structured search unavailable",
        )
    return None


def _candidate_from_row(
    row: StoredStructuredCandidateRow,
    *,
    rank: int,
    total_constraint_count: int,
) -> ProductCandidate:
    source_revision = row.source_revision if row.source_revision is not None else None
    return ProductCandidate(
        offer_id=ProductOfferId(row.offer_id),
        channel=RetrievalChannel.STRUCTURED,
        rank=rank,
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(row.offer_id),
            catalog_id=row.catalog_id,
            source_revision=source_revision,
        ),
        channel_score=StructuredChannelScore(
            matched_constraint_count=row.matched_constraint_count,
            total_constraint_count=total_constraint_count,
        ),
    )


def _search_params(prepared: PreparedStructuredSearchQuery) -> tuple[list[int], list[str], list[str], list[str], int]:
    ordinals: list[int] = []
    keys: list[str] = []
    operators: list[str] = []
    values: list[str] = []
    for constraint in prepared.constraints:
        ordinals.append(constraint.constraint_ordinal)
        keys.append(constraint.normalized_key)
        operators.append(constraint.operator.value)
        values.append(constraint.normalized_value)
    return ordinals, keys, operators, values, prepared.limit


@dataclass(slots=True)
class PostgreSqlStructuredCandidateSearchAdapter:
    """``StructuredCandidateSearchPort`` over indexed ``vpi_structured_attribute`` rows."""

    _provider: PostgreSQLConnectionProvider
    _configuration: PostgreSqlBootstrapConfiguration
    _contains_capability_available: bool | None = None

    @classmethod
    def from_env(
        cls,
        *,
        schema_name: str,
        structured_attribute_table_name: str | None = None,
        statement_timeout_ms: int | None = None,
        application_name: str = DEFAULT_APPLICATION_NAME,
    ) -> PostgreSqlStructuredCandidateSearchAdapter:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
            DEFAULT_STRUCTURED_ATTRIBUTE_TABLE_NAME,
        )

        configuration = PostgreSqlBootstrapConfiguration.from_env(
            schema_name=schema_name,
            structured_attribute_table_name=(
                structured_attribute_table_name or DEFAULT_STRUCTURED_ATTRIBUTE_TABLE_NAME
            ),
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
    ) -> PostgreSqlStructuredCandidateSearchAdapter:
        provider = PostgreSQLConnectionProvider(
            configuration.integration,
            tenant_schema=configuration.schema_name,
        )
        return cls(_provider=provider, _configuration=configuration)

    def search(self, query: StructuredSearchQuery) -> StructuredSearchResult:
        prepared = prepare_structured_search_query(query)
        if isinstance(prepared, CatalogSearchFailure):
            return StructuredSearchResult(candidates=(), failure=prepared)

        if prepared.has_contains and not self._resolve_contains_capability():
            return StructuredSearchResult(
                candidates=(),
                failure=CatalogSearchFailure(
                    kind=CatalogSearchFailureKind.INVALID_QUERY,
                    message=(
                        "CONTAINS requires pg_trgm and compatible trigram index"
                    ),
                ),
            )

        table_spec = StructuredAttributeTableSpec(
            schema_name=self._configuration.schema_name,
            table_name=self._configuration.structured_attribute_table_name,
        )
        search_sql = structured_constraint_search_dml(
            table_spec,
            include_contains_branch=prepared.has_contains,
        )
        ordinals, keys, operators, values, limit = _search_params(prepared)
        params = (ordinals, keys, operators, values, limit)
        _, errors, _, _ = import_psycopg()
        try:
            with self._provider.connection() as session:
                self._apply_session_limits(session)
                rows = session.execute(search_sql, params).fetchall()
        except (OSError, ConnectionError, errors.QueryCanceled, errors.OperationalError) as exc:
            mapped = _map_provider_failure(exc)
            if mapped is not None:
                return StructuredSearchResult(candidates=(), failure=mapped)
            raise

        candidates = tuple(
            _candidate_from_row(
                stored_structured_candidate_row_from_fetched_row(row),
                rank=index,
                total_constraint_count=prepared.total_constraint_count,
            )
            for index, row in enumerate(rows)
        )
        return StructuredSearchResult(candidates=candidates)

    def _resolve_contains_capability(self) -> bool:
        if self._contains_capability_available is not None:
            return self._contains_capability_available
        table_spec = StructuredAttributeTableSpec(
            schema_name=self._configuration.schema_name,
            table_name=self._configuration.structured_attribute_table_name,
        )
        try:
            with self._provider.connection() as session:
                self._apply_session_limits(session)
                available = structured_contains_capability_available(session, table_spec)
        except (OSError, ConnectionError):
            available = False
        self._contains_capability_available = available
        return available

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
