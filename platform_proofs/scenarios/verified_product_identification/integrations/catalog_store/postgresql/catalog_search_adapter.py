"""PostgreSQL reference adapters for catalog search ports."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
)

from platform_proofs.scenarios.verified_product_identification.application.catalog.identifier_normalization import (
    normalize_exact_lookup_value,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    LexicalSearchQuery,
    StructuredSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    ExactIdentifierLookupResult,
    LexicalSearchResult,
    SourceRecordFetchResult,
    StructuredSearchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ExactChannelScore,
    LexicalChannelScore,
    ProductCandidate,
    RetrievalChannel,
    StructuredChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    ProductSourceProvenance,
    ProductSourceRecord,
    SourceRecordRef,
)


@dataclass(slots=True)
class _PostgreSQLCatalogSearchBackend:
    _provider: PostgreSQLConnectionProvider
    _catalog_id: str

    @classmethod
    def from_env(cls, *, schema_name: str, catalog_id: str) -> _PostgreSQLCatalogSearchBackend:
        config = PostgreSQLIntegrationConfig.from_env(tenant_schema=schema_name)
        provider = PostgreSQLConnectionProvider(config, tenant_schema=schema_name)
        return cls(_provider=provider, _catalog_id=catalog_id)

    def lookup_exact(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
        identifier = query.identifier
        lookup_value = normalize_exact_lookup_value(identifier.identifier_type, identifier.value)
        with self._provider.connection() as session:
            rows = session.execute(
                """
                SELECT offer_id, source_revision
                FROM vpi_identifier
                WHERE catalog_id = %s
                  AND identifier_type = %s
                  AND lookup_value = %s
                LIMIT %s
                """,
                (
                    self._catalog_id,
                    identifier.identifier_type.value,
                    lookup_value,
                    query.limit,
                ),
            ).fetchall()
        candidates = tuple(
            ProductCandidate(
                offer_id=ProductOfferId(row["offer_id"]),
                channel=RetrievalChannel.EXACT,
                rank=index,
                source_ref=SourceRecordRef(
                    offer_id=ProductOfferId(row["offer_id"]),
                    catalog_id=self._catalog_id,
                    source_revision=row["source_revision"],
                ),
                channel_score=ExactChannelScore(matched_identifier=identifier),
            )
            for index, row in enumerate(rows)
        )
        return ExactIdentifierLookupResult(candidates=candidates)

    def search_lexical(self, query: LexicalSearchQuery) -> LexicalSearchResult:
        pattern = f"%{query.query_text.strip()}%"
        with self._provider.connection() as session:
            rows = session.execute(
                """
                SELECT offer_id, source_revision
                FROM vpi_source_offer
                WHERE catalog_id = %s
                  AND record_json ILIKE %s
                LIMIT %s
                """,
                (self._catalog_id, pattern, query.limit),
            ).fetchall()
        candidates = tuple(
            ProductCandidate(
                offer_id=ProductOfferId(row["offer_id"]),
                channel=RetrievalChannel.LEXICAL,
                rank=index,
                source_ref=SourceRecordRef(
                    offer_id=ProductOfferId(row["offer_id"]),
                    catalog_id=self._catalog_id,
                    source_revision=row["source_revision"],
                ),
                channel_score=LexicalChannelScore(bm25_score=float(query.limit - index)),
            )
            for index, row in enumerate(rows)
        )
        return LexicalSearchResult(candidates=candidates)

    def search_structured(self, query: StructuredSearchQuery) -> StructuredSearchResult:
        constraint = query.constraints[0]
        with self._provider.connection() as session:
            rows = session.execute(
                """
                SELECT offer_id, source_revision
                FROM vpi_structured_attribute
                WHERE catalog_id = %s
                  AND (
                    canonical_key = %s
                    OR source_key = %s
                  )
                  AND (
                    normalized_text_value ILIKE %s
                    OR source_value ILIKE %s
                  )
                LIMIT %s
                """,
                (
                    self._catalog_id,
                    constraint.attribute_name,
                    constraint.attribute_name,
                    f"%{constraint.value.strip()}%",
                    f"%{constraint.value.strip()}%",
                    query.limit,
                ),
            ).fetchall()
        total = len(query.constraints)
        candidates = tuple(
            ProductCandidate(
                offer_id=ProductOfferId(row["offer_id"]),
                channel=RetrievalChannel.STRUCTURED,
                rank=index,
                source_ref=SourceRecordRef(
                    offer_id=ProductOfferId(row["offer_id"]),
                    catalog_id=self._catalog_id,
                    source_revision=row["source_revision"],
                ),
                channel_score=StructuredChannelScore(
                    matched_constraint_count=1,
                    total_constraint_count=total,
                ),
            )
            for index, row in enumerate(rows)
        )
        return StructuredSearchResult(candidates=candidates)

    def fetch(self, source_ref: SourceRecordRef) -> SourceRecordFetchResult:
        with self._provider.connection() as session:
            row = session.execute(
                """
                SELECT record_json, source_revision
                FROM vpi_source_offer
                WHERE catalog_id = %s AND offer_id = %s
                """,
                (source_ref.catalog_id, source_ref.offer_id.value),
            ).fetchone()
        if row is None:
            return SourceRecordFetchResult(record=None, failure=None)
        return SourceRecordFetchResult(
            record=ProductSourceRecord(
                offer_id=source_ref.offer_id,
                record_payload_ref=row["record_json"],
                provenance=ProductSourceProvenance(
                    catalog_id=source_ref.catalog_id,
                    source_revision=row["source_revision"],
                ),
            )
        )

    def close(self) -> None:
        return None


@dataclass(slots=True)
class PostgreSQLExactIdentifierLookupAdapter:
    _backend: _PostgreSQLCatalogSearchBackend

    @classmethod
    def from_env(cls, *, schema_name: str, catalog_id: str) -> PostgreSQLExactIdentifierLookupAdapter:
        return cls(_backend=_PostgreSQLCatalogSearchBackend.from_env(schema_name=schema_name, catalog_id=catalog_id))

    def lookup(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
        return self._backend.lookup_exact(query)

    def close(self) -> None:
        self._backend.close()


@dataclass(slots=True)
class PostgreSQLLexicalSearchAdapter:
    _backend: _PostgreSQLCatalogSearchBackend

    @classmethod
    def from_env(cls, *, schema_name: str, catalog_id: str) -> PostgreSQLLexicalSearchAdapter:
        return cls(_backend=_PostgreSQLCatalogSearchBackend.from_env(schema_name=schema_name, catalog_id=catalog_id))

    def search(self, query: LexicalSearchQuery) -> LexicalSearchResult:
        return self._backend.search_lexical(query)

    def close(self) -> None:
        self._backend.close()


@dataclass(slots=True)
class PostgreSQLStructuredSearchAdapter:
    _backend: _PostgreSQLCatalogSearchBackend

    @classmethod
    def from_env(cls, *, schema_name: str, catalog_id: str) -> PostgreSQLStructuredSearchAdapter:
        return cls(_backend=_PostgreSQLCatalogSearchBackend.from_env(schema_name=schema_name, catalog_id=catalog_id))

    def search(self, query: StructuredSearchQuery) -> StructuredSearchResult:
        return self._backend.search_structured(query)

    def close(self) -> None:
        self._backend.close()


@dataclass(slots=True)
class PostgreSQLSourceRecordFetchAdapter:
    _backend: _PostgreSQLCatalogSearchBackend

    @classmethod
    def from_env(cls, *, schema_name: str, catalog_id: str) -> PostgreSQLSourceRecordFetchAdapter:
        return cls(_backend=_PostgreSQLCatalogSearchBackend.from_env(schema_name=schema_name, catalog_id=catalog_id))

    def fetch(self, source_ref: SourceRecordRef) -> SourceRecordFetchResult:
        return self._backend.fetch(source_ref)

    def close(self) -> None:
        self._backend.close()
