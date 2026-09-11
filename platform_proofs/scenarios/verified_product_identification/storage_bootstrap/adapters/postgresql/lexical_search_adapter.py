"""PostgreSQL BM25 lexical search adapter for indexed Data Pack retrieval."""

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
    LexicalSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    LexicalSearchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    LexicalChannelScore,
    ProductCandidate,
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.bm25_engine import (
    BM25_B,
    BM25_K1,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    DEFAULT_APPLICATION_NAME,
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.lexical_tokenization import (
    tokenize_lexical_document,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    LEXICAL_CORPUS_STATS_SINGLETON_KEY,
    LexicalCorpusStatsTableSpec,
    LexicalDocumentTableSpec,
    LexicalPostingTableSpec,
    LexicalTermStatsTableSpec,
    lexical_bm25_ranked_search_dml,
)


def _map_provider_failure(exc: BaseException) -> CatalogSearchFailure | None:
    _, errors, _, _ = import_psycopg()
    if isinstance(exc, errors.QueryCanceled):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.TIMEOUT,
            message="lexical search timed out",
        )
    if isinstance(exc, errors.OperationalError):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="lexical search unavailable",
        )
    if isinstance(exc, (OSError, ConnectionError)):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="lexical search unavailable",
        )
    return None


@dataclass(frozen=True, slots=True)
class _RankedLexicalRow:
    catalog_id: str
    offer_id: str
    source_revision_norm: str
    source_revision: str | None
    bm25_score: float


def _candidate_from_row(row: _RankedLexicalRow, rank: int) -> ProductCandidate:
    return ProductCandidate(
        offer_id=ProductOfferId(row.offer_id),
        channel=RetrievalChannel.LEXICAL,
        rank=rank,
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(row.offer_id),
            catalog_id=row.catalog_id,
            source_revision=row.source_revision,
        ),
        channel_score=LexicalChannelScore(bm25_score=row.bm25_score),
    )


@dataclass(slots=True)
class PostgreSqlLexicalCandidateSearchAdapter:
    """``LexicalCandidateSearchPort`` over indexed BM25 lexical postings."""

    _provider: PostgreSQLConnectionProvider
    _configuration: PostgreSqlBootstrapConfiguration

    @classmethod
    def from_env(
        cls,
        *,
        schema_name: str,
        lexical_document_table_name: str | None = None,
        lexical_posting_table_name: str | None = None,
        statement_timeout_ms: int | None = None,
        application_name: str = DEFAULT_APPLICATION_NAME,
    ) -> PostgreSqlLexicalCandidateSearchAdapter:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
            DEFAULT_LEXICAL_DOCUMENT_TABLE_NAME,
            DEFAULT_LEXICAL_POSTING_TABLE_NAME,
        )

        configuration = PostgreSqlBootstrapConfiguration.from_env(
            schema_name=schema_name,
            lexical_document_table_name=(
                lexical_document_table_name or DEFAULT_LEXICAL_DOCUMENT_TABLE_NAME
            ),
            lexical_posting_table_name=(
                lexical_posting_table_name or DEFAULT_LEXICAL_POSTING_TABLE_NAME
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
    ) -> PostgreSqlLexicalCandidateSearchAdapter:
        provider = PostgreSQLConnectionProvider(
            configuration.integration,
            tenant_schema=configuration.schema_name,
        )
        return cls(_provider=provider, _configuration=configuration)

    def search(self, query: LexicalSearchQuery) -> LexicalSearchResult:
        query_terms = tokenize_lexical_document(query.query_text)
        if not query_terms:
            return LexicalSearchResult(
                candidates=(),
                failure=CatalogSearchFailure(
                    kind=CatalogSearchFailureKind.INVALID_QUERY,
                    message="lexical query produced no searchable tokens",
                ),
            )

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
        _, errors, _, _ = import_psycopg()
        try:
            with self._provider.connection() as session:
                self._apply_session_limits(session)
                ranked_rows = self._fetch_ranked_rows(
                    session,
                    document_spec=document_spec,
                    posting_spec=posting_spec,
                    corpus_stats_spec=corpus_stats_spec,
                    term_stats_spec=term_stats_spec,
                    query_terms=query_terms,
                    limit=query.limit,
                )
        except (OSError, ConnectionError, errors.QueryCanceled, errors.OperationalError) as exc:
            mapped = _map_provider_failure(exc)
            if mapped is not None:
                return LexicalSearchResult(candidates=(), failure=mapped)
            raise

        candidates = tuple(
            _candidate_from_row(row, index) for index, row in enumerate(ranked_rows)
        )
        return LexicalSearchResult(candidates=candidates)

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

    def _fetch_ranked_rows(
        self,
        session: PostgreSQLSession,
        *,
        document_spec: LexicalDocumentTableSpec,
        posting_spec: LexicalPostingTableSpec,
        corpus_stats_spec: LexicalCorpusStatsTableSpec,
        term_stats_spec: LexicalTermStatsTableSpec,
        query_terms: tuple[str, ...],
        limit: int,
    ) -> tuple[_RankedLexicalRow, ...]:
        rows = session.execute(
            lexical_bm25_ranked_search_dml(
                document_spec,
                posting_spec,
                corpus_stats_spec,
                term_stats_spec,
                k1=BM25_K1,
                b=BM25_B,
            ),
            (list(query_terms), LEXICAL_CORPUS_STATS_SINGLETON_KEY, limit),
        ).fetchall()
        return tuple(
            _RankedLexicalRow(
                catalog_id=str(row["catalog_id"]),
                offer_id=str(row["offer_id"]),
                source_revision_norm=str(row["source_revision_norm"]),
                source_revision=(
                    str(row["source_revision"])
                    if row["source_revision"] is not None
                    else None
                ),
                bm25_score=float(row["bm25_score"]),
            )
            for row in rows
        )
