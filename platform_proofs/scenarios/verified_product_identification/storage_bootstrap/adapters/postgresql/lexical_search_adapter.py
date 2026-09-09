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
    Bm25CorpusStatistics,
    Bm25IndexedDocument,
    rank_bm25_documents,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    DEFAULT_APPLICATION_NAME,
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.lexical_tokenization import (
    tokenize_lexical_document,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    LexicalDocumentTableSpec,
    LexicalPostingTableSpec,
    lexical_corpus_stats_dml,
    lexical_document_lookup_dml,
    lexical_posting_lookup_dml,
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
class _PostingHit:
    term: str
    catalog_id: str
    offer_id: str
    source_revision_norm: str
    term_frequency: int


def _candidate_from_score(
    *,
    score: float,
    rank: int,
    document: Bm25IndexedDocument,
) -> ProductCandidate:
    source_revision = document.source_revision if document.source_revision is not None else None
    return ProductCandidate(
        offer_id=ProductOfferId(document.offer_id),
        channel=RetrievalChannel.LEXICAL,
        rank=rank,
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(document.offer_id),
            catalog_id=document.catalog_id,
            source_revision=source_revision,
        ),
        channel_score=LexicalChannelScore(bm25_score=score),
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
        _, errors, _, _ = import_psycopg()
        try:
            with self._provider.connection() as session:
                self._apply_session_limits(session)
                corpus = self._fetch_corpus_statistics(session, document_spec)
                postings = self._fetch_postings(
                    session,
                    posting_spec,
                    query_terms=query_terms,
                )
                documents = self._fetch_documents_for_postings(
                    session,
                    document_spec,
                    postings=postings,
                )
        except (OSError, ConnectionError, errors.QueryCanceled, errors.OperationalError) as exc:
            mapped = _map_provider_failure(exc)
            if mapped is not None:
                return LexicalSearchResult(candidates=(), failure=mapped)
            raise

        if corpus.document_count <= 0:
            return LexicalSearchResult(candidates=())

        term_document_frequencies = self._term_document_frequencies(postings)
        ranked = rank_bm25_documents(
            query_text=query.query_text,
            documents=documents,
            corpus=corpus,
            term_document_frequencies=term_document_frequencies,
            limit=query.limit,
        )
        document_by_identity = {
            (item.catalog_id, item.offer_id, item.source_revision_norm): item
            for item in documents
        }
        candidates = tuple(
            _candidate_from_score(
                score=scored.bm25_score,
                rank=index,
                document=document_by_identity[
                    (scored.catalog_id, scored.offer_id, scored.source_revision_norm)
                ],
            )
            for index, scored in enumerate(ranked)
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

    def _fetch_corpus_statistics(
        self,
        session: PostgreSQLSession,
        document_spec: LexicalDocumentTableSpec,
    ) -> Bm25CorpusStatistics:
        row = session.execute(lexical_corpus_stats_dml(document_spec)).fetchone()
        if row is None:
            return Bm25CorpusStatistics(document_count=0, average_document_length=0.0)
        return Bm25CorpusStatistics(
            document_count=int(row["document_count"]),
            average_document_length=float(row["average_document_length"]),
        )

    def _fetch_postings(
        self,
        session: PostgreSQLSession,
        posting_spec: LexicalPostingTableSpec,
        *,
        query_terms: tuple[str, ...],
    ) -> tuple[_PostingHit, ...]:
        rows = session.execute(
            lexical_posting_lookup_dml(posting_spec),
            (list(query_terms),),
        ).fetchall()
        return tuple(
            _PostingHit(
                term=str(row["term"]),
                catalog_id=str(row["catalog_id"]),
                offer_id=str(row["offer_id"]),
                source_revision_norm=str(row["source_revision_norm"]),
                term_frequency=int(row["term_frequency"]),
            )
            for row in rows
        )

    def _fetch_documents_for_postings(
        self,
        session: PostgreSQLSession,
        document_spec: LexicalDocumentTableSpec,
        *,
        postings: tuple[_PostingHit, ...],
    ) -> tuple[Bm25IndexedDocument, ...]:
        grouped: dict[tuple[str, str, str], dict[str, int]] = {}
        for posting in postings:
            identity = (posting.catalog_id, posting.offer_id, posting.source_revision_norm)
            term_map = grouped.setdefault(identity, {})
            term_map[posting.term] = posting.term_frequency

        documents: list[Bm25IndexedDocument] = []
        for (catalog_id, offer_id, revision_norm), term_frequencies in grouped.items():
            row = session.execute(
                lexical_document_lookup_dml(document_spec),
                (catalog_id, offer_id, revision_norm),
            ).fetchone()
            if row is None:
                continue
            source_revision_raw = row["source_revision"]
            source_revision = (
                str(source_revision_raw)
                if source_revision_raw is not None
                else None
            )
            documents.append(
                Bm25IndexedDocument(
                    catalog_id=catalog_id,
                    offer_id=offer_id,
                    source_revision_norm=revision_norm,
                    source_revision=source_revision,
                    document_length=int(row["document_length"]),
                    term_frequencies=term_frequencies,
                )
            )
        return tuple(documents)

    @staticmethod
    def _term_document_frequencies(
        postings: tuple[_PostingHit, ...],
    ) -> dict[str, int]:
        frequencies: dict[str, set[tuple[str, str, str]]] = {}
        for posting in postings:
            identity = (posting.catalog_id, posting.offer_id, posting.source_revision_norm)
            bucket = frequencies.setdefault(posting.term, set())
            bucket.add(identity)
        return {term: len(identities) for term, identities in frequencies.items()}
