"""Unit tests for PostgreSQL BM25 lexical search adapter and engine."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

import pytest

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    import_psycopg,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    LexicalSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    LexicalCandidateSearchPort,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval import (
    MultiChannelRetrievalRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.service import (
    MultiChannelRetrievalService,
)
from platform_proofs.scenarios.verified_product_identification.retrieval.composition import (
    build_lexical_candidate_search,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.bm25_engine import (
    Bm25CorpusStatistics,
    Bm25IndexedDocument,
    Bm25EngineConfiguration,
    build_term_frequencies,
    compute_bm25_score,
    rank_bm25_documents,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.lexical_search_adapter import (
    PostgreSqlLexicalCandidateSearchAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.lexical_tokenization import (
    tokenize_lexical_document,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    LexicalCorpusStatsTableSpec,
    LexicalDocumentTableSpec,
    LexicalPostingTableSpec,
    LexicalTermStatsTableSpec,
    _LEXICAL_POSTING_LOOKUP_INDEX_NAME,
    lexical_bm25_ranked_search_dml,
    lexical_posting_lookup_dml,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalLoadRecord,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_catalog_contracts import (
    FakeExactLookupA,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_multi_channel_retrieval import (
    RecordingStructuredSearch,
    RecordingVectorSearch,
)

pytestmark = pytest.mark.unit

_, _, _, _PSYCOPG_SQL = import_psycopg()

SqlParam = str | int | None | list[str]
SqlParams = tuple[SqlParam, ...]
SqlStatement = str | _PSYCOPG_SQL.Composable
ExecutedStatement = tuple[SqlStatement, SqlParams]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_ADAPTER_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/adapters/postgresql"
)
_LEGACY_ADAPTER_PATH = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/integrations/catalog_store/postgresql/catalog_search_adapter.py"
)
_APPLICATION_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/application"
)
_CATALOG_ID = "vpi-lexical-catalog"


def _executed_sql_text(statement: SqlStatement) -> str:
    if isinstance(statement, str):
        return statement
    return statement.as_string(None)


def _ranked_search_executions(connection: _RoutingFakeConnection) -> list[ExecutedStatement]:
    return [
        executed
        for executed in connection.executed
        if "term_contributions" in _executed_sql_text(executed[0]).lower()
    ]


def _configuration() -> PostgreSqlBootstrapConfiguration:
    integration = PostgreSQLIntegrationConfig(
        host="localhost",
        port=5432,
        user="intergrax",
        password="secret-value",
        database="intergrax",
        tenant_schema="vpi_test_schema",
    )
    return PostgreSqlBootstrapConfiguration(
        integration=integration,
        schema_name="vpi_test_schema",
        table_name="vpi_data_pack_relational_record",
        identifier_table_name="vpi_product_identifiers",
        lexical_document_table_name="vpi_lexical_document",
        lexical_posting_table_name="vpi_lexical_posting",
        lexical_corpus_stats_table_name="vpi_lexical_corpus_stats",
        lexical_term_stats_table_name="vpi_lexical_term_stats",
        structured_attribute_table_name="vpi_structured_attribute",
    )


def _record_json(
    *,
    offer_id: str,
    title: str,
    brand: str,
    description: str,
    mpn: str | None = None,
    capacity: str | None = None,
) -> str:
    payload: dict[str, object] = {
        "id": offer_id,
        "title": title,
        "brand": brand,
        "description": description,
        "identifiers": [],
        "keyValuePairs": {},
    }
    if mpn is not None:
        payload["identifiers"] = [{"/mpn": f"[{mpn}]"}]
    if capacity is not None:
        payload["keyValuePairs"] = {"Capacity": capacity}
    return json.dumps(payload, ensure_ascii=False)


def _load_record(
    *,
    offer_id: str,
    title: str,
    brand: str,
    description: str,
    mpn: str | None = None,
    capacity: str | None = None,
    index: int,
) -> RelationalLoadRecord:
    return RelationalLoadRecord(
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(offer_id),
            catalog_id=_CATALOG_ID,
            source_revision="rev-lexical",
        ),
        global_row_index=index,
        record_json=_record_json(
            offer_id=offer_id,
            title=title,
            brand=brand,
            description=description,
            mpn=mpn,
            capacity=capacity,
        ),
        semantic_text=f"semantic-{offer_id}",
        semantic_text_hash=f"hash-{offer_id}",
        derivation_version="v2",
    )


def _indexed_document_from_text(
    *,
    offer_id: str,
    text: str,
) -> Bm25IndexedDocument:
    tokens, term_frequencies = build_term_frequencies(text)
    return Bm25IndexedDocument(
        catalog_id=_CATALOG_ID,
        offer_id=offer_id,
        source_revision_norm="rev-lexical",
        source_revision="rev-lexical",
        document_length=len(tokens),
        term_frequencies=term_frequencies,
    )


def _fixture_documents() -> tuple[Bm25IndexedDocument, ...]:
    records = (
        _load_record(
            offer_id="990-pro-2tb",
            title=f"Samsung SSD 990 PRO 2TB MZ-V9P2T0BW",
            brand="Samsung",
            description="NVMe internal storage 2TB",
            mpn="MZ-V9P2T0BW",
            capacity="2TB",
            index=0,
        ),
        _load_record(
            offer_id="990-pro-1tb",
            title="Samsung SSD 990 PRO 1TB",
            brand="Samsung",
            description="NVMe internal storage 1TB",
            mpn="MZ-V9P1T0BW",
            capacity="1TB",
            index=1,
        ),
        _load_record(
            offer_id="980-pro-2tb",
            title="Samsung SSD 980 PRO 2TB",
            brand="Samsung",
            description="NVMe internal storage 2TB",
            mpn="MZ-V8P2T0BW",
            capacity="2TB",
            index=2,
        ),
        _load_record(
            offer_id="sn850x-2tb",
            title="WD_BLACK SN850X 2TB",
            brand="WD_BLACK",
            description="NVMe internal storage 2TB",
            mpn="WDS200T2XHE",
            capacity="2TB",
            index=3,
        ),
        _load_record(
            offer_id="sn850-1tb",
            title="WD_BLACK SN850 1TB",
            brand="WD_BLACK",
            description="NVMe internal storage 1TB",
            mpn="WDS100T1XHE",
            capacity="1TB",
            index=4,
        ),
    )
    from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.lexical_projection import (
        project_lexical_from_load_record,
    )

    documents: list[Bm25IndexedDocument] = []
    for record in records:
        projection = project_lexical_from_load_record(record)
        assert projection is not None
        tokens, term_frequencies = build_term_frequencies(projection.lexical_document)
        documents.append(
            Bm25IndexedDocument(
                catalog_id=record.source_ref.catalog_id,
                offer_id=record.source_ref.offer_id.value,
                source_revision_norm="rev-lexical",
                source_revision=record.source_ref.source_revision,
                document_length=len(tokens),
                term_frequencies=term_frequencies,
            )
        )
    return tuple(documents)


def _corpus_for_documents(
    documents: tuple[Bm25IndexedDocument, ...],
) -> tuple[Bm25CorpusStatistics, dict[str, int]]:
    average_length = sum(document.document_length for document in documents) / len(documents)
    corpus = Bm25CorpusStatistics(
        document_count=len(documents),
        average_document_length=average_length,
    )
    term_document_frequencies: dict[str, set[tuple[str, str, str]]] = {}
    for document in documents:
        identity = (document.catalog_id, document.offer_id, document.source_revision_norm)
        for term in document.term_frequencies:
            term_document_frequencies.setdefault(term, set()).add(identity)
    return corpus, {term: len(identities) for term, identities in term_document_frequencies.items()}


@dataclass
class _FakeCursor:
    _rows: list[Mapping[str, str | int | float | None]] = field(default_factory=list)

    def fetchone(self) -> Mapping[str, str | int | float | None] | None:
        if not self._rows:
            return None
        return self._rows[0]

    def fetchall(self) -> list[Mapping[str, str | int | float | None]]:
        return list(self._rows)


@dataclass
class _RoutingFakeConnection:
    ranked_rows: list[Mapping[str, str | int | float | None]]
    executed: list[ExecutedStatement] = field(default_factory=list)
    fail_with: BaseException | None = None
    common_term_posting_count: int = 0

    def execute(self, sql: SqlStatement, params: SqlParams = ()) -> _FakeCursor:
        self.executed.append((sql, params))
        if self.fail_with is not None:
            raise self.fail_with
        sql_text = _executed_sql_text(sql).lower()
        if "term_contributions" in sql_text and "limit %s" in sql_text:
            limit = int(params[-1]) if params else len(self.ranked_rows)
            return _FakeCursor(_rows=self.ranked_rows[:limit])
        return _FakeCursor(_rows=[])

    def close(self) -> None:
        return None


def _adapter_with_connection(connection: _RoutingFakeConnection) -> PostgreSqlLexicalCandidateSearchAdapter:
    config = _configuration()

    def _factory() -> _RoutingFakeConnection:
        return connection

    provider = PostgreSQLConnectionProvider(
        config.integration,
        tenant_schema=config.schema_name,
        connection_factory=_factory,
    )
    return PostgreSqlLexicalCandidateSearchAdapter(
        _provider=provider,
        _configuration=config,
    )


def test_tokenizer_preserves_hyphenated_and_alphanumeric_tokens() -> None:
    tokens = tokenize_lexical_document("Model MZ-V9P2T0BW SN850X RTX4090 ABC-123-XY")
    assert "mz-v9p2t0bw" in tokens
    assert "sn850x" in tokens
    assert "rtx4090" in tokens
    assert "abc-123-xy" in tokens


def test_exact_model_token_ranks_target_offer_first() -> None:
    documents = _fixture_documents()
    corpus, term_df = _corpus_for_documents(documents)
    ranked = rank_bm25_documents(
        query_text="MZ-V9P2T0BW",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=5,
    )
    assert ranked
    assert ranked[0].offer_id == "990-pro-2tb"


def test_near_model_does_not_outrank_exact_model() -> None:
    documents = _fixture_documents()
    corpus, term_df = _corpus_for_documents(documents)
    ranked = rank_bm25_documents(
        query_text="990 PRO 2TB",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=5,
    )
    offer_ids = [item.offer_id for item in ranked]
    assert offer_ids.index("990-pro-2tb") < offer_ids.index("990-pro-1tb")
    assert offer_ids.index("990-pro-2tb") < offer_ids.index("980-pro-2tb")


def test_capacity_variant_influences_ranking() -> None:
    documents = _fixture_documents()
    corpus, term_df = _corpus_for_documents(documents)
    ranked = rank_bm25_documents(
        query_text="990 PRO 1TB",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=5,
    )
    assert ranked[0].offer_id == "990-pro-1tb"


def test_case_insensitive_query_policy() -> None:
    documents = _fixture_documents()
    corpus, term_df = _corpus_for_documents(documents)
    lower = rank_bm25_documents(
        query_text="sn850x",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=3,
    )
    upper = rank_bm25_documents(
        query_text="SN850X",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=3,
    )
    assert [item.offer_id for item in lower] == [item.offer_id for item in upper]


def test_zero_match_returns_success_empty() -> None:
    documents = _fixture_documents()
    corpus, term_df = _corpus_for_documents(documents)
    ranked = rank_bm25_documents(
        query_text="ZZZ-NOMATCH-TOKEN",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=5,
    )
    assert ranked == ()


def test_top_k_limit_honored() -> None:
    documents = _fixture_documents()
    corpus, term_df = _corpus_for_documents(documents)
    ranked = rank_bm25_documents(
        query_text="Samsung SSD",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=2,
    )
    assert len(ranked) == 2


def test_deterministic_tie_breaking() -> None:
    left_doc = _indexed_document_from_text(
        offer_id="offer-a",
        text="shared token shared token",
    )
    right_doc = _indexed_document_from_text(
        offer_id="offer-b",
        text="shared token shared token",
    )
    right_doc = Bm25IndexedDocument(
        catalog_id=right_doc.catalog_id,
        offer_id=right_doc.offer_id,
        source_revision_norm=right_doc.source_revision_norm,
        source_revision=right_doc.source_revision,
        document_length=right_doc.document_length,
        term_frequencies=right_doc.term_frequencies,
    )
    left_doc = Bm25IndexedDocument(
        catalog_id=left_doc.catalog_id,
        offer_id=left_doc.offer_id,
        source_revision_norm=left_doc.source_revision_norm,
        source_revision=left_doc.source_revision,
        document_length=left_doc.document_length,
        term_frequencies=left_doc.term_frequencies,
    )
    documents = (right_doc, left_doc)
    corpus, term_df = _corpus_for_documents(documents)
    first = rank_bm25_documents(
        query_text="shared",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=2,
    )
    second = rank_bm25_documents(
        query_text="shared",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=2,
    )
    assert first == second
    assert first[0].offer_id == "offer-a"


def test_score_is_real_backend_bm25_not_rank_derived() -> None:
    documents = _fixture_documents()
    corpus, term_df = _corpus_for_documents(documents)
    ranked = rank_bm25_documents(
        query_text="MZ-V9P2T0BW",
        documents=documents,
        corpus=corpus,
        term_document_frequencies=term_df,
        limit=5,
    )
    assert ranked[0].bm25_score > 0.0
    assert ranked[0].bm25_score != float(5 - 0)


def test_adapter_satisfies_lexical_candidate_search_port() -> None:
    connection = _RoutingFakeConnection(ranked_rows=[])
    adapter = _adapter_with_connection(connection)
    port: LexicalCandidateSearchPort = adapter
    assert callable(port.search)


def test_adapter_search_uses_storage_side_ranked_bm25_not_ilike() -> None:
    connection = _RoutingFakeConnection(
        ranked_rows=[
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "sn850x-2tb",
                "source_revision_norm": "rev-lexical",
                "source_revision": "rev-lexical",
                "bm25_score": 2.5,
            }
        ],
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(LexicalSearchQuery(query_text="SN850X", limit=3))
    assert result.failure is None
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.channel is RetrievalChannel.LEXICAL
    assert candidate.rank == 0
    assert candidate.source_ref.offer_id.value == "sn850x-2tb"
    assert candidate.channel_score is not None
    assert candidate.channel_score.bm25_score > 0.0
    ranked_executions = _ranked_search_executions(connection)
    assert len(ranked_executions) == 1
    ranked_sql = _executed_sql_text(ranked_executions[0][0]).lower()
    assert "term_contributions" in ranked_sql
    assert "limit %s" in ranked_sql
    assert "ilike" not in ranked_sql
    assert "record_json" not in ranked_sql


def test_adapter_zero_match_success_empty() -> None:
    connection = _RoutingFakeConnection(ranked_rows=[])
    adapter = _adapter_with_connection(connection)
    result = adapter.search(LexicalSearchQuery(query_text="missing-token", limit=5))
    assert result.failure is None
    assert result.candidates == ()


def test_posting_lookup_dml_is_index_backed_shape() -> None:
    composed = lexical_posting_lookup_dml(
        LexicalPostingTableSpec(
            schema_name="vpi_test_schema",
            table_name="vpi_lexical_posting",
        )
    )
    sql = _executed_sql_text(composed).lower()
    assert '"vpi_test_schema"."vpi_lexical_posting"' in sql
    assert "term = any(%s)" in sql
    assert "ilike" not in sql


def test_lexical_posting_lookup_index_name_declared() -> None:
    assert _LEXICAL_POSTING_LOOKUP_INDEX_NAME == "vpi_lexical_posting_term_idx"


def test_canonical_adapter_has_no_ilike_or_fake_score_patterns() -> None:
    source = (_ADAPTER_ROOT / "lexical_search_adapter.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "ilike" not in lowered
    assert "record_json" not in lowered
    assert "query.limit -" not in source
    assert "semantic_text" not in lowered


def test_legacy_adapter_is_marked_reference_only() -> None:
    source = _LEGACY_ADAPTER_PATH.read_text(encoding="utf-8")
    assert "LEGACY / REFERENCE ONLY" in source
    assert "ILIKE" in source or "ilike" in source.lower()


def test_composition_builds_lexical_port_without_legacy_adapter() -> None:
    source = (
        _REPO_ROOT
        / "platform_proofs/scenarios/verified_product_identification/retrieval/composition.py"
    ).read_text(encoding="utf-8")
    assert "build_lexical_candidate_search" in source
    assert "PostgreSQLLexicalSearchAdapter" not in source
    assert "catalog_search_adapter" not in source


def test_application_has_zero_postgresql_imports() -> None:
    violations: list[str] = []
    for module_path in sorted(_APPLICATION_ROOT.rglob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("psycopg"):
                        violations.append(str(module_path))
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("psycopg"):
                violations.append(str(module_path))
    assert violations == []


def test_multi_channel_service_unchanged_signature() -> None:
    param_names = [service_field.name for service_field in fields(MultiChannelRetrievalService)]
    assert param_names == [
        "exact_lookup",
        "lexical_search",
        "structured_search",
        "vector_search",
    ]


def test_pluginability_with_fake_and_postgresql_lexical_adapter() -> None:
    connection = _RoutingFakeConnection(ranked_rows=[])
    request = MultiChannelRetrievalRequest(
        lexical_query=LexicalSearchQuery(query_text="SN850X", limit=2),
    )
    service_with_fake = MultiChannelRetrievalService(
        exact_lookup=FakeExactLookupA(),
        lexical_search=_RecordingLexicalSearch(),
        structured_search=RecordingStructuredSearch(),
        vector_search=RecordingVectorSearch(),
    )
    service_with_pg = MultiChannelRetrievalService(
        exact_lookup=FakeExactLookupA(),
        lexical_search=_adapter_with_connection(connection),
        structured_search=RecordingStructuredSearch(),
        vector_search=RecordingVectorSearch(),
    )
    fake_result = service_with_fake.retrieve(request)
    pg_result = service_with_pg.retrieve(request)
    assert fake_result.lexical.search_result is not None
    assert pg_result.lexical.search_result is not None


@dataclass
class _RecordingLexicalSearch:
    queries: list[LexicalSearchQuery] = field(default_factory=list)

    def search(self, query: LexicalSearchQuery):
        from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
            LexicalSearchResult,
        )

        self.queries.append(query)
        return LexicalSearchResult(candidates=())


def test_build_lexical_candidate_search_returns_port() -> None:
    port = build_lexical_candidate_search(schema_name="vpi_test_schema")
    assert isinstance(port, PostgreSqlLexicalCandidateSearchAdapter)


def test_invalid_empty_token_query_maps_to_invalid_query_failure() -> None:
    connection = _RoutingFakeConnection(ranked_rows=[])
    adapter = _adapter_with_connection(connection)
    result = adapter.search(LexicalSearchQuery(query_text="***", limit=3))
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_ranked_search_dml_enforces_storage_side_limit_and_joins() -> None:
    composed = lexical_bm25_ranked_search_dml(
        LexicalDocumentTableSpec(schema_name="vpi_test_schema", table_name="vpi_lexical_document"),
        LexicalPostingTableSpec(schema_name="vpi_test_schema", table_name="vpi_lexical_posting"),
        LexicalCorpusStatsTableSpec(
            schema_name="vpi_test_schema",
            table_name="vpi_lexical_corpus_stats",
        ),
        LexicalTermStatsTableSpec(
            schema_name="vpi_test_schema",
            table_name="vpi_lexical_term_stats",
        ),
        k1=1.2,
        b=0.75,
    )
    sql = _executed_sql_text(composed).lower()
    assert "unnest(%s::text[])" in sql
    assert '"vpi_lexical_posting"' in sql
    assert '"vpi_lexical_document"' in sql
    assert '"vpi_lexical_term_stats"' in sql
    assert '"vpi_lexical_corpus_stats"' in sql
    assert "order by" in sql
    assert "limit %s" in sql
    assert "count(*)" not in sql
    assert "avg(" not in sql


def test_adapter_search_uses_single_db_round_trip() -> None:
    connection = _RoutingFakeConnection(
        ranked_rows=[
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "offer-a",
                "source_revision_norm": "rev-lexical",
                "source_revision": "rev-lexical",
                "bm25_score": 1.0,
            }
        ],
        common_term_posting_count=1500,
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(LexicalSearchQuery(query_text="common", limit=5))
    assert result.failure is None
    assert len(_ranked_search_executions(connection)) == 1
    assert len(result.candidates) <= 5


def test_common_term_fixture_does_not_materialize_all_matches_in_python() -> None:
    limit = 3
    connection = _RoutingFakeConnection(
        ranked_rows=[
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": f"offer-{index}",
                "source_revision_norm": "rev-lexical",
                "source_revision": "rev-lexical",
                "bm25_score": float(limit - index),
            }
            for index in range(limit)
        ],
        common_term_posting_count=1200,
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(LexicalSearchQuery(query_text="common", limit=limit))
    assert result.failure is None
    assert len(result.candidates) == limit
    ranked_executions = _ranked_search_executions(connection)
    assert len(ranked_executions) == 1
    sql_text = _executed_sql_text(ranked_executions[0][0]).lower()
    assert "where catalog_id = %s" not in sql_text
    assert "term = any(%s)" not in sql_text


def test_duplicate_query_terms_double_count_semantics() -> None:
    document = _indexed_document_from_text(
        offer_id="dup-offer",
        text="alpha beta",
    )
    corpus = Bm25CorpusStatistics(document_count=1, average_document_length=float(document.document_length))
    term_df = {"alpha": 1, "beta": 1}
    single = compute_bm25_score(
        query_terms=("alpha",),
        document=document,
        corpus=corpus,
        term_document_frequencies=term_df,
        configuration=Bm25EngineConfiguration(),
    )
    double = compute_bm25_score(
        query_terms=("alpha", "alpha"),
        document=document,
        corpus=corpus,
        term_document_frequencies=term_df,
        configuration=Bm25EngineConfiguration(),
    )
    assert double == single * 2.0
    ranked_sql = _executed_sql_text(
        lexical_bm25_ranked_search_dml(
            LexicalDocumentTableSpec(schema_name="s", table_name="d"),
            LexicalPostingTableSpec(schema_name="s", table_name="p"),
            LexicalCorpusStatsTableSpec(schema_name="s", table_name="c"),
            LexicalTermStatsTableSpec(schema_name="s", table_name="t"),
            k1=1.2,
            b=0.75,
        )
    ).lower()
    assert "unnest(%s::text[])" in ranked_sql
