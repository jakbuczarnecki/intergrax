"""Unit tests for PostgreSQL structured attribute search adapter and storage projection."""

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
from platform_proofs.scenarios.verified_product_identification.application.catalog.structured_attribute_normalization import (
    normalize_structured_query_attribute_name,
    normalize_structured_query_value,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
    StructuredSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    RetrievalChannel,
    StructuredChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    StructuredCandidateSearchPort,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval import (
    MultiChannelRetrievalRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.service import (
    MultiChannelRetrievalService,
)
from platform_proofs.scenarios.verified_product_identification.retrieval.composition import (
    build_structured_candidate_search,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    StructuredAttributeTableSpec,
    _STRUCTURED_CANONICAL_EQUALS_INDEX_NAME,
    _STRUCTURED_CONTAINS_INDEX_NAME,
    _STRUCTURED_SOURCE_EQUALS_INDEX_NAME,
    structured_constraint_search_dml,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.structured_projection import (
    project_structured_from_load_record,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.structured_query import (
    prepare_structured_search_query,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.structured_search_adapter import (
    PostgreSqlStructuredCandidateSearchAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalLoadRecord,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_catalog_contracts import (
    FakeExactLookupA,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_multi_channel_retrieval import (
    RecordingLexicalSearch,
    RecordingVectorSearch,
)

pytestmark = pytest.mark.unit

_, _, _, _PSYCOPG_SQL = import_psycopg()

SqlParam = str | int | None | list[int] | list[str]
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
_CATALOG_ID = "vpi-structured-catalog"


def _executed_sql_text(statement: SqlStatement) -> str:
    if isinstance(statement, str):
        return statement
    return statement.as_string(None)


def _configuration(*, pg_trgm: bool = False) -> PostgreSqlBootstrapConfiguration:
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
    brand: str,
    model: str,
    capacity: str,
) -> str:
    payload = {
        "id": offer_id,
        "title": f"{brand} {model}",
        "brand": brand,
        "description": f"{brand} {model} {capacity}",
        "identifiers": [],
        "keyValuePairs": {
            "Brand": brand,
            "Model": model,
            "Capacity": capacity,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _load_record(
    *,
    offer_id: str,
    brand: str,
    model: str,
    capacity: str,
    index: int,
) -> RelationalLoadRecord:
    return RelationalLoadRecord(
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId(offer_id),
            catalog_id=_CATALOG_ID,
            source_revision="rev-structured",
        ),
        global_row_index=index,
        record_json=_record_json(
            offer_id=offer_id,
            brand=brand,
            model=model,
            capacity=capacity,
        ),
        semantic_text=f"semantic-{offer_id}",
        semantic_text_hash=f"hash-{offer_id}",
        derivation_version="v2",
    )


def _quality_fixture_records() -> tuple[RelationalLoadRecord, ...]:
    return (
        _load_record(
            offer_id="offer-a",
            brand="Samsung",
            model="990 PRO",
            capacity="2TB",
            index=1,
        ),
        _load_record(
            offer_id="offer-b",
            brand="Samsung",
            model="990 PRO",
            capacity="1TB",
            index=2,
        ),
        _load_record(
            offer_id="offer-c",
            brand="Samsung",
            model="980 PRO",
            capacity="2TB",
            index=3,
        ),
        _load_record(
            offer_id="offer-d",
            brand="Western Digital",
            model="SN850X",
            capacity="2TB",
            index=4,
        ),
    )


def _quality_query(*, limit: int = 10) -> StructuredSearchQuery:
    return StructuredSearchQuery(
        constraints=(
            StructuredAttributeConstraint(
                attribute_name="Brand",
                operator=StructuredConstraintOperator.EQUALS,
                value="Samsung",
            ),
            StructuredAttributeConstraint(
                attribute_name="Capacity",
                operator=StructuredConstraintOperator.EQUALS,
                value="2TB",
            ),
            StructuredAttributeConstraint(
                attribute_name="Model",
                operator=StructuredConstraintOperator.EQUALS,
                value="990 PRO",
            ),
        ),
        limit=limit,
    )


@dataclass
class _FakeCursor:
    _rows: list[Mapping[str, str | int | None]] = field(default_factory=list)

    def fetchall(self) -> list[Mapping[str, str | int | None]]:
        return list(self._rows)

    def fetchone(self) -> Mapping[str, str | int | None] | None:
        if not self._rows:
            return None
        return self._rows[0]


@dataclass(frozen=True, slots=True)
class _ContainsIndexCapability:
    schema_name: str
    table_name: str
    valid_definition: bool = True


@dataclass
class _FakeConnection:
    ranked_rows: list[Mapping[str, str | int | None]] = field(default_factory=list)
    executed: list[ExecutedStatement] = field(default_factory=list)
    fail_with: BaseException | None = None
    pg_trgm_available: bool = False
    contains_index: _ContainsIndexCapability | None = None

    def execute(self, sql: SqlStatement, params: SqlParams = ()) -> _FakeCursor:
        self.executed.append((sql, params))
        if self.fail_with is not None:
            raise self.fail_with
        sql_text = _executed_sql_text(sql).lower()
        if "pg_extension" in sql_text and "extname" in sql_text:
            if self.pg_trgm_available and params == ("pg_trgm",):
                return _FakeCursor(_rows=[{"present": 1}])
            return _FakeCursor(_rows=[])
        if "pg_index" in sql_text and "gin_trgm_ops" in sql_text:
            if (
                self.contains_index is not None
                and self.contains_index.valid_definition
                and len(params) == 3
                and str(params[0]) == self.contains_index.schema_name
                and str(params[1]) == self.contains_index.table_name
                and str(params[2]) == _STRUCTURED_CONTAINS_INDEX_NAME
            ):
                return _FakeCursor(_rows=[{"present": 1}])
            return _FakeCursor(_rows=[])
        return _FakeCursor(_rows=self.ranked_rows)

    def close(self) -> None:
        return None


def _adapter_with_connection(
    connection: _FakeConnection,
    *,
    configuration: PostgreSqlBootstrapConfiguration | None = None,
) -> PostgreSqlStructuredCandidateSearchAdapter:
    config = configuration or _configuration()
    return PostgreSqlStructuredCandidateSearchAdapter(
        _provider=PostgreSQLConnectionProvider(
            config.integration,
            tenant_schema=config.schema_name,
            connection_factory=lambda: connection,
        ),
        _configuration=config,
    )


def _contains_query(*, value: str = "pro") -> StructuredSearchQuery:
    return StructuredSearchQuery(
        constraints=(
            StructuredAttributeConstraint(
                attribute_name="Model",
                operator=StructuredConstraintOperator.CONTAINS,
                value=value,
            ),
        )
    )


def _configured_contains_index() -> _ContainsIndexCapability:
    return _ContainsIndexCapability(
        schema_name="vpi_test_schema",
        table_name="vpi_structured_attribute",
    )


def _capability_executions(connection: _FakeConnection) -> list[ExecutedStatement]:
    return [
        executed
        for executed in connection.executed
        if "gin_trgm_ops" in _executed_sql_text(executed[0]).lower()
        or (
            "pg_extension" in _executed_sql_text(executed[0]).lower()
            and "extname" in _executed_sql_text(executed[0]).lower()
        )
    ]


def _search_executions(connection: _FakeConnection) -> list[ExecutedStatement]:
    return [
        executed
        for executed in connection.executed
        if "constraint_matches" in _executed_sql_text(executed[0]).lower()
    ]


def test_adapter_satisfies_structured_candidate_search_port() -> None:
    adapter = _adapter_with_connection(_FakeConnection())
    port: StructuredCandidateSearchPort = adapter
    assert callable(port.search)


def test_one_equals_hit_maps_candidate() -> None:
    connection = _FakeConnection(
        ranked_rows=[
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "offer-a",
                "source_revision_norm": "rev-structured",
                "source_revision": "rev-structured",
                "matched_constraint_count": 1,
            }
        ]
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(
        StructuredSearchQuery(
            constraints=(
                StructuredAttributeConstraint(
                    attribute_name="Brand",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="Samsung",
                ),
            ),
            limit=5,
        )
    )
    assert result.failure is None
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.channel is RetrievalChannel.STRUCTURED
    assert candidate.rank == 0
    assert candidate.source_ref.catalog_id == _CATALOG_ID
    assert candidate.source_ref.offer_id.value == "offer-a"
    assert isinstance(candidate.channel_score, StructuredChannelScore)
    assert candidate.channel_score.matched_constraint_count == 1
    assert candidate.channel_score.total_constraint_count == 1


def test_zero_match_success_empty() -> None:
    adapter = _adapter_with_connection(_FakeConnection(ranked_rows=[]))
    result = adapter.search(
        StructuredSearchQuery(
            constraints=(
                StructuredAttributeConstraint(
                    attribute_name="Brand",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="NoSuchBrand",
                ),
            )
        )
    )
    assert result.failure is None
    assert result.candidates == ()


def test_quality_fixture_scores_and_ordering() -> None:
    connection = _FakeConnection(
        ranked_rows=[
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "offer-a",
                "source_revision_norm": "rev-structured",
                "source_revision": "rev-structured",
                "matched_constraint_count": 3,
            },
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "offer-b",
                "source_revision_norm": "rev-structured",
                "source_revision": "rev-structured",
                "matched_constraint_count": 2,
            },
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "offer-c",
                "source_revision_norm": "rev-structured",
                "source_revision": "rev-structured",
                "matched_constraint_count": 2,
            },
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "offer-d",
                "source_revision_norm": "rev-structured",
                "source_revision": "rev-structured",
                "matched_constraint_count": 1,
            },
        ]
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(_quality_query())
    assert result.failure is None
    scores = [
        (candidate.source_ref.offer_id.value, candidate.channel_score)
        for candidate in result.candidates
        if isinstance(candidate.channel_score, StructuredChannelScore)
    ]
    assert scores[0] == ("offer-a", StructuredChannelScore(3, 3))
    assert scores[1][1] == StructuredChannelScore(2, 3)
    assert scores[2][1] == StructuredChannelScore(2, 3)
    assert scores[3][1] == StructuredChannelScore(1, 3)
    assert [candidate.rank for candidate in result.candidates] == [0, 1, 2, 3]


def test_duplicate_query_constraints_deduplicated() -> None:
    prepared = prepare_structured_search_query(
        StructuredSearchQuery(
            constraints=(
                StructuredAttributeConstraint(
                    attribute_name="Brand",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="Samsung",
                ),
                StructuredAttributeConstraint(
                    attribute_name=" Brand ",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="Samsung",
                ),
            )
        )
    )
    assert not isinstance(prepared, CatalogSearchFailure)
    assert prepared.total_constraint_count == 1
    assert len(prepared.constraints) == 1


def test_projection_preserves_multiple_source_keys_and_values() -> None:
    record = _load_record(
        offer_id="offer-dup",
        brand="Samsung",
        model="990 PRO",
        capacity="2TB",
        index=9,
    )
    rows = project_structured_from_load_record(record)
    brand_rows = [row for row in rows if row.source_key == "Brand"]
    assert len(brand_rows) == 1
    assert brand_rows[0].canonical_key == "Brand"
    assert brand_rows[0].normalized_text_value == "Samsung"


def test_query_normalization_matches_projection_values() -> None:
    record = _quality_fixture_records()[0]
    projected = project_structured_from_load_record(record)
    capacity = next(row for row in projected if row.source_key == "Capacity")
    assert normalize_structured_query_attribute_name(" Capacity ") == capacity.canonical_key
    assert normalize_structured_query_value(" 2TB ") == capacity.normalized_text_value


def test_contains_without_pg_trgm_fails_closed() -> None:
    connection = _FakeConnection(pg_trgm_available=False)
    adapter = _adapter_with_connection(connection)
    result = adapter.search(_contains_query())
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY
    assert "pg_trgm" in result.failure.message
    assert _search_executions(connection) == []


def test_contains_without_index_fails_closed() -> None:
    connection = _FakeConnection(pg_trgm_available=True, contains_index=None)
    adapter = _adapter_with_connection(connection)
    result = adapter.search(_contains_query())
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY
    assert _search_executions(connection) == []


def test_contains_with_index_in_wrong_schema_fails_closed() -> None:
    connection = _FakeConnection(
        pg_trgm_available=True,
        contains_index=_ContainsIndexCapability(
            schema_name="other_schema",
            table_name="vpi_structured_attribute",
        ),
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(_contains_query())
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY
    assert _search_executions(connection) == []


def test_contains_with_index_on_wrong_table_fails_closed() -> None:
    connection = _FakeConnection(
        pg_trgm_available=True,
        contains_index=_ContainsIndexCapability(
            schema_name="vpi_test_schema",
            table_name="other_structured_table",
        ),
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(_contains_query())
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY
    assert _search_executions(connection) == []


def test_contains_with_incompatible_index_definition_fails_closed() -> None:
    connection = _FakeConnection(
        pg_trgm_available=True,
        contains_index=_ContainsIndexCapability(
            schema_name="vpi_test_schema",
            table_name="vpi_structured_attribute",
            valid_definition=False,
        ),
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(_contains_query())
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY
    assert _search_executions(connection) == []


def test_contains_available_executes_single_search_query() -> None:
    connection = _FakeConnection(
        pg_trgm_available=True,
        contains_index=_configured_contains_index(),
        ranked_rows=[
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "offer-a",
                "source_revision_norm": "rev-structured",
                "source_revision": "rev-structured",
                "matched_constraint_count": 1,
            }
        ],
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(_contains_query())
    assert result.failure is None
    assert len(result.candidates) == 1
    assert len(_search_executions(connection)) == 1


def test_equals_only_query_does_not_probe_contains_capability() -> None:
    connection = _FakeConnection(
        pg_trgm_available=False,
        ranked_rows=[
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "offer-a",
                "source_revision_norm": "rev-structured",
                "source_revision": "rev-structured",
                "matched_constraint_count": 1,
            }
        ],
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(
        StructuredSearchQuery(
            constraints=(
                StructuredAttributeConstraint(
                    attribute_name="Brand",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="Samsung",
                ),
            )
        )
    )
    assert result.failure is None
    assert _capability_executions(connection) == []


def test_contains_capability_result_is_cached() -> None:
    connection = _FakeConnection(
        pg_trgm_available=True,
        contains_index=_configured_contains_index(),
        ranked_rows=[],
    )
    adapter = _adapter_with_connection(connection)
    first = adapter.search(_contains_query())
    assert first.failure is None
    capability_count_after_first = len(_capability_executions(connection))
    second = adapter.search(_contains_query())
    assert second.failure is None
    assert len(_capability_executions(connection)) == capability_count_after_first


def test_contains_branch_included_when_requested() -> None:
    sql = _executed_sql_text(
        structured_constraint_search_dml(
            StructuredAttributeTableSpec(
                schema_name="vpi_test_schema",
                table_name="vpi_structured_attribute",
            ),
            include_contains_branch=True,
        )
    ).lower()
    assert "operator = 'contains'" in sql
    assert "ilike" in sql


def test_equals_search_dml_has_no_record_json_or_semantic_text() -> None:
    sql = _executed_sql_text(
        structured_constraint_search_dml(
            StructuredAttributeTableSpec(
                schema_name="vpi_test_schema",
                table_name="vpi_structured_attribute",
            ),
            include_contains_branch=False,
        )
    ).lower()
    assert "record_json" not in sql
    assert "semantic_text" not in sql
    assert "count(distinct constraint_ordinal)" in sql
    assert "limit %s" in sql


def test_adapter_search_uses_single_db_round_trip() -> None:
    connection = _FakeConnection(
        ranked_rows=[
            {
                "catalog_id": _CATALOG_ID,
                "offer_id": "offer-a",
                "source_revision_norm": "rev-structured",
                "source_revision": "rev-structured",
                "matched_constraint_count": 1,
            }
        ]
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.search(
        StructuredSearchQuery(
            constraints=(
                StructuredAttributeConstraint(
                    attribute_name="Brand",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="Samsung",
                ),
                StructuredAttributeConstraint(
                    attribute_name="Capacity",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="2TB",
                ),
            ),
            limit=2,
        )
    )
    assert result.failure is None
    assert len(_search_executions(connection)) == 1
    statement, params = _search_executions(connection)[0]
    assert params[-1] == 2


def test_timeout_maps_to_catalog_search_failure() -> None:
    _, errors, _, _ = import_psycopg()
    connection = _FakeConnection(fail_with=errors.QueryCanceled("timeout"))
    adapter = _adapter_with_connection(connection)
    result = adapter.search(
        StructuredSearchQuery(
            constraints=(
                StructuredAttributeConstraint(
                    attribute_name="Brand",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="Samsung",
                ),
            )
        )
    )
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.TIMEOUT


def test_unavailable_maps_to_catalog_search_failure() -> None:
    _, errors, _, _ = import_psycopg()
    connection = _FakeConnection(fail_with=errors.OperationalError("down"))
    adapter = _adapter_with_connection(connection)
    result = adapter.search(
        StructuredSearchQuery(
            constraints=(
                StructuredAttributeConstraint(
                    attribute_name="Brand",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="Samsung",
                ),
            )
        )
    )
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.UNAVAILABLE


def test_canonical_adapter_has_no_legacy_structured_patterns() -> None:
    source = (_ADAPTER_ROOT / "structured_search_adapter.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "record_json" not in lowered
    assert "semantic_text" not in lowered
    assert "query.constraints[0]" not in source


def test_legacy_structured_adapter_marked_reference_only() -> None:
    legacy = _LEGACY_ADAPTER_PATH.read_text(encoding="utf-8")
    assert "LEGACY / REFERENCE ONLY" in legacy
    assert "query.constraints[0]" in legacy
    assert "matched_constraint_count=1" in legacy


def test_application_layer_has_zero_postgresql_imports() -> None:
    violations: list[str] = []
    for module_path in _APPLICATION_ROOT.rglob("*.py"):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("psycopg") or "postgresql" in alias.name:
                        violations.append(str(module_path))
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("psycopg") or "postgresql" in node.module:
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


def test_build_structured_candidate_search_returns_port() -> None:
    port = build_structured_candidate_search(schema_name="vpi_test_schema")
    assert isinstance(port, PostgreSqlStructuredCandidateSearchAdapter)


def test_schema_declares_structured_equals_indexes() -> None:
    assert _STRUCTURED_CANONICAL_EQUALS_INDEX_NAME == "vpi_structured_canonical_equals_idx"
    assert _STRUCTURED_SOURCE_EQUALS_INDEX_NAME == "vpi_structured_source_equals_idx"


def test_pluginability_with_fake_and_postgresql_structured_adapter() -> None:
    @dataclass
    class _RecordingStructuredSearch:
        queries: list[StructuredSearchQuery] = field(default_factory=list)

        def search(self, query: StructuredSearchQuery):
            from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
                StructuredSearchResult,
            )

            self.queries.append(query)
            return StructuredSearchResult(candidates=())

    request = MultiChannelRetrievalRequest(structured_query=_quality_query(limit=2))
    service_with_fake = MultiChannelRetrievalService(
        exact_lookup=FakeExactLookupA(),
        lexical_search=RecordingLexicalSearch(),
        structured_search=_RecordingStructuredSearch(),
        vector_search=RecordingVectorSearch(),
    )
    service_with_pg = MultiChannelRetrievalService(
        exact_lookup=FakeExactLookupA(),
        lexical_search=RecordingLexicalSearch(),
        structured_search=_adapter_with_connection(_FakeConnection(ranked_rows=[])),
        vector_search=RecordingVectorSearch(),
    )
    fake_result = service_with_fake.retrieve(request)
    pg_result = service_with_pg.retrieve(request)
    assert fake_result.structured.search_result is not None
    assert pg_result.structured.search_result is not None
