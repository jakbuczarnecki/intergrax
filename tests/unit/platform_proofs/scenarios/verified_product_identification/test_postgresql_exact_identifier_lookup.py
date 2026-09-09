"""Unit tests for PostgreSQL exact identifier lookup adapter."""

from __future__ import annotations

import ast
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
    ExactIdentifierQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    ExactIdentifierLookupPort,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.service import (
    MultiChannelRetrievalService,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.exact_identifier_lookup import (
    PostgreSqlExactIdentifierLookupAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    IdentifierTableSpec,
    _IDENTIFIER_LOOKUP_INDEX_NAME,
    identifier_lookup_dml,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval import (
    MultiChannelRetrievalRequest,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_catalog_contracts import (
    FakeExactLookupA,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_multi_channel_retrieval import (
    RecordingLexicalSearch,
    RecordingStructuredSearch,
    RecordingVectorSearch,
)

pytestmark = pytest.mark.unit

_, _, _, _PSYCOPG_SQL = import_psycopg()

SqlParam = str | int | None
SqlParams = tuple[SqlParam, ...]
SqlStatement = str | _PSYCOPG_SQL.Composable
ExecutedStatement = tuple[SqlStatement, SqlParams]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_ADAPTER_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/adapters/postgresql"
)
_APPLICATION_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/application"
)


def _executed_sql_text(statement: SqlStatement) -> str:
    if isinstance(statement, str):
        return statement
    return statement.as_string(None)


def _lookup_sql_text(executed: list[ExecutedStatement]) -> str:
    statement, _params = executed[-1]
    return _executed_sql_text(statement)


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
    )


@dataclass
class _FakeCursor:
    _rows: list[Mapping[str, str | None]] = field(default_factory=list)

    def fetchall(self) -> list[Mapping[str, str | None]]:
        return list(self._rows)


@dataclass
class _FakeConnection:
    rows: list[Mapping[str, str | None]] = field(default_factory=list)
    executed: list[ExecutedStatement] = field(default_factory=list)
    fail_with: BaseException | None = None

    def execute(self, sql: SqlStatement, params: SqlParams = ()) -> _FakeCursor:
        self.executed.append((sql, params))
        if self.fail_with is not None:
            raise self.fail_with
        return _FakeCursor(_rows=self.rows)

    def close(self) -> None:
        return None


def _identifier_row(
    *,
    catalog_id: str = "catalog-a",
    offer_id: str = "offer-1",
    source_revision_norm: str = "",
    source_revision: str | None = None,
    identifier_type: str = "gtin",
    source_value: str = "[8806095123456]",
    normalized_value: str = "8806095123456",
    source_field: str = "/gtin13",
) -> dict[str, str | None]:
    return {
        "catalog_id": catalog_id,
        "offer_id": offer_id,
        "source_revision_norm": source_revision_norm,
        "source_revision": source_revision,
        "identifier_type": identifier_type,
        "source_value": source_value,
        "normalized_value": normalized_value,
        "source_field": source_field,
    }


def _adapter_with_connection(
    connection: _FakeConnection,
    *,
    configuration: PostgreSqlBootstrapConfiguration | None = None,
) -> PostgreSqlExactIdentifierLookupAdapter:
    config = configuration or _configuration()

    def _factory() -> _FakeConnection:
        return connection

    provider = PostgreSQLConnectionProvider(
        config.integration,
        tenant_schema=config.schema_name,
        connection_factory=_factory,
    )
    return PostgreSqlExactIdentifierLookupAdapter(
        _provider=provider,
        _configuration=config,
    )


def _query(
    identifier_type: ProductIdentifierType,
    value: str,
    *,
    limit: int = 10,
) -> ExactIdentifierQuery:
    return ExactIdentifierQuery(
        identifier=ProductIdentifier(identifier_type=identifier_type, value=value),
        limit=limit,
    )


def test_adapter_satisfies_exact_identifier_lookup_port() -> None:
    adapter = _adapter_with_connection(_FakeConnection())
    port: ExactIdentifierLookupPort = adapter
    assert callable(port.lookup)


def test_gtin_exact_hit_maps_candidate() -> None:
    connection = _FakeConnection(rows=[_identifier_row()])
    adapter = _adapter_with_connection(connection)
    result = adapter.lookup(_query(ProductIdentifierType.GTIN, "[8806095123456]"))
    assert result.failure is None
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.channel is RetrievalChannel.EXACT
    assert candidate.rank == 0
    assert candidate.source_ref.catalog_id == "catalog-a"
    assert candidate.source_ref.offer_id.value == "offer-1"
    assert candidate.channel_score is not None
    assert candidate.channel_score.matched_identifier.source_field == "/gtin13"
    assert candidate.channel_score.matched_identifier.value == "[8806095123456]"


@pytest.mark.parametrize(
    ("identifier_type", "row_type", "value", "normalized"),
    [
        (ProductIdentifierType.MPN, "mpn", "[MZ-V9P2T0BW]", "MZ-V9P2T0BW"),
        (ProductIdentifierType.SKU, "sku", "[SKU-NEUTRAL-01]", "SKU-NEUTRAL-01"),
        (ProductIdentifierType.PRODUCT_ID, "product_id", "[PROD-NEUTRAL-01]", "PROD-NEUTRAL-01"),
    ],
)
def test_identifier_family_exact_hits(
    identifier_type: ProductIdentifierType,
    row_type: str,
    value: str,
    normalized: str,
) -> None:
    connection = _FakeConnection(
        rows=[
            _identifier_row(
                identifier_type=row_type,
                source_value=value,
                normalized_value=normalized,
                source_field=f"/{row_type}",
            )
        ]
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.lookup(_query(identifier_type, value))
    assert result.failure is None
    assert len(result.candidates) == 1


def test_zero_match_returns_success_empty() -> None:
    adapter = _adapter_with_connection(_FakeConnection(rows=[]))
    result = adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456"))
    assert result.failure is None
    assert result.candidates == ()


def test_multi_match_deterministic_order_and_ranking() -> None:
    connection = _FakeConnection(
        rows=[
            _identifier_row(catalog_id="catalog-a", offer_id="offer-1"),
            _identifier_row(catalog_id="catalog-a", offer_id="offer-9"),
            _identifier_row(catalog_id="catalog-b", offer_id="offer-2"),
        ]
    )
    adapter = _adapter_with_connection(connection)
    result = adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456"))
    assert [candidate.rank for candidate in result.candidates] == [0, 1, 2]
    assert [candidate.source_ref.offer_id.value for candidate in result.candidates] == [
        "offer-1",
        "offer-9",
        "offer-2",
    ]


def test_sql_limit_enforced_at_provider_boundary() -> None:
    connection = _FakeConnection(rows=[_identifier_row()])
    adapter = _adapter_with_connection(connection)
    adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456", limit=3))
    assert connection.executed
    _, params = connection.executed[-1]
    assert params[-1] == 3


def test_lookup_uses_indexed_predicate_without_like_or_semantic_scan() -> None:
    connection = _FakeConnection(rows=[])
    adapter = _adapter_with_connection(connection)
    adapter.lookup(_query(ProductIdentifierType.MPN, "MZ-V9P2T0BW"))
    sql = _lookup_sql_text(connection.executed).lower()
    params = connection.executed[-1][1]
    assert "identifier_type = %s" in sql
    assert "normalized_value = %s" in sql
    assert "like" not in sql
    assert "semantic_text" not in sql
    assert "record_json" not in sql
    assert params[0] == "mpn"
    assert params[1] == "MZ-V9P2T0BW"


def test_hostile_identifier_value_is_bound_data() -> None:
    connection = _FakeConnection(rows=[])
    adapter = _adapter_with_connection(connection)
    hostile = "'; DROP TABLE vpi_product_identifiers; --"
    adapter.lookup(_query(ProductIdentifierType.MPN, hostile))
    _, params = connection.executed[-1]
    assert params[1] == hostile
    assert "drop table" not in _lookup_sql_text(connection.executed).lower()


def test_provider_timeout_maps_to_typed_failure() -> None:
    _, errors, _, _ = import_psycopg()
    connection = _FakeConnection(fail_with=errors.QueryCanceled("timeout"))
    adapter = _adapter_with_connection(connection)
    result = adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456"))
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.TIMEOUT


def test_provider_unavailable_maps_to_typed_failure() -> None:
    _, errors, _, _ = import_psycopg()
    connection = _FakeConnection(fail_with=errors.OperationalError("connection lost"))
    adapter = _adapter_with_connection(connection)
    result = adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456"))
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.UNAVAILABLE


def test_unnormalizable_identifier_returns_invalid_query_failure() -> None:
    adapter = _adapter_with_connection(_FakeConnection())
    result = adapter.lookup(_query(ProductIdentifierType.GTIN, "[12345]"))
    assert result.candidates == ()
    assert result.failure is not None
    assert result.failure.kind is CatalogSearchFailureKind.INVALID_QUERY


def test_programming_error_not_swallowed() -> None:
    connection = _FakeConnection(fail_with=RuntimeError("unexpected defect"))
    adapter = _adapter_with_connection(connection)
    with pytest.raises(RuntimeError, match="unexpected defect"):
        adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456"))


def test_lookup_sql_uses_deterministic_order_by() -> None:
    connection = _FakeConnection(rows=[])
    adapter = _adapter_with_connection(connection)
    adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456"))
    sql = " ".join(_lookup_sql_text(connection.executed).lower().split())
    assert (
        "order by catalog_id asc, offer_id asc, source_revision_norm asc"
        in sql
    )


def test_lookup_sql_uses_composed_qualified_identifier_table() -> None:
    connection = _FakeConnection(rows=[])
    adapter = _adapter_with_connection(connection)
    adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456"))
    statement, _params = connection.executed[-1]
    _, _, _, sql_module = import_psycopg()
    assert isinstance(statement, sql_module.Composable)
    assert '"vpi_test_schema"."vpi_product_identifiers"' in _lookup_sql_text(connection.executed).lower()


def test_lookup_sql_schema_and_table_explicitly_qualified() -> None:
    configuration = PostgreSqlBootstrapConfiguration(
        integration=_configuration().integration,
        schema_name="vpi_lookup_schema",
        table_name="vpi_data_pack_relational_record",
        identifier_table_name="vpi_lookup_identifiers",
        lexical_document_table_name="vpi_lexical_document",
        lexical_posting_table_name="vpi_lexical_posting",
        lexical_corpus_stats_table_name="vpi_lexical_corpus_stats",
        lexical_term_stats_table_name="vpi_lexical_term_stats",
    )
    connection = _FakeConnection(rows=[])
    adapter = _adapter_with_connection(connection, configuration=configuration)
    adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456"))
    assert (
        '"vpi_lookup_schema"."vpi_lookup_identifiers"'
        in _lookup_sql_text(connection.executed).lower()
    )


def test_lookup_limit_and_values_remain_bind_parameters() -> None:
    connection = _FakeConnection(rows=[])
    adapter = _adapter_with_connection(connection)
    adapter.lookup(_query(ProductIdentifierType.GTIN, "8806095123456", limit=3))
    sql = _lookup_sql_text(connection.executed)
    assert sql.count("%s") == 3
    _, params = connection.executed[-1]
    assert params == ("gtin", "8806095123456", 3)


def test_invalid_identifier_table_configuration_still_rejected() -> None:
    with pytest.raises(ValueError, match="table_name must be a simple SQL identifier"):
        PostgreSqlBootstrapConfiguration.from_env(
            schema_name="vpi_test_schema",
            identifier_table_name="bad-table",
        )


def test_identifier_lookup_dml_matches_adapter_query_shape() -> None:
    composed = identifier_lookup_dml(
        IdentifierTableSpec(
            schema_name="vpi_test_schema",
            table_name="vpi_product_identifiers",
        )
    )
    sql = _executed_sql_text(composed).lower()
    assert '"vpi_test_schema"."vpi_product_identifiers"' in sql
    assert "identifier_type = %s" in sql
    assert "normalized_value = %s" in sql
    assert "limit %s" in sql


def test_no_f_string_identifier_sql_in_exact_lookup_module() -> None:
    module_path = _ADAPTER_ROOT / "exact_identifier_lookup.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                violations.append(ast.get_source_segment(source, value) or "")
    identifier_violations = [
        fragment
        for fragment in violations
        if "identifier_table_name" in fragment or "FROM {" in fragment
    ]
    assert identifier_violations == []


def test_index_name_is_declared_for_qualification() -> None:
    assert _IDENTIFIER_LOOKUP_INDEX_NAME == "vpi_product_identifiers_lookup_idx"


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


def test_pluginability_with_fake_and_postgresql_adapter() -> None:
    query = _query(ProductIdentifierType.GTIN, "[8806095123456]", limit=1)
    request = MultiChannelRetrievalRequest(exact_queries=(query,))
    service_with_fake = MultiChannelRetrievalService(
        exact_lookup=FakeExactLookupA(),
        lexical_search=RecordingLexicalSearch(),
        structured_search=RecordingStructuredSearch(),
        vector_search=RecordingVectorSearch(),
    )
    service_with_pg = MultiChannelRetrievalService(
        exact_lookup=_adapter_with_connection(_FakeConnection(rows=[_identifier_row()])),
        lexical_search=RecordingLexicalSearch(),
        structured_search=RecordingStructuredSearch(),
        vector_search=RecordingVectorSearch(),
    )
    fake_result = service_with_fake.retrieve(request)
    pg_result = service_with_pg.retrieve(request)
    assert fake_result.exact.candidates
    assert pg_result.exact.candidates


def test_adapter_has_no_qdrant_imports() -> None:
    violations: list[str] = []
    for module_path in sorted(_ADAPTER_ROOT.rglob("exact_identifier_lookup.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in {"qdrant", "qdrant_client", "pgvector"}:
                    violations.append(node.module)
    assert violations == []
