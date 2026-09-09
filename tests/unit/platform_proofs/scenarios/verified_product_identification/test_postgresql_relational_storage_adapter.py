"""Unit tests for PostgreSQL relational storage bootstrap adapter."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    import_psycopg,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
    _record_payload_matches,
    _source_revision_norm,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    PostgreSqlBootstrapConfiguration,
    validate_table_identifier,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.errors import (
    PostgreSqlBootstrapConfigurationError,
    PostgreSqlBootstrapIdentityConflictError,
    PostgreSqlBootstrapOperationError,
    PostgreSqlBootstrapSchemaError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema import (
    IdentifierTableSpec,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.stored_row import (
    StoredRelationalRow,
    stored_relational_row_from_fetched_row,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.target_mapping import (
    reject_unsafe_logical_target,
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

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_ADAPTER_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/adapters/postgresql"
)
_CORE_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/data_pack_load"
)
_FORBIDDEN_ADAPTER_IMPORTS = frozenset(
    {
        "qdrant",
        "pgvector",
        "torch",
        "transformers",
        "sentence_transformers",
    }
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".")[0])
    return imports


def _source_ref(offer_suffix: str) -> SourceRecordRef:
    return SourceRecordRef(
        offer_id=ProductOfferId(f"offer-{offer_suffix}"),
        catalog_id="wdc-v2-selected",
        source_revision=None,
    )


def _record(
    index: int,
    *,
    offer_suffix: str | None = None,
    semantic_text: str = "semantic",
    semantic_hash: str = "hash-a",
    record_json: str | None = None,
) -> RelationalLoadRecord:
    suffix = offer_suffix or str(index)
    payload = record_json or json.dumps({"id": f"offer-{suffix}", "title": "relay"})
    return RelationalLoadRecord(
        source_ref=_source_ref(suffix),
        global_row_index=index,
        record_json=payload,
        semantic_text=semantic_text,
        semantic_text_hash=semantic_hash,
        derivation_version="v1",
    )


def _batch(*records: RelationalLoadRecord, batch_number: int = 0) -> RelationalBatch:
    return RelationalBatch(
        batch_number=batch_number,
        target=RelationalTargetId("vpi-products"),
        records=tuple(records),
    )


def _configuration(schema_name: str = "vpi_test_schema") -> PostgreSqlBootstrapConfiguration:
    integration = PostgreSQLIntegrationConfig(
        host="localhost",
        port=5432,
        user="intergrax",
        password="secret-value",
        database="intergrax",
        tenant_schema=schema_name,
    )
    return PostgreSqlBootstrapConfiguration(
        integration=integration,
        schema_name=schema_name,
        table_name="vpi_data_pack_relational_record",
        identifier_table_name="vpi_product_identifiers",
    )


@dataclass
class _FakeCursor:
    rowcount: int = 1
    _rows: list[Mapping[str, object]] = field(default_factory=list)

    def fetchone(self) -> Mapping[str, object] | None:
        if not self._rows:
            return None
        return self._rows[0]

    def fetchall(self) -> list[Mapping[str, object]]:
        return list(self._rows)


@dataclass
class _FakeConnection:
    storage: dict[tuple[str, str, str], dict[str, object]] = field(default_factory=dict)
    by_row_index: dict[int, tuple[str, str, str]] = field(default_factory=dict)
    identifier_rows: list[tuple[Any, ...]] = field(default_factory=list)
    committed: int = 0
    rolled_back: int = 0
    in_transaction: bool = False
    fail_insert_on_offer: str | None = None
    schema_columns: list[Mapping[str, object]] | None = None
    schema_constraints: list[Mapping[str, object]] | None = None
    _txn_storage: dict[tuple[str, str, str], dict[str, object]] | None = None
    _txn_by_row_index: dict[int, tuple[str, str, str]] | None = None
    _txn_identifier_rows: list[tuple[Any, ...]] | None = None
    executed: list[tuple[Any, tuple[Any, ...]]] = field(default_factory=list)

    def _begin_snapshot(self) -> None:
        self.in_transaction = True
        self._txn_storage = dict(self.storage)
        self._txn_by_row_index = dict(self.by_row_index)
        self._txn_identifier_rows = list(self.identifier_rows)

    def execute(self, sql: Any, params: tuple[Any, ...] = ()) -> _FakeCursor:
        self.executed.append((sql, params))
        sql_text = _executed_sql_text(sql).lower()
        if "set transaction isolation level" in sql_text:
            self._begin_snapshot()
            return _FakeCursor()
        if sql_text.startswith("set "):
            return _FakeCursor()
        if "create table" in sql_text or "create schema" in sql_text:
            return _FakeCursor()
        if "information_schema.columns" in sql_text:
            columns = self.schema_columns or [
                {"column_name": "catalog_id", "data_type": "text", "is_nullable": "NO"},
                {"column_name": "offer_id", "data_type": "text", "is_nullable": "NO"},
                {"column_name": "source_revision_norm", "data_type": "text", "is_nullable": "NO"},
                {"column_name": "source_revision", "data_type": "text", "is_nullable": "YES"},
                {"column_name": "global_row_index", "data_type": "bigint", "is_nullable": "NO"},
                {"column_name": "record_json", "data_type": "jsonb", "is_nullable": "NO"},
                {"column_name": "semantic_text", "data_type": "text", "is_nullable": "NO"},
                {"column_name": "semantic_text_hash", "data_type": "text", "is_nullable": "NO"},
                {"column_name": "derivation_version", "data_type": "text", "is_nullable": "NO"},
            ]
            return _FakeCursor(_rows=columns)
        if "information_schema.table_constraints" in sql_text:
            if self.schema_constraints is None:
                constraints = [
                    {"constraint_name": "vpi_dpr_source_identity_pk"},
                    {"constraint_name": "vpi_dpr_global_row_index_uq"},
                ]
            else:
                constraints = self.schema_constraints
            return _FakeCursor(_rows=constraints)
        if (
            "insert into" in sql_text
            and "normalized_value" in sql_text
            and len(params) == 8
        ):
            self.identifier_rows.append(tuple(params))
            return _FakeCursor(rowcount=1)
        if "insert into" in sql_text and params:
            catalog_id, offer_id, revision_norm = params[0], params[1], params[2]
            if self.fail_insert_on_offer == offer_id:
                raise _pg_unique_violation()
            identity = (str(catalog_id), str(offer_id), str(revision_norm))
            if identity in self.storage:
                return _FakeCursor(rowcount=0)
            row_index = int(params[4])
            if row_index in self.by_row_index and self.by_row_index[row_index] != identity:
                return _FakeCursor(rowcount=0)
            row = {
                "catalog_id": catalog_id,
                "offer_id": offer_id,
                "source_revision_norm": revision_norm,
                "source_revision": params[3],
                "global_row_index": row_index,
                "record_json": str(params[5]),
                "semantic_text": params[6],
                "semantic_text_hash": params[7],
                "derivation_version": params[8],
            }
            self.storage[identity] = row
            self.by_row_index[row_index] = identity
            return _FakeCursor(rowcount=1)
        if "where catalog_id = %s" in sql_text and params:
            identity = (str(params[0]), str(params[1]), str(params[2]))
            row = self.storage.get(identity)
            return _FakeCursor(_rows=[row] if row else [])
        if "where global_row_index = %s" in sql_text and params:
            identity = self.by_row_index.get(int(params[0]))
            row = self.storage.get(identity) if identity else None
            return _FakeCursor(_rows=[row] if row else [])
        return _FakeCursor()

    def commit(self) -> None:
        self.committed += 1
        self.in_transaction = False
        self._txn_storage = None
        self._txn_by_row_index = None
        self._txn_identifier_rows = None

    def rollback(self) -> None:
        self.rolled_back += 1
        if self._txn_storage is not None:
            self.storage = dict(self._txn_storage)
            self.by_row_index = dict(self._txn_by_row_index or {})
            self.identifier_rows = list(self._txn_identifier_rows or [])
        self.in_transaction = False
        self._txn_storage = None
        self._txn_by_row_index = None
        self._txn_identifier_rows = None

    def close(self) -> None:
        return None


def _executed_sql_text(statement: object) -> str:
    if isinstance(statement, str):
        return statement
    _, _, _, sql_module = import_psycopg()
    if isinstance(statement, sql_module.Composable):
        return statement.as_string(None)
    return str(statement)


def _identifier_insert_sql_text(executed: list[tuple[Any, tuple[Any, ...]]]) -> str:
    for statement, params in executed:
        if len(params) != 8:
            continue
        sql_text = _executed_sql_text(statement).lower()
        if "insert into" in sql_text and "normalized_value" in sql_text:
            return _executed_sql_text(statement)
    raise AssertionError("expected identifier insert SQL execution")


def _pg_unique_violation() -> Exception:
    _, pg_errors, _, _ = import_psycopg()
    return pg_errors.UniqueViolation("duplicate key value violates unique constraint")


def _adapter_with_fake(
    connection: _FakeConnection,
    *,
    prepared: bool = True,
) -> PostgreSqlRelationalStorageAdapter:
    configuration = _configuration()

    def _factory() -> _FakeConnection:
        return connection

    provider = PostgreSQLConnectionProvider(
        configuration.integration,
        tenant_schema=configuration.schema_name,
        connection_factory=_factory,
    )
    provider._apply_search_path_on_connection = lambda _connection: None  # type: ignore[method-assign]
    provider.ensure_schema_exists = lambda _session, _schema_name=None: None  # type: ignore[method-assign]
    adapter = PostgreSqlRelationalStorageAdapter(
        _provider=provider,
        _configuration=configuration,
        _prepared_targets=set({"vpi-products"} if prepared else set()),
    )
    return adapter


# --- CONFIGURATION ---


def test_valid_postgresql_configuration() -> None:
    config = _configuration()
    assert config.schema_name == "vpi_test_schema"
    assert config.table_name == "vpi_data_pack_relational_record"


def test_invalid_table_mapping_rejected() -> None:
    with pytest.raises(ValueError):
        validate_table_identifier("bad-table")


def test_credentials_not_exposed_in_repr() -> None:
    config = _configuration()
    rendered = repr(config)
    assert "secret-value" not in rendered
    assert "password" not in rendered.lower()


def test_unmapped_logical_target_rejected() -> None:
    with pytest.raises(PostgreSqlBootstrapConfigurationError):
        resolve_physical_target(RelationalTargetId("unknown-target"), _configuration())


def test_unsafe_logical_target_rejected() -> None:
    with pytest.raises(PostgreSqlBootstrapConfigurationError):
        reject_unsafe_logical_target("'; DROP TABLE users; --")


# --- SCHEMA ---


def test_prepare_new_schema_table() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection, prepared=False)
    with (
        patch(
            "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema.create_table_ddl",
            return_value="CREATE TABLE IF NOT EXISTS vpi_data_pack_relational_record (id int)",
        ),
        patch(
            "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema.create_identifier_table_ddl",
            return_value="CREATE TABLE IF NOT EXISTS vpi_product_identifiers (id int)",
        ),
        patch(
            "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema.create_identifier_lookup_index_ddl",
            return_value="CREATE INDEX IF NOT EXISTS vpi_product_identifiers_lookup_idx ON vpi_product_identifiers (identifier_type, normalized_value)",
        ),
        patch(
            "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter.verify_identifier_table_compatible",
            return_value=None,
        ),
    ):
        adapter.prepare_target(RelationalTargetId("vpi-products"))
    assert connection.committed >= 1


def test_prepare_existing_compatible_table() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection, prepared=False)
    with (
        patch(
            "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema.create_table_ddl",
            return_value="CREATE TABLE IF NOT EXISTS vpi_data_pack_relational_record (id int)",
        ),
        patch(
            "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema.create_identifier_table_ddl",
            return_value="CREATE TABLE IF NOT EXISTS vpi_product_identifiers (id int)",
        ),
        patch(
            "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema.create_identifier_lookup_index_ddl",
            return_value="CREATE INDEX IF NOT EXISTS vpi_product_identifiers_lookup_idx ON vpi_product_identifiers (identifier_type, normalized_value)",
        ),
        patch(
            "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter.verify_identifier_table_compatible",
            return_value=None,
        ),
    ):
        adapter.prepare_target(RelationalTargetId("vpi-products"))
        adapter.prepare_target(RelationalTargetId("vpi-products"))


def test_incompatible_table_rejected() -> None:
    connection = _FakeConnection(
        schema_constraints=[{"constraint_name": "vpi_dpr_source_identity_pk"}],
    )
    adapter = _adapter_with_fake(connection, prepared=False)
    with patch(
        "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema.create_table_ddl",
        return_value="CREATE TABLE IF NOT EXISTS vpi_data_pack_relational_record (id int)",
    ):
        with pytest.raises(PostgreSqlBootstrapSchemaError):
            adapter.prepare_target(RelationalTargetId("vpi-products"))


def test_required_unique_constraints_checked() -> None:
    connection = _FakeConnection(schema_constraints=[])
    adapter = _adapter_with_fake(connection, prepared=False)
    with patch(
        "platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.schema.create_table_ddl",
        return_value="CREATE TABLE IF NOT EXISTS vpi_data_pack_relational_record (id int)",
    ):
        with pytest.raises(PostgreSqlBootstrapSchemaError):
            adapter.prepare_target(RelationalTargetId("vpi-products"))


# --- WRITE ---


def test_one_record_write_pass() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    result = adapter.write_batch(_batch(_record(0)))
    assert result.written_count == 1
    assert result.failed_count == 0


def test_multi_record_batch_pass() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    result = adapter.write_batch(_batch(_record(0), _record(1, offer_suffix="1")))
    assert result.written_count == 2


def test_whole_batch_transaction_atomic_on_conflict() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    first = _record(0)
    second = _record(0, offer_suffix="0", semantic_hash="different-hash")
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(first, second))
    assert connection.rolled_back >= 1
    assert len(connection.storage) == 0


def test_failure_rolls_back_batch() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    good = _record(0)
    bad = _record(0, offer_suffix="0", semantic_hash="changed")
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(good, bad))
    assert len(connection.storage) == 0


def test_requested_written_counts_correct() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    result = adapter.write_batch(_batch(_record(0), _record(1, offer_suffix="1")))
    assert result.requested_count == 2
    assert result.written_count + result.skipped_count == 2


# --- IDEMPOTENCY ---


def test_identical_retry_does_not_duplicate() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    batch = _batch(_record(0))
    adapter.write_batch(batch)
    adapter.write_batch(batch)
    assert len(connection.storage) == 1


def test_identical_retry_reports_skipped() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    batch = _batch(_record(0))
    adapter.write_batch(batch)
    second = adapter.write_batch(batch)
    assert second.skipped_count == 1
    assert second.written_count == 0


def test_same_identity_changed_semantic_hash_fails() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(_batch(_record(0, semantic_hash="hash-a")))
    with pytest.raises(StorageBootstrapWriteError, match="IDENTITY_CONTENT_CONFLICT"):
        adapter.write_batch(_batch(_record(0, semantic_hash="hash-b")))


def test_same_identity_changed_payload_fails() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(_batch(_record(0)))
    changed = _record(0, record_json=json.dumps({"id": "offer-0", "title": "changed"}))
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(changed))


def test_duplicate_global_row_index_with_different_identity_fails() -> None:
    pytest.importorskip("psycopg")
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(_batch(_record(7, offer_suffix="a")))
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(_record(7, offer_suffix="b")))


# --- IDENTITY ---


def test_source_identity_preserved() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    record = _record(3, offer_suffix="preserve")
    adapter.write_batch(_batch(record))
    key = (
        record.source_ref.catalog_id,
        record.source_ref.offer_id.value,
        _source_revision_norm(record.source_ref.source_revision),
    )
    stored = connection.storage[key]
    assert stored["catalog_id"] == record.source_ref.catalog_id
    assert stored["offer_id"] == record.source_ref.offer_id.value


def test_global_row_index_preserved() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    record = _record(42, offer_suffix="idx")
    adapter.write_batch(_batch(record))
    key = (
        record.source_ref.catalog_id,
        record.source_ref.offer_id.value,
        _source_revision_norm(record.source_ref.source_revision),
    )
    assert connection.storage[key]["global_row_index"] == 42


def test_derivation_version_preserved() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    record = _record(1, offer_suffix="dv")
    adapter.write_batch(_batch(record))
    key = (
        record.source_ref.catalog_id,
        record.source_ref.offer_id.value,
        _source_revision_norm(record.source_ref.source_revision),
    )
    assert connection.storage[key]["derivation_version"] == "v1"


def test_semantic_text_hash_preserved() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    record = _record(1, offer_suffix="sth", semantic_hash="hash-xyz")
    adapter.write_batch(_batch(record))
    key = (
        record.source_ref.catalog_id,
        record.source_ref.offer_id.value,
        _source_revision_norm(record.source_ref.source_revision),
    )
    assert connection.storage[key]["semantic_text_hash"] == "hash-xyz"


# --- VERIFICATION ---


def test_verification_pass_after_write() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    batch = _batch(_record(0), _record(1, offer_suffix="1"))
    adapter.write_batch(batch)
    verify = adapter.verify_batch(batch)
    assert verify.failed_count == 0
    assert verify.written_count == 2


def test_missing_identity_detected() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    batch = _batch(_record(0))
    verify = adapter.verify_batch(batch)
    assert verify.failed_count == 1


def test_hash_mismatch_detected() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    record = _record(0)
    adapter.write_batch(_batch(record))
    connection.storage[
        (
            record.source_ref.catalog_id,
            record.source_ref.offer_id.value,
            _source_revision_norm(record.source_ref.source_revision),
        )
    ]["semantic_text_hash"] = "mutated"
    verify = adapter.verify_batch(_batch(record))
    assert verify.failed_count == 1


def test_count_mismatch_detected() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(_batch(_record(0)))
    verify = adapter.verify_batch(_batch(_record(0), _record(1, offer_suffix="1")))
    assert verify.failed_count == 1


# --- SQL SAFETY ---


def test_data_values_parameterized() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(_batch(_record(0, semantic_text="'; DROP TABLE x; --")))
    assert len(connection.storage) == 1


def test_unsafe_logical_target_cannot_become_raw_sql_identifier() -> None:
    with pytest.raises(PostgreSqlBootstrapConfigurationError):
        reject_unsafe_logical_target("vpi_products;drop")


def test_identifier_write_sql_uses_composed_qualified_table() -> None:
    record_json = json.dumps(
        {
            "id": "offer-identifiers",
            "identifiers": [{"/gtin13": "[8806095123456]"}],
        }
    )
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(
        _batch(
            _record(
                0,
                offer_suffix="identifiers",
                record_json=record_json,
            )
        )
    )
    insert_sql = _identifier_insert_sql_text(connection.executed)
    _, _, _, sql_module = import_psycopg()
    composed_statement = next(
        statement for statement, params in connection.executed if len(params) == 8
    )
    assert isinstance(composed_statement, sql_module.Composable)
    assert '"vpi_test_schema"."vpi_product_identifiers"' in insert_sql.lower()


def test_identifier_write_schema_and_table_explicitly_qualified() -> None:
    record_json = json.dumps(
        {
            "id": "offer-identifiers",
            "identifiers": [{"/gtin13": "[8806095123456]"}],
        }
    )
    configuration = PostgreSqlBootstrapConfiguration(
        integration=_configuration().integration,
        schema_name="vpi_alt_schema",
        table_name="vpi_data_pack_relational_record",
        identifier_table_name="vpi_alt_identifiers",
    )

    connection = _FakeConnection()

    def _factory() -> _FakeConnection:
        return connection

    provider = PostgreSQLConnectionProvider(
        configuration.integration,
        tenant_schema=configuration.schema_name,
        connection_factory=_factory,
    )
    provider._apply_search_path_on_connection = lambda _connection: None  # type: ignore[method-assign]
    provider.ensure_schema_exists = lambda _session, _schema_name=None: None  # type: ignore[method-assign]
    adapter = PostgreSqlRelationalStorageAdapter(
        _provider=provider,
        _configuration=configuration,
        _prepared_targets={"vpi-products"},
    )
    adapter.write_batch(
        _batch(
            _record(
                0,
                offer_suffix="identifiers",
                record_json=record_json,
            )
        )
    )
    insert_sql = _identifier_insert_sql_text(connection.executed)
    assert '"vpi_alt_schema"."vpi_alt_identifiers"' in insert_sql.lower()


def test_identifier_write_values_remain_bind_parameters() -> None:
    record_json = json.dumps(
        {
            "id": "offer-identifiers",
            "identifiers": [{"/gtin13": "[8806095123456]"}],
        }
    )
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(
        _batch(
            _record(
                0,
                offer_suffix="identifiers",
                record_json=record_json,
            )
        )
    )
    insert_sql = _identifier_insert_sql_text(connection.executed)
    assert insert_sql.count("%s") == 8
    _, params = next(
        (statement, params) for statement, params in connection.executed if len(params) == 8
    )
    assert len(params) == 8


def test_invalid_identifier_table_configuration_still_rejected() -> None:
    with pytest.raises(ValueError, match="table_name must be a simple SQL identifier"):
        validate_table_identifier("bad-table")


def test_no_f_string_identifier_sql_in_adapter_identifier_write() -> None:
    module_path = _ADAPTER_ROOT / "adapter.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                violations.append(ast.get_source_segment(module_path.read_text(encoding="utf-8"), value) or "")
    identifier_violations = [
        fragment
        for fragment in violations
        if "identifier_table_name" in fragment or "INSERT INTO {" in fragment
    ]
    assert identifier_violations == []


def test_malicious_source_text_remains_data() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    payload = "'; DELETE FROM vpi_data_pack_relational_record; --"
    record = _record(0, semantic_text=payload)
    adapter.write_batch(_batch(record))
    key = (
        record.source_ref.catalog_id,
        record.source_ref.offer_id.value,
        _source_revision_norm(record.source_ref.source_revision),
    )
    assert connection.storage[key]["semantic_text"] == payload


# --- PORT CONTRACT ---


def test_adapter_satisfies_relational_storage_load_port() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    assert hasattr(adapter, "write_batch")
    assert hasattr(adapter, "verify_batch")


def test_result_is_storage_load_batch_result() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    result = adapter.write_batch(_batch(_record(0)))
    assert isinstance(result, StorageLoadBatchResult)


def test_no_postgresql_type_leaks_into_core() -> None:
    violations: list[str] = []
    for module_path in sorted(_CORE_ROOT.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported in {"psycopg", "asyncpg"}:
                violations.append(str(module_path))
    assert violations == []


# --- RESOURCE / ARCHITECTURE ---


@pytest.mark.parametrize("forbidden", sorted(_FORBIDDEN_ADAPTER_IMPORTS))
def test_adapter_has_no_vector_or_model_import(forbidden: str) -> None:
    violations: list[str] = []
    for module_path in sorted(_ADAPTER_ROOT.rglob("*.py")):
        if forbidden in _module_imports(module_path):
            violations.append(str(module_path))
    assert violations == []


def _stored_row_from_record(record: RelationalLoadRecord) -> StoredRelationalRow:
    return StoredRelationalRow(
        catalog_id=record.source_ref.catalog_id,
        offer_id=record.source_ref.offer_id.value,
        source_revision_norm=_source_revision_norm(record.source_ref.source_revision),
        source_revision=record.source_ref.source_revision,
        global_row_index=record.global_row_index,
        record_json=record.record_json,
        semantic_text=record.semantic_text,
        semantic_text_hash=record.semantic_text_hash,
        derivation_version=record.derivation_version,
    )


def test_postgresql_row_converts_to_stored_relational_row() -> None:
    record = _record(0)
    row = {
        "catalog_id": record.source_ref.catalog_id,
        "offer_id": record.source_ref.offer_id.value,
        "source_revision_norm": "",
        "source_revision": None,
        "global_row_index": record.global_row_index,
        "record_json": record.record_json,
        "semantic_text": record.semantic_text,
        "semantic_text_hash": record.semantic_text_hash,
        "derivation_version": record.derivation_version,
    }
    converted = stored_relational_row_from_fetched_row(row)
    assert isinstance(converted, StoredRelationalRow)
    assert converted.offer_id == record.source_ref.offer_id.value


def test_nullable_source_revision_handled_in_row_conversion() -> None:
    record = _record(1, offer_suffix="rev")
    record_with_revision = RelationalLoadRecord(
        source_ref=SourceRecordRef(
            offer_id=record.source_ref.offer_id,
            catalog_id=record.source_ref.catalog_id,
            source_revision="rev-1",
        ),
        global_row_index=record.global_row_index,
        record_json=record.record_json,
        semantic_text=record.semantic_text,
        semantic_text_hash=record.semantic_text_hash,
        derivation_version=record.derivation_version,
    )
    converted = stored_relational_row_from_fetched_row(
        {
            "catalog_id": record_with_revision.source_ref.catalog_id,
            "offer_id": record_with_revision.source_ref.offer_id.value,
            "source_revision_norm": "rev-1",
            "source_revision": "rev-1",
            "global_row_index": record_with_revision.global_row_index,
            "record_json": record_with_revision.record_json,
            "semantic_text": record_with_revision.semantic_text,
            "semantic_text_hash": record_with_revision.semantic_text_hash,
            "derivation_version": record_with_revision.derivation_version,
        }
    )
    assert converted.source_revision == "rev-1"


def test_malformed_row_conversion_fails_closed() -> None:
    with pytest.raises(PostgreSqlBootstrapOperationError, match="missing catalog_id"):
        stored_relational_row_from_fetched_row(
            {
                "offer_id": "offer-0",
                "source_revision_norm": "",
                "source_revision": None,
                "global_row_index": 0,
                "record_json": "{}",
                "semantic_text": "semantic",
                "semantic_text_hash": "hash-a",
                "derivation_version": "v1",
            }
        )


def test_record_payload_matches_helper() -> None:
    record = _record(0)
    existing = _stored_row_from_record(record)
    assert _record_payload_matches(existing, record) is True


def test_identity_conflict_error_type() -> None:
    with pytest.raises(PostgreSqlBootstrapIdentityConflictError):
        raise PostgreSqlBootstrapIdentityConflictError("conflict")


def test_verify_batch_integrity_error_translation() -> None:
    configuration = _configuration()
    provider = PostgreSQLConnectionProvider(
        configuration.integration,
        tenant_schema=configuration.schema_name,
        connection_factory=lambda: (_ for _ in ()).throw(OSError("down")),
    )
    adapter = PostgreSqlRelationalStorageAdapter(
        _provider=provider,
        _configuration=configuration,
        _prepared_targets={"vpi-products"},
    )
    with pytest.raises(StorageBootstrapIntegrityError):
        adapter.verify_batch(_batch(_record(0)))


def test_insert_sql_uses_placeholders_not_interpolation() -> None:
    source = (_ADAPTER_ROOT / "adapter.py").read_text(encoding="utf-8")
    assert re.search(r"VALUES\s*\(%s", source)
    assert "record_json" in source
    assert "ON CONFLICT DO NOTHING" in source


def test_global_row_conflict_does_not_abort_transaction() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(_batch(_record(7, offer_suffix="a")))
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(_record(7, offer_suffix="b")))
    assert connection.in_transaction is False
    assert any(
        "where global_row_index = %s" in str(sql).lower()
        for sql, _params in connection.executed
    )


def test_global_row_conflict_classifies_different_identity() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(_batch(_record(9, offer_suffix="first")))
    with pytest.raises(StorageBootstrapWriteError, match="IDENTITY_CONTENT_CONFLICT"):
        adapter.write_batch(_batch(_record(9, offer_suffix="second")))


def test_source_identity_identical_retry_skipped_without_exception() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    batch = _batch(_record(2, offer_suffix="retry"))
    adapter.write_batch(batch)
    result = adapter.write_batch(batch)
    assert result.skipped_count == 1
    assert result.written_count == 0
    assert not any(
        "where global_row_index = %s" in str(sql).lower()
        for sql, _params in connection.executed[-3:]
        if "insert into" not in str(sql).lower()
    )


def test_insert_conflict_readback_only_after_valid_insert_state() -> None:
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    adapter.write_batch(_batch(_record(11, offer_suffix="row")))
    connection.executed.clear()
    adapter.write_batch(_batch(_record(11, offer_suffix="row")))
    post_insert_sql = [
        (str(sql).lower(), params)
        for sql, params in connection.executed
        if "insert into" not in str(sql).lower()
        and "set transaction" not in str(sql).lower()
        and "set_config" not in str(sql).lower()
    ]
    assert post_insert_sql
    for sql, _params in post_insert_sql:
        assert "where catalog_id = %s" in sql or "where global_row_index = %s" in sql


def test_logical_target_maps_to_approved_table() -> None:
    physical = resolve_physical_target(RelationalTargetId("vpi-products"), _configuration())
    assert physical.table_name == "vpi_data_pack_relational_record"


def test_configuration_table_mismatch_rejected() -> None:
    config = PostgreSqlBootstrapConfiguration(
        integration=_configuration().integration,
        schema_name="vpi_test_schema",
        table_name="other_table",
        identifier_table_name="vpi_product_identifiers",
    )
    with pytest.raises(PostgreSqlBootstrapConfigurationError):
        resolve_physical_target(RelationalTargetId("vpi-products"), config)


# --- SESSION CONFIG ---


def test_adapter_source_has_no_set_local_bind_parameters() -> None:
    source = (_ADAPTER_ROOT / "adapter.py").read_text(encoding="utf-8")
    assert "SET LOCAL" not in source
    assert "set_local_config" in source


def test_apply_session_limits_uses_parameterized_set_config() -> None:
    conn = _FakeConnection()
    configuration = PostgreSqlBootstrapConfiguration(
        integration=_configuration().integration,
        schema_name="vpi_test_schema",
        table_name="vpi_data_pack_relational_record",
        identifier_table_name="vpi_product_identifiers",
        statement_timeout_ms=7500,
        application_name="vpi-relational-bootstrap",
    )
    provider = PostgreSQLConnectionProvider(
        configuration.integration,
        tenant_schema=configuration.schema_name,
        connection_factory=lambda: conn,
    )
    adapter = PostgreSqlRelationalStorageAdapter(
        _provider=provider,
        _configuration=configuration,
        _prepared_targets=set(),
    )
    with provider.connection() as session:
        adapter._apply_session_limits(session)
    set_config_calls = [
        (str(params[0]), str(params[1]))
        for sql, params in conn.executed
        if "set_config" in str(sql).lower() and params
    ]
    assert ("statement_timeout", "7500") in set_config_calls
    assert ("application_name", "vpi-relational-bootstrap") in set_config_calls


def test_apply_session_limits_skips_empty_application_name() -> None:
    conn = _FakeConnection()
    configuration = PostgreSqlBootstrapConfiguration(
        integration=_configuration().integration,
        schema_name="vpi_test_schema",
        table_name="vpi_data_pack_relational_record",
        identifier_table_name="vpi_product_identifiers",
        application_name="",
    )
    provider = PostgreSQLConnectionProvider(
        configuration.integration,
        tenant_schema=configuration.schema_name,
        connection_factory=lambda: conn,
    )
    adapter = PostgreSqlRelationalStorageAdapter(
        _provider=provider,
        _configuration=configuration,
        _prepared_targets=set(),
    )
    with provider.connection() as session:
        adapter._apply_session_limits(session)
    assert all(
        params[0] != "application_name"
        for sql, params in conn.executed
        if "set_config" in str(sql).lower() and params
    )


def test_apply_session_limits_skips_none_statement_timeout() -> None:
    conn = _FakeConnection()
    configuration = PostgreSqlBootstrapConfiguration(
        integration=_configuration().integration,
        schema_name="vpi_test_schema",
        table_name="vpi_data_pack_relational_record",
        identifier_table_name="vpi_product_identifiers",
        statement_timeout_ms=None,
    )
    provider = PostgreSQLConnectionProvider(
        configuration.integration,
        tenant_schema=configuration.schema_name,
        connection_factory=lambda: conn,
    )
    adapter = PostgreSqlRelationalStorageAdapter(
        _provider=provider,
        _configuration=configuration,
        _prepared_targets=set(),
    )
    with provider.connection() as session:
        adapter._apply_session_limits(session)
    assert all(
        params[0] != "statement_timeout"
        for sql, params in conn.executed
        if "set_config" in str(sql).lower() and params
    )


def test_apply_session_limits_hostile_application_name_is_value_only() -> None:
    hostile = "'; DROP TABLE users; --"
    conn = _FakeConnection()
    configuration = PostgreSqlBootstrapConfiguration(
        integration=_configuration().integration,
        schema_name="vpi_test_schema",
        table_name="vpi_data_pack_relational_record",
        identifier_table_name="vpi_product_identifiers",
        application_name=hostile,
    )
    provider = PostgreSQLConnectionProvider(
        configuration.integration,
        tenant_schema=configuration.schema_name,
        connection_factory=lambda: conn,
    )
    adapter = PostgreSqlRelationalStorageAdapter(
        _provider=provider,
        _configuration=configuration,
        _prepared_targets=set(),
    )
    with provider.connection() as session:
        adapter._apply_session_limits(session)
    for sql, params in conn.executed:
        if "set_config" in str(sql).lower() and params and params[0] == "application_name":
            assert hostile not in str(sql)
            assert params[1] == hostile
            return
    raise AssertionError("expected application_name set_config call")


def test_write_batch_persists_identifier_projection_rows() -> None:
    record_json = json.dumps(
        {
            "id": "offer-identifiers",
            "identifiers": [
                {"/gtin13": "[8806095123456]"},
                {"/mpn": "[MZ-V9P2T0BW]"},
            ],
        }
    )
    connection = _FakeConnection()
    adapter = _adapter_with_fake(connection)
    result = adapter.write_batch(
        _batch(
            _record(
                7,
                offer_suffix="identifiers",
                record_json=record_json,
            )
        )
    )
    assert result.written_count == 1
    assert len(connection.identifier_rows) == 2
    identifier_types = {row[4] for row in connection.identifier_rows}
    assert identifier_types == {"gtin", "mpn"}


@dataclass
class _FailSecondIdentifierConnection(_FakeConnection):
    def execute(self, sql: Any, params: tuple[Any, ...] = ()) -> _FakeCursor:
        sql_text = _executed_sql_text(sql).lower()
        if (
            "insert into" in sql_text
            and "normalized_value" in sql_text
            and len(params) == 8
            and self.identifier_rows
        ):
            raise RuntimeError("identifier write failed")
        return super().execute(sql, params)


def test_failed_batch_rolls_back_identifier_projection_rows() -> None:
    ok_json = json.dumps(
        {
            "id": "offer-ok",
            "identifiers": [{"/gtin13": "[8806095123456]"}],
        }
    )
    fail_json = json.dumps(
        {
            "id": "offer-fail",
            "identifiers": [{"/mpn": "[MZ-V9P2T0BW]"}],
        }
    )
    connection = _FailSecondIdentifierConnection()
    adapter = _adapter_with_fake(connection)
    with pytest.raises(RuntimeError, match="identifier write failed"):
        adapter.write_batch(
            _batch(
                _record(10, offer_suffix="ok", record_json=ok_json),
                _record(11, offer_suffix="fail", record_json=fail_json),
            )
        )
    assert connection.rolled_back >= 1
    assert connection.identifier_rows == []
