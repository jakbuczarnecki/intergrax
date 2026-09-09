"""Unit tests for pgvector vector storage bootstrap adapter."""

from __future__ import annotations

import ast
import copy
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.integrations._shared.p2.configs import SqlIntegrationConfig
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    pgvector_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.adapter import (
    PgVectorStorageAdapter,
    _validate_vector_record,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.configuration import (
    CANONICAL_EMBEDDING_DIMENSION,
    CANONICAL_EMBEDDING_MODEL,
    CANONICAL_EMBEDDING_PROVIDER,
    CANONICAL_EMBEDDING_REVISION,
    ExpectedVectorIdentity,
    PgVectorBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.connection import (
    PgVectorConnectionProvider,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.errors import (
    PgVectorBootstrapConfigurationError,
    PgVectorBootstrapIdentityConflictError,
    PgVectorBootstrapSchemaError,
    PgVectorBootstrapVectorValidationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.stored_row import (
    normalize_vector_float32,
    record_matches_stored,
    stored_row_from_record,
    stored_row_identity_matches,
    vectors_transport_equal,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.target_mapping import (
    reject_unsafe_logical_target,
    resolve_physical_target,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapFinalStatus,
    StorageLoadBatchResult,
    VectorBatch,
    VectorLoadRecord,
    VectorTargetId,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapIntegrityError,
    StorageBootstrapWriteError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.service import (
    StorageBootstrapDependencies,
    StorageBootstrapService,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_storage_bootstrap_data_pack_load import (
    FakeDataPackReader,
    FakeRelationalAdapter,
    FakeVectorAdapter,
    _build_pairs,
    _ready_manifest,
    _request,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_ADAPTER_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/adapters/pgvector"
)
_CORE_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/data_pack_load"
)
_QDRANT_ADAPTER_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/adapters/qdrant"
)
_FORBIDDEN_ADAPTER_IMPORTS = frozenset(
    {
        "qdrant",
        "qdrant_client",
        "torch",
        "transformers",
        "sentence_transformers",
    }
)
_FORBIDDEN_CORE_IMPORTS = frozenset({"qdrant", "qdrant_client", "pgvector"})


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


def _unit_vector(dimension: int = CANONICAL_EMBEDDING_DIMENSION, scale: float = 1.0) -> tuple[float, ...]:
    values = [0.0] * dimension
    values[0] = scale
    return tuple(values)


def _vector_record(
    index: int,
    *,
    offer_suffix: str | None = None,
    semantic_hash: str = "hash-a",
    vector: tuple[float, ...] | None = None,
    model: str = CANONICAL_EMBEDDING_MODEL,
    revision: str = CANONICAL_EMBEDDING_REVISION,
    provider: str = CANONICAL_EMBEDDING_PROVIDER,
    dimension: int = CANONICAL_EMBEDDING_DIMENSION,
) -> VectorLoadRecord:
    suffix = offer_suffix or str(index)
    source_ref = _source_ref(suffix)
    logical_point_id = search_representation_point_id(
        catalog_id=source_ref.catalog_id,
        offer_id=source_ref.offer_id.value,
        derivation_version="v1",
    )
    dense = vector if vector is not None else _unit_vector(dimension)
    return VectorLoadRecord(
        logical_point_id=logical_point_id,
        source_ref=source_ref,
        semantic_text_hash=semantic_hash,
        embedding_provider=provider,
        embedding_model=model,
        embedding_revision=revision,
        embedding_dimension=dimension,
        dense_embedding=dense,
        derivation_version="v1",
    )


def _batch(*records: VectorLoadRecord, batch_number: int = 0) -> VectorBatch:
    return VectorBatch(
        batch_number=batch_number,
        target=VectorTargetId("vpi-product-embeddings"),
        records=tuple(records),
    )


def _configuration() -> PgVectorBootstrapConfiguration:
    sql_integration = SqlIntegrationConfig(
        dsn="postgresql://user:secret-password@localhost:5432/vpi",
        tenant_schema="vpi_vectors",
    )
    return PgVectorBootstrapConfiguration(
        sql_integration=sql_integration,
        schema_name="vpi_vectors",
        table_name="vpi_data_pack_vector_embedding",
        expected_vector_identity=ExpectedVectorIdentity.canonical_vpi(),
        insert_batch_size=64,
        allow_create_extension=True,
    )


@dataclass
class _FakeCursor:
    query: str
    params: Sequence[object]
    backend: _FakePgVectorBackend


@dataclass
class _FakePgVectorBackend:
    rows: dict[str, dict[str, object]] = field(default_factory=dict)
    extension_installed: bool = True
    table_prepared: bool = False
    dimension: int = CANONICAL_EMBEDDING_DIMENSION
    fail_insert_for: str | None = None
    transaction_active: bool = False
    staged_rows: dict[str, dict[str, object]] = field(default_factory=dict)
    query_log: list[str] = field(default_factory=list)

    def active_rows(self) -> dict[str, dict[str, object]]:
        return self.staged_rows if self.transaction_active else self.rows

    def begin_transaction(self) -> None:
        self.transaction_active = True
        self.staged_rows = copy.deepcopy(self.rows)

    def commit(self) -> None:
        if self.transaction_active:
            self.rows = copy.deepcopy(self.staged_rows)
            self.staged_rows = {}
            self.transaction_active = False

    def rollback(self) -> None:
        self.staged_rows = {}
        self.transaction_active = False

    def execute(self, query: str, params: Sequence[object] = ()) -> _FakeCursor:
        normalized = " ".join(query.strip().lower().split())
        self.query_log.append(normalized)
        if normalized.startswith("set transaction isolation level"):
            self.begin_transaction()
            return _FakeCursor(query, params, self)
        if normalized.startswith("set local application_name"):
            return _FakeCursor(query, params, self)
        if normalized.startswith("set search_path"):
            return _FakeCursor(query, params, self)
        if "create schema if not exists" in normalized:
            return _FakeCursor(query, params, self)
        if "create extension if not exists vector" in normalized:
            if not self.extension_installed:
                self.extension_installed = True
            return _FakeCursor(query, params, self)
        if normalized.startswith("select exists (select 1 from pg_extension"):
            return _FakeExistsCursor(self.extension_installed, self)
        if "create table if not exists" in normalized:
            self.table_prepared = True
            return _FakeCursor(query, params, self)
        if "from information_schema.columns" in normalized:
            return _FakeColumnCursor(self.table_prepared, self)
        if "from pg_attribute" in normalized:
            return _FakeVectorTypeCursor(self.dimension, self)
        if "from information_schema.table_constraints" in normalized:
            return _FakeConstraintCursor(self.table_prepared, self)
        if normalized.startswith("insert into"):
            logical_point_id = str(params[0])
            if self.fail_insert_for == logical_point_id:
                raise OSError("simulated insert failure")
            row = {
                "logical_point_id": params[0],
                "catalog_id": params[1],
                "offer_id": params[2],
                "source_revision_norm": params[3],
                "source_revision": params[4],
                "semantic_text_hash": params[5],
                "embedding_provider": params[6],
                "embedding_model": params[7],
                "embedding_revision": params[8],
                "embedding_dimension": params[9],
                "derivation_version": params[10],
                "dense_embedding": list(params[11]),
            }
            self.active_rows()[logical_point_id] = row
            return _FakeRowCountCursor(1, self)
        if "where logical_point_id = any" in normalized:
            requested = {str(value) for value in params[0]}
            matches = [
                row
                for row in self.active_rows().values()
                if str(row["logical_point_id"]) in requested
            ]
            return _FakeSelectCursor(matches, self)
        return _FakeCursor(query, params, self)

    def close(self) -> None:
        return None


@dataclass
class _FakeExistsCursor:
    value: bool
    backend: _FakePgVectorBackend

    def fetchone(self) -> Mapping[str, object]:
        return {"exists": self.value}

    def fetchall(self) -> list[Mapping[str, object]]:
        return []


@dataclass
class _FakeColumnCursor:
    table_prepared: bool
    backend: _FakePgVectorBackend

    def fetchone(self) -> Mapping[str, object] | None:
        return None

    def fetchall(self) -> list[Mapping[str, object]]:
        if not self.table_prepared:
            return []
        return [
            {"column_name": "logical_point_id", "data_type": "text", "is_nullable": "NO", "udt_name": "text"},
            {"column_name": "catalog_id", "data_type": "text", "is_nullable": "NO", "udt_name": "text"},
            {"column_name": "offer_id", "data_type": "text", "is_nullable": "NO", "udt_name": "text"},
            {"column_name": "source_revision_norm", "data_type": "text", "is_nullable": "NO", "udt_name": "text"},
            {"column_name": "source_revision", "data_type": "text", "is_nullable": "YES", "udt_name": "text"},
            {"column_name": "semantic_text_hash", "data_type": "text", "is_nullable": "NO", "udt_name": "text"},
            {"column_name": "embedding_provider", "data_type": "text", "is_nullable": "NO", "udt_name": "text"},
            {"column_name": "embedding_model", "data_type": "text", "is_nullable": "NO", "udt_name": "text"},
            {"column_name": "embedding_revision", "data_type": "text", "is_nullable": "YES", "udt_name": "text"},
            {"column_name": "embedding_dimension", "data_type": "integer", "is_nullable": "NO", "udt_name": "int4"},
            {"column_name": "derivation_version", "data_type": "text", "is_nullable": "NO", "udt_name": "text"},
            {
                "column_name": "dense_embedding",
                "data_type": "USER-DEFINED",
                "is_nullable": "NO",
                "udt_name": "vector",
            },
        ]


@dataclass
class _FakeVectorTypeCursor:
    dimension: int
    backend: _FakePgVectorBackend

    def fetchone(self) -> Mapping[str, object]:
        return {"vector_type": f"vector({self.dimension})"}

    def fetchall(self) -> list[Mapping[str, object]]:
        return []


@dataclass
class _FakeConstraintCursor:
    table_prepared: bool
    backend: _FakePgVectorBackend

    def fetchall(self) -> list[Mapping[str, object]]:
        if not self.table_prepared:
            return []
        return [
            {"constraint_name": "vpi_dpv_logical_point_id_pk"},
            {"constraint_name": "vpi_dpv_source_identity_uq"},
        ]


@dataclass
class _FakeRowCountCursor:
    count: int
    backend: _FakePgVectorBackend

    @property
    def rowcount(self) -> int:
        return self.count

    def fetchone(self) -> Mapping[str, object] | None:
        return None

    def fetchall(self) -> list[Mapping[str, object]]:
        return []


@dataclass
class _FakeSelectCursor:
    rows: list[dict[str, object]]
    backend: _FakePgVectorBackend

    def fetchone(self) -> Mapping[str, object] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[Mapping[str, object]]:
        return self.rows


@dataclass
class _FakeConnection:
    backend: _FakePgVectorBackend

    def execute(self, query: str | object, params: Sequence[object] = ()) -> _FakeCursor:
        if not isinstance(query, str):
            self.backend.table_prepared = True
            return _FakeCursor("composed ddl", params, self.backend)
        return self.backend.execute(query, params)

    def commit(self) -> None:
        self.backend.commit()

    def rollback(self) -> None:
        self.backend.rollback()

    def close(self) -> None:
        self.backend.close()


def _adapter_with_fake(
    backend: _FakePgVectorBackend,
    *,
    configuration: PgVectorBootstrapConfiguration | None = None,
    prepared: bool = False,
) -> PgVectorStorageAdapter:
    config = configuration or _configuration()
    provider = PgVectorConnectionProvider(
        sql_integration=config.sql_integration,
        schema_name=config.schema_name,
        connection_factory=lambda: _FakeConnection(backend),
    )
    adapter = PgVectorStorageAdapter(provider=provider, configuration=config)
    if prepared:
        adapter._prepared_targets.add("vpi-product-embeddings")
    return adapter


# --- CONFIGURATION ---


def test_valid_typed_pgvector_configuration() -> None:
    config = _configuration()
    assert config.expected_vector_identity.dimension == 1024


def test_configuration_repr_excludes_secrets() -> None:
    text = repr(_configuration())
    assert "secret-password" not in text
    assert "password" not in text


def test_dimension_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="dimension does not match"):
        PgVectorBootstrapConfiguration.from_env(
            schema_name="vpi_vectors",
            dimension=768,
            expected_vector_identity=ExpectedVectorIdentity.canonical_vpi(),
        )


def test_unsafe_logical_target_rejected() -> None:
    with pytest.raises(PgVectorBootstrapConfigurationError):
        reject_unsafe_logical_target("vpi_vectors;drop")


# --- TARGET ---


def test_logical_target_maps_to_approved_table() -> None:
    physical = resolve_physical_target(VectorTargetId("vpi-product-embeddings"), _configuration())
    assert physical.table_name == "vpi_data_pack_vector_embedding"


def test_configuration_table_mismatch_rejected() -> None:
    config = PgVectorBootstrapConfiguration(
        sql_integration=_configuration().sql_integration,
        schema_name="vpi_vectors",
        table_name="other-table",
        expected_vector_identity=ExpectedVectorIdentity.canonical_vpi(),
    )
    with pytest.raises(PgVectorBootstrapConfigurationError):
        resolve_physical_target(VectorTargetId("vpi-product-embeddings"), config)


# --- EXTENSION ---


def test_prepare_target_with_extension_available() -> None:
    backend = _FakePgVectorBackend(extension_installed=True)
    adapter = _adapter_with_fake(backend)
    adapter.prepare_target(VectorTargetId("vpi-product-embeddings"))
    assert backend.table_prepared is True


def test_prepare_target_extension_unavailable_fails_closed() -> None:
    backend = _FakePgVectorBackend(extension_installed=False)
    config = PgVectorBootstrapConfiguration(
        sql_integration=_configuration().sql_integration,
        schema_name="vpi_vectors",
        table_name="vpi_data_pack_vector_embedding",
        expected_vector_identity=ExpectedVectorIdentity.canonical_vpi(),
        allow_create_extension=False,
    )
    adapter = _adapter_with_fake(backend, configuration=config)
    with pytest.raises(PgVectorBootstrapSchemaError, match="PGVECTOR_EXTENSION_UNAVAILABLE"):
        adapter.prepare_target(VectorTargetId("vpi-product-embeddings"))


# --- SCHEMA ---


def test_prepare_target_creates_compatible_table() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend)
    adapter.prepare_target(VectorTargetId("vpi-product-embeddings"))
    assert "vpi-product-embeddings" in adapter._prepared_targets


# --- VECTOR VALIDATION ---


def test_valid_1024_vector_passes() -> None:
    _validate_vector_record(_vector_record(0), ExpectedVectorIdentity.canonical_vpi())


def test_wrong_dimension_rejected() -> None:
    record = _vector_record(0, dimension=768, vector=_unit_vector(768))
    with pytest.raises(PgVectorBootstrapVectorValidationError, match="VECTOR_IDENTITY_INCOMPATIBLE"):
        _validate_vector_record(record, ExpectedVectorIdentity.canonical_vpi())


def test_nan_rejected() -> None:
    vector = list(_unit_vector())
    vector[1] = math.nan
    record = _vector_record(0, vector=tuple(vector))
    with pytest.raises(PgVectorBootstrapVectorValidationError):
        _validate_vector_record(record, ExpectedVectorIdentity.canonical_vpi())


def test_positive_infinity_rejected() -> None:
    vector = list(_unit_vector())
    vector[1] = math.inf
    with pytest.raises(PgVectorBootstrapVectorValidationError):
        _validate_vector_record(_vector_record(0, vector=tuple(vector)), ExpectedVectorIdentity.canonical_vpi())


def test_negative_infinity_rejected() -> None:
    vector = list(_unit_vector())
    vector[1] = -math.inf
    with pytest.raises(PgVectorBootstrapVectorValidationError):
        _validate_vector_record(_vector_record(0, vector=tuple(vector)), ExpectedVectorIdentity.canonical_vpi())


def test_zero_vector_rejected() -> None:
    record = _vector_record(0, vector=tuple(0.0 for _ in range(CANONICAL_EMBEDDING_DIMENSION)))
    with pytest.raises(PgVectorBootstrapVectorValidationError):
        _validate_vector_record(record, ExpectedVectorIdentity.canonical_vpi())


# --- IDENTITY ---


def test_deterministic_logical_point_id_mapping() -> None:
    record = _vector_record(0)
    assert record.logical_point_id == _vector_record(0).logical_point_id


def test_payload_source_identity_preserved() -> None:
    row = stored_row_from_record(_vector_record(0))
    record = _vector_record(0)
    assert row.catalog_id == record.source_ref.catalog_id
    assert row.offer_id == record.source_ref.offer_id.value


def test_payload_semantic_hash_preserved() -> None:
    row = stored_row_from_record(_vector_record(0, semantic_hash="hash-xyz"))
    assert row.semantic_text_hash == "hash-xyz"


def test_payload_embedding_provider_preserved() -> None:
    assert stored_row_from_record(_vector_record(0)).embedding_provider == CANONICAL_EMBEDDING_PROVIDER


def test_payload_model_preserved() -> None:
    assert stored_row_from_record(_vector_record(0)).embedding_model == CANONICAL_EMBEDDING_MODEL


def test_payload_revision_preserved() -> None:
    assert stored_row_from_record(_vector_record(0)).embedding_revision == CANONICAL_EMBEDDING_REVISION


def test_payload_derivation_version_preserved() -> None:
    assert stored_row_from_record(_vector_record(0)).derivation_version == "v1"


# --- WRITE ---


def test_new_vector_written() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    result = adapter.write_batch(_batch(_vector_record(0)))
    assert result.written_count == 1
    assert result.skipped_count == 0


def test_batch_write_passes() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    result = adapter.write_batch(_batch(_vector_record(0), _vector_record(1, offer_suffix="1")))
    assert result.is_complete_success


def test_existing_identical_skipped() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    result = adapter.write_batch(_batch(record))
    assert result.written_count == 0
    assert result.skipped_count == 1
    assert len(backend.rows) == 1


def test_same_point_different_semantic_hash_fails() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    record = _vector_record(0, semantic_hash="hash-a")
    adapter.write_batch(_batch(record))
    with pytest.raises(StorageBootstrapWriteError, match="VECTOR_CONTENT_CONFLICT"):
        adapter.write_batch(_batch(_vector_record(0, semantic_hash="hash-b")))


def test_same_point_different_model_identity_fails() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    backend.rows[record.logical_point_id]["embedding_model"] = "other-model"
    with pytest.raises(StorageBootstrapWriteError, match="VECTOR_CONTENT_CONFLICT"):
        adapter.write_batch(_batch(record))


def test_incoming_model_identity_incompatible_fails() -> None:
    adapter = _adapter_with_fake(_FakePgVectorBackend(), prepared=True)
    with pytest.raises(StorageBootstrapWriteError, match="VECTOR_IDENTITY_INCOMPATIBLE"):
        adapter.write_batch(_batch(_vector_record(0, model="other-model")))


def test_same_point_different_vector_fails() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    vector = list(_unit_vector())
    vector[0] = 0.5
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(_vector_record(0, vector=tuple(vector))))


def test_blind_overwrite_impossible() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    stored_vector = backend.rows[record.logical_point_id]["dense_embedding"]
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(_vector_record(0, semantic_hash="other")))
    assert backend.rows[record.logical_point_id]["dense_embedding"] == stored_vector


# --- TRANSACTION ---


def test_vector_batch_failure_rolls_back_new_rows() -> None:
    backend = _FakePgVectorBackend()
    record_a = _vector_record(0)
    record_b = _vector_record(1, offer_suffix="1")
    backend.fail_insert_for = record_b.logical_point_id
    adapter = _adapter_with_fake(backend, prepared=True)
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(record_a, record_b))
    assert backend.rows == {}


# --- VERIFY ---


def test_verify_expected_point_exists() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    verify = adapter.verify_batch(_batch(record))
    assert verify.failed_count == 0


def test_verify_missing_point_detected() -> None:
    adapter = _adapter_with_fake(_FakePgVectorBackend(), prepared=True)
    verify = adapter.verify_batch(_batch(_vector_record(0)))
    assert verify.failed_count == 1


def test_verify_metadata_mismatch_detected() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    backend.rows[record.logical_point_id]["semantic_text_hash"] = "mutated"
    verify = adapter.verify_batch(_batch(record))
    assert verify.failed_count == 1


def test_verify_vector_mismatch_detected() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    vector = list(backend.rows[record.logical_point_id]["dense_embedding"])
    vector[0] = 0.25
    backend.rows[record.logical_point_id]["dense_embedding"] = vector
    verify = adapter.verify_batch(_batch(record))
    assert verify.failed_count == 1


def test_verify_uses_bounded_batch_query() -> None:
    backend = _FakePgVectorBackend()
    adapter = _adapter_with_fake(backend, prepared=True)
    records = tuple(_vector_record(index, offer_suffix=str(index)) for index in range(3))
    adapter.write_batch(_batch(*records))
    backend.query_log.clear()
    adapter.verify_batch(_batch(*records))
    assert any("where logical_point_id = any" in query for query in backend.query_log)


# --- PORT / ARCHITECTURE ---


def test_adapter_satisfies_vector_storage_load_port() -> None:
    adapter = _adapter_with_fake(_FakePgVectorBackend())
    assert hasattr(adapter, "write_batch")
    assert hasattr(adapter, "verify_batch")


def test_result_is_storage_load_batch_result() -> None:
    adapter = _adapter_with_fake(_FakePgVectorBackend(), prepared=True)
    result = adapter.write_batch(_batch())
    assert isinstance(result, StorageLoadBatchResult)


def test_no_pgvector_type_leaks_into_core() -> None:
    violations: list[str] = []
    for module_path in sorted(_CORE_ROOT.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported in _FORBIDDEN_CORE_IMPORTS:
                violations.append(str(module_path))
    assert violations == []


@pytest.mark.parametrize("forbidden", sorted(_FORBIDDEN_ADAPTER_IMPORTS))
def test_pgvector_adapter_has_no_qdrant_or_model_imports(forbidden: str) -> None:
    violations: list[str] = []
    for module_path in sorted(_ADAPTER_ROOT.rglob("*.py")):
        if forbidden in _module_imports(module_path):
            violations.append(str(module_path))
    assert violations == []


def test_qdrant_adapter_has_no_pgvector_import() -> None:
    violations: list[str] = []
    for module_path in sorted(_QDRANT_ADAPTER_ROOT.rglob("*.py")):
        if "pgvector" in _module_imports(module_path):
            violations.append(str(module_path))
    assert violations == []


def test_payload_identity_helper() -> None:
    left = stored_row_from_record(_vector_record(0))
    right = stored_row_from_record(_vector_record(0))
    assert stored_row_identity_matches(left, right) is True


def test_float32_transport_equality() -> None:
    left = normalize_vector_float32((1.0, 0.0))
    right = normalize_vector_float32((1.0, 0.0))
    assert vectors_transport_equal(left, right, tolerance=0.0) is True


def test_record_matches_stored_helper() -> None:
    record = _vector_record(0)
    stored = stored_row_from_record(record)
    assert record_matches_stored(record, stored, tolerance=0.0) is True


def test_adapter_has_no_forbidden_contract_patterns() -> None:
    forbidden_fragments = (
        ": object",
        "-> object",
        "Sequence[object]",
        "dict[str, object]",
        "Mapping[str, object]",
        ": Any",
        "dict[str, Any]",
    )
    for module_path in sorted(_ADAPTER_ROOT.rglob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {module_path.name}"


def test_identity_conflict_error_type() -> None:
    with pytest.raises(PgVectorBootstrapIdentityConflictError):
        raise PgVectorBootstrapIdentityConflictError("conflict")


def test_verify_batch_integrity_error_translation() -> None:
    backend = _FakePgVectorBackend()
    original_execute = backend.execute

    def failing_execute(query: str, params: Sequence[object] = ()) -> _FakeCursor:
        if "where logical_point_id = any" in " ".join(query.lower().split()):
            raise OSError("down")
        return original_execute(query, params)

    backend.execute = failing_execute  # type: ignore[method-assign]
    adapter = _adapter_with_fake(backend, prepared=True)
    with pytest.raises(StorageBootstrapIntegrityError):
        adapter.verify_batch(_batch(_vector_record(0)))


# --- PORTABILITY ---


def _portability_configuration() -> PgVectorBootstrapConfiguration:
    return PgVectorBootstrapConfiguration(
        sql_integration=_configuration().sql_integration,
        schema_name="vpi_vectors",
        table_name="vpi_data_pack_vector_embedding",
        expected_vector_identity=ExpectedVectorIdentity(
            provider=CANONICAL_EMBEDDING_PROVIDER,
            model=CANONICAL_EMBEDDING_MODEL,
            revision=CANONICAL_EMBEDDING_REVISION,
            dimension=8,
        ),
    )


def test_same_storage_bootstrap_service_with_pgvector_adapter() -> None:
    pairs = _build_pairs(2)
    reader = FakeDataPackReader(manifest=_ready_manifest(2), pairs=pairs)
    backend = _FakePgVectorBackend(dimension=8)
    vector = _adapter_with_fake(backend, configuration=_portability_configuration())
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=FakeRelationalAdapter(),
            vector=vector,
        )
    )
    result = service.run(_request(batch_size=2))
    assert result.status is BootstrapFinalStatus.SUCCESS


def test_same_storage_bootstrap_service_with_qdrant_compatible_fake() -> None:
    pairs = _build_pairs(2)
    reader = FakeDataPackReader(manifest=_ready_manifest(2), pairs=pairs)
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=FakeRelationalAdapter(),
            vector=FakeVectorAdapter(adapter_label="qdrant-compatible"),
        )
    )
    result = service.run(_request(batch_size=2))
    assert result.status is BootstrapFinalStatus.SUCCESS


# --- REAL PGVECTOR (optional) ---


@pytest.mark.integration
def test_real_pgvector_bounded_qualification() -> None:
    if not pgvector_environment_available():
        pytest.skip("pgvector environment unavailable")

    schema_name = f"vpi_pg_bootstrap_{uuid.uuid4().hex[:10]}"
    adapter = PgVectorStorageAdapter.from_env(schema_name=schema_name)
    target = VectorTargetId("vpi-product-embeddings")
    try:
        adapter.prepare_target(target)
        record = _vector_record(0)
        first = adapter.write_batch(_batch(record))
        assert first.written_count == 1
        second = adapter.write_batch(_batch(record))
        assert second.skipped_count == 1
        verify = adapter.verify_batch(_batch(record))
        assert verify.failed_count == 0
        with pytest.raises(StorageBootstrapWriteError):
            adapter.write_batch(_batch(_vector_record(0, semantic_hash="other")))
    finally:
        provider = adapter._provider
        with provider.connection() as session:
            session.execute(f"DROP TABLE IF EXISTS {adapter._configuration.table_name}")
            if schema_name != "public":
                session.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
            session.commit()
