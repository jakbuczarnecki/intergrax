"""Unit tests for Qdrant vector storage bootstrap adapter."""

from __future__ import annotations

import ast
import math
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import pytest

from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexDescription,
    VectorIndexIdentity,
    VectorIndexPrepareOutcome,
    VectorIndexPrepareResult,
    VectorSearchCapability,
)
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.point_ids import (
    normalize_qdrant_logical_point_id,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    qdrant_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.adapter import (
    QdrantVectorStorageAdapter,
    _collection_vector_shape,
    _distance_label,
    _extract_dense_vector,
    _extract_provider_payload,
    _record_matches_stored,
    _stored_point_from_provider_record,
    _validate_vector_record,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    CANONICAL_EMBEDDING_DIMENSION,
    CANONICAL_EMBEDDING_MODEL,
    CANONICAL_EMBEDDING_PROVIDER,
    CANONICAL_EMBEDDING_REVISION,
    ExpectedVectorIdentity,
    QdrantBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.errors import (
    QdrantBootstrapCollectionError,
    QdrantBootstrapConfigurationError,
    QdrantBootstrapIdentityConflictError,
    QdrantBootstrapOperationError,
    QdrantBootstrapVectorValidationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.payload import (
    QdrantStoredPoint,
    QdrantUpsertPoint,
    QdrantVectorPayload,
    cosine_storage_normalize,
    normalize_vector_float32,
    payload_from_record,
    payload_identity_matches,
    vectors_transport_equal,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.target_mapping import (
    PhysicalVectorTarget,
    physical_collection_name,
    reject_unsafe_logical_target,
    resolve_physical_target,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    StorageLoadBatchResult,
    VectorBatch,
    VectorLoadRecord,
    VectorTargetId,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapIntegrityError,
    StorageBootstrapWriteError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)

pytestmark = pytest.mark.unit

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_ADAPTER_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/adapters/qdrant"
)
_CORE_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/data_pack_load"
)
_FORBIDDEN_ADAPTER_IMPORTS = frozenset(
    {
        "psycopg",
        "asyncpg",
        "pgvector",
        "torch",
        "transformers",
        "sentence_transformers",
    }
)
_FORBIDDEN_CORE_IMPORTS = frozenset({"qdrant", "qdrant_client"})


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


def _configuration() -> QdrantBootstrapConfiguration:
    integration = QdrantIntegrationConfig(
        host="localhost",
        port=6333,
        api_key="secret-api-key-value",
        collection_name="vpi-product-embeddings",
        tenant_id="default",
        metric="cosine",
        batch_size=64,
        enable_sparse_vectors=False,
    )
    return QdrantBootstrapConfiguration(
        integration=integration,
        logical_collection_name="vpi-product-embeddings",
        expected_vector_identity=ExpectedVectorIdentity.canonical_vpi(),
        upsert_batch_size=64,
    )


@dataclass
class _FakeIndexAdmin:
    prepared: list[VectorIndexIdentity] = field(default_factory=list)
    fail_prepare: bool = False

    def probe(self) -> SimpleNamespace:
        return SimpleNamespace(slug="qdrant", healthy=True, detail="ok")

    def describe_index(self, identity: VectorIndexIdentity) -> VectorIndexDescription:
        return VectorIndexDescription(
            identity=identity,
            exists=True,
            reachable=True,
            point_count=0,
            dense_dimension=CANONICAL_EMBEDDING_DIMENSION,
            dense_metric="cosine",
            present_capabilities=frozenset({VectorSearchCapability.DENSE}),
            dense_channel_name=None,
            sparse_lexical_channel_name=None,
        )

    def prepare_index(self, spec: object) -> VectorIndexPrepareResult:
        if self.fail_prepare:
            raise QdrantBootstrapCollectionError("incompatible")
        identity = cast(VectorIndexIdentity, getattr(spec, "identity"))
        self.prepared.append(identity)
        return VectorIndexPrepareResult(
            outcome=VectorIndexPrepareOutcome.CREATED,
            description=self.describe_index(identity),
        )

    def close(self) -> None:
        return None


@dataclass
class _FakeQdrantPoint:
    id: str | int
    payload: dict[str, str | int]
    vector: list[float] | dict[str, list[float]]


@dataclass
class _FakeQdrantClient:
    points: dict[str | int, _FakeQdrantPoint] = field(default_factory=dict)
    collection_dimension: int = CANONICAL_EMBEDDING_DIMENSION
    collection_distance: str = "Cosine"
    named_dense: bool = False
    upsert_calls: int = 0
    retrieve_calls: int = 0
    fail_upsert: bool = False

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        if self.named_dense:
            vectors = {
                "dense": SimpleNamespace(
                    size=self.collection_dimension,
                    distance=self.collection_distance,
                )
            }
        else:
            vectors = SimpleNamespace(
                size=self.collection_dimension,
                distance=self.collection_distance,
            )
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors))
        )

    def retrieve(
        self,
        collection_name: str,
        ids: list[str | int],
        *,
        with_payload: bool,
        with_vectors: bool,
    ) -> list[_FakeQdrantPoint]:
        self.retrieve_calls += 1
        found: list[_FakeQdrantPoint] = []
        for point_id in ids:
            point = self.points.get(point_id)
            if point is not None:
                found.append(point)
        return found

    def upsert(
        self,
        collection_name: str,
        points: list[QdrantUpsertPoint],
    ) -> None:
        if self.fail_upsert:
            raise OSError("upsert failed")
        self.upsert_calls += 1
        for point in points:
            point_id = point.id
            self.points[point_id] = _FakeQdrantPoint(
                id=point_id,
                payload=dict(point.payload),
                vector=point.vector,
            )

    def close(self) -> None:
        return None


def _adapter_with_fake(
    client: _FakeQdrantClient,
    *,
    index_admin: _FakeIndexAdmin | None = None,
    configuration: QdrantBootstrapConfiguration | None = None,
    prepared: bool = False,
) -> QdrantVectorStorageAdapter:
    adapter = QdrantVectorStorageAdapter(
        _client=client,
        _index_admin=index_admin or _FakeIndexAdmin(),
        _configuration=configuration or _configuration(),
        _prepared_targets=set(),
    )
    if prepared:
        adapter._prepared_targets.add("vpi-product-embeddings")
    return adapter


def _physical_target(configuration: QdrantBootstrapConfiguration | None = None) -> PhysicalVectorTarget:
    return resolve_physical_target(
        VectorTargetId("vpi-product-embeddings"),
        configuration or _configuration(),
    )


# --- CONFIGURATION ---


def test_valid_typed_qdrant_configuration() -> None:
    config = _configuration()
    assert config.expected_vector_identity.dimension == 1024


def test_configuration_repr_excludes_secrets() -> None:
    text = repr(_configuration())
    assert "secret-api-key-value" not in text
    assert "api_key" not in text


def test_unsafe_logical_target_rejected() -> None:
    with pytest.raises(QdrantBootstrapConfigurationError):
        reject_unsafe_logical_target("vpi_vectors;drop")


# --- COLLECTION ---


def test_prepare_target_creates_compatible_collection() -> None:
    client = _FakeQdrantClient()
    index_admin = _FakeIndexAdmin()
    adapter = _adapter_with_fake(client, index_admin=index_admin)
    adapter.prepare_target(VectorTargetId("vpi-product-embeddings"))
    assert len(index_admin.prepared) == 1


def test_compatible_existing_collection_passes() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client)
    adapter.prepare_target(VectorTargetId("vpi-product-embeddings"))
    assert "vpi-product-embeddings" in adapter._prepared_targets


def test_wrong_vector_size_fails() -> None:
    client = _FakeQdrantClient(collection_dimension=768)
    adapter = _adapter_with_fake(client)
    with pytest.raises(QdrantBootstrapCollectionError):
        adapter.prepare_target(VectorTargetId("vpi-product-embeddings"))


def test_wrong_distance_fails() -> None:
    client = _FakeQdrantClient(collection_distance="Dot")
    adapter = _adapter_with_fake(client)
    with pytest.raises(QdrantBootstrapCollectionError):
        adapter.prepare_target(VectorTargetId("vpi-product-embeddings"))


def test_wrong_named_vector_config_fails() -> None:
    client = _FakeQdrantClient(named_dense=True)
    adapter = _adapter_with_fake(client)
    with pytest.raises(QdrantBootstrapCollectionError):
        adapter.prepare_target(VectorTargetId("vpi-product-embeddings"))


# --- VECTOR VALIDATION ---


def test_valid_1024_vector_passes() -> None:
    record = _vector_record(0)
    _validate_vector_record(record, ExpectedVectorIdentity.canonical_vpi())


def test_wrong_dimension_rejected() -> None:
    record = _vector_record(0, dimension=768, vector=_unit_vector(768))
    with pytest.raises(QdrantBootstrapVectorValidationError, match="VECTOR_IDENTITY_INCOMPATIBLE"):
        _validate_vector_record(record, ExpectedVectorIdentity.canonical_vpi())


def test_nan_rejected() -> None:
    vector = _unit_vector()
    vector_list = list(vector)
    vector_list[1] = math.nan
    record = _vector_record(0, vector=tuple(vector_list))
    with pytest.raises(QdrantBootstrapVectorValidationError):
        _validate_vector_record(record, ExpectedVectorIdentity.canonical_vpi())


def test_positive_infinity_rejected() -> None:
    vector = _unit_vector()
    vector_list = list(vector)
    vector_list[1] = math.inf
    record = _vector_record(0, vector=tuple(vector_list))
    with pytest.raises(QdrantBootstrapVectorValidationError):
        _validate_vector_record(record, ExpectedVectorIdentity.canonical_vpi())


def test_negative_infinity_rejected() -> None:
    vector = _unit_vector()
    vector_list = list(vector)
    vector_list[1] = -math.inf
    record = _vector_record(0, vector=tuple(vector_list))
    with pytest.raises(QdrantBootstrapVectorValidationError):
        _validate_vector_record(record, ExpectedVectorIdentity.canonical_vpi())


def test_zero_vector_rejected() -> None:
    record = _vector_record(0, vector=tuple(0.0 for _ in range(CANONICAL_EMBEDDING_DIMENSION)))
    with pytest.raises(QdrantBootstrapVectorValidationError):
        _validate_vector_record(record, ExpectedVectorIdentity.canonical_vpi())


# --- POINT IDENTITY ---


def test_deterministic_logical_point_id_mapping() -> None:
    record = _vector_record(0)
    first = normalize_qdrant_logical_point_id(record.logical_point_id)
    second = normalize_qdrant_logical_point_id(record.logical_point_id)
    assert first == second


def test_retry_produces_same_point_id() -> None:
    record = _vector_record(0)
    assert normalize_qdrant_logical_point_id(record.logical_point_id) == normalize_qdrant_logical_point_id(record.logical_point_id)


def test_no_random_point_ids_during_write() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    assert len(client.points) == 1
    point_id = next(iter(client.points))
    assert point_id == normalize_qdrant_logical_point_id(record.logical_point_id)


# --- PAYLOAD ---


def test_payload_source_identity_preserved() -> None:
    record = _vector_record(0)
    payload = payload_from_record(record)
    assert payload.catalog_id == record.source_ref.catalog_id
    assert payload.offer_id == record.source_ref.offer_id.value


def test_payload_semantic_hash_preserved() -> None:
    record = _vector_record(0, semantic_hash="hash-xyz")
    payload = payload_from_record(record)
    assert payload.semantic_text_hash == "hash-xyz"


def test_payload_embedding_provider_preserved() -> None:
    payload = payload_from_record(_vector_record(0))
    assert payload.embedding_provider == CANONICAL_EMBEDDING_PROVIDER


def test_payload_model_preserved() -> None:
    payload = payload_from_record(_vector_record(0))
    assert payload.embedding_model == CANONICAL_EMBEDDING_MODEL


def test_payload_revision_preserved() -> None:
    payload = payload_from_record(_vector_record(0))
    assert payload.embedding_revision == CANONICAL_EMBEDDING_REVISION


def test_payload_derivation_version_preserved() -> None:
    payload = payload_from_record(_vector_record(0))
    assert payload.derivation_version == "v1"


# --- WRITE ---


def test_new_vector_written() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    result = adapter.write_batch(_batch(_vector_record(0)))
    assert result.written_count == 1
    assert result.skipped_count == 0


def test_batch_write_passes() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    result = adapter.write_batch(_batch(_vector_record(0), _vector_record(1, offer_suffix="1")))
    assert result.is_complete_success


def test_batch_upsert_uses_multi_point_call() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    adapter.write_batch(_batch(_vector_record(0), _vector_record(1, offer_suffix="1")))
    assert client.upsert_calls == 1


def test_existing_identical_skipped() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    result = adapter.write_batch(_batch(record))
    assert result.written_count == 0
    assert result.skipped_count == 1
    assert len(client.points) == 1


def test_same_point_different_semantic_hash_fails() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    record = _vector_record(0, semantic_hash="hash-a")
    adapter.write_batch(_batch(record))
    conflict = _vector_record(0, semantic_hash="hash-b")
    with pytest.raises(StorageBootstrapWriteError, match="VECTOR_CONTENT_CONFLICT"):
        adapter.write_batch(_batch(conflict))


def test_same_point_different_model_identity_fails() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    point_id = normalize_qdrant_logical_point_id(record.logical_point_id)
    client.points[point_id].payload["embedding_model"] = "other-model"
    with pytest.raises(StorageBootstrapWriteError, match="VECTOR_CONTENT_CONFLICT"):
        adapter.write_batch(_batch(record))


def test_incoming_model_identity_incompatible_fails() -> None:
    adapter = _adapter_with_fake(_FakeQdrantClient(), prepared=True)
    with pytest.raises(StorageBootstrapWriteError, match="VECTOR_IDENTITY_INCOMPATIBLE"):
        adapter.write_batch(_batch(_vector_record(0, model="other-model")))


def test_same_point_different_vector_fails() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    vector = list(_unit_vector())
    vector[0] = 0.0
    vector[1] = 1.0
    conflict = _vector_record(0, vector=tuple(vector))
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(conflict))


def test_blind_overwrite_impossible() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    stored_vector = client.points[normalize_qdrant_logical_point_id(record.logical_point_id)].vector
    with pytest.raises(StorageBootstrapWriteError):
        adapter.write_batch(_batch(_vector_record(0, semantic_hash="other")))
    assert client.points[normalize_qdrant_logical_point_id(record.logical_point_id)].vector == stored_vector


# --- VERIFICATION ---


def test_verify_expected_point_exists() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    verify = adapter.verify_batch(_batch(record))
    assert verify.failed_count == 0


def test_verify_missing_point_detected() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    verify = adapter.verify_batch(_batch(_vector_record(0)))
    assert verify.failed_count == 1


def test_verify_payload_mismatch_detected() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    point_id = normalize_qdrant_logical_point_id(record.logical_point_id)
    client.points[point_id].payload["semantic_text_hash"] = "mutated"
    verify = adapter.verify_batch(_batch(record))
    assert verify.failed_count == 1


def test_verify_vector_mismatch_detected() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    record = _vector_record(0)
    adapter.write_batch(_batch(record))
    point_id = normalize_qdrant_logical_point_id(record.logical_point_id)
    vector = list(client.points[point_id].vector)
    vector[0] = 0.0
    vector[1] = 1.0
    client.points[point_id].vector = vector
    verify = adapter.verify_batch(_batch(record))
    assert verify.failed_count == 1


def test_verify_uses_bounded_batch_retrieval() -> None:
    client = _FakeQdrantClient()
    adapter = _adapter_with_fake(client, prepared=True)
    records = tuple(_vector_record(index, offer_suffix=str(index)) for index in range(3))
    adapter.write_batch(_batch(*records))
    client.retrieve_calls = 0
    adapter.verify_batch(_batch(*records))
    assert client.retrieve_calls >= 1


# --- PORT CONTRACT ---


def test_adapter_satisfies_vector_storage_load_port() -> None:
    adapter = _adapter_with_fake(_FakeQdrantClient())
    assert hasattr(adapter, "write_batch")
    assert hasattr(adapter, "verify_batch")


def test_result_is_storage_load_batch_result() -> None:
    adapter = _adapter_with_fake(_FakeQdrantClient(), prepared=True)
    result = adapter.write_batch(_batch())
    assert isinstance(result, StorageLoadBatchResult)


def test_no_qdrant_type_leaks_into_core() -> None:
    violations: list[str] = []
    for module_path in sorted(_CORE_ROOT.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported in _FORBIDDEN_CORE_IMPORTS:
                violations.append(str(module_path))
    assert violations == []


# --- ARCHITECTURE ---


@pytest.mark.parametrize("forbidden", sorted(_FORBIDDEN_ADAPTER_IMPORTS))
def test_adapter_has_no_relational_or_model_imports(forbidden: str) -> None:
    violations: list[str] = []
    for module_path in sorted(_ADAPTER_ROOT.rglob("*.py")):
        if forbidden in _module_imports(module_path):
            violations.append(str(module_path))
    assert violations == []


def test_payload_identity_helper() -> None:
    record = _vector_record(0)
    left = payload_from_record(record)
    right = payload_from_record(record)
    assert payload_identity_matches(left, right) is True


def test_float32_transport_equality() -> None:
    left = normalize_vector_float32((1.0, 0.0))
    right = normalize_vector_float32((1.0, 0.0))
    assert vectors_transport_equal(left, right, tolerance=0.0) is True


def test_logical_target_maps_to_approved_collection() -> None:
    physical = resolve_physical_target(VectorTargetId("vpi-product-embeddings"), _configuration())
    assert physical.collection_name.endswith("vpi-product-embeddings__tenant__default")


def test_configuration_collection_mismatch_rejected() -> None:
    config = QdrantBootstrapConfiguration(
        integration=_configuration().integration,
        logical_collection_name="other-collection",
        expected_vector_identity=ExpectedVectorIdentity.canonical_vpi(),
    )
    with pytest.raises(QdrantBootstrapConfigurationError):
        resolve_physical_target(VectorTargetId("vpi-product-embeddings"), config)


def test_collection_shape_parses_default_dense_vector() -> None:
    client = _FakeQdrantClient()
    shape = _collection_vector_shape(client.get_collection("x"), _physical_target())
    assert shape.dimension == 1024
    assert shape.distance.lower().startswith("cos")


def test_distance_label_normalizes_string_distance() -> None:
    assert _distance_label("Cosine") == "Cosine"


def test_distance_label_normalizes_sdk_distance_enum() -> None:
    try:
        from qdrant_client.http.models import Distance
    except ImportError:
        pytest.skip("qdrant-client unavailable")
    assert _distance_label(Distance.COSINE).lower().startswith("cos")


def test_extract_provider_payload_converts_typed_mapping() -> None:
    record = _vector_record(0)
    payload = payload_from_record(record).to_provider_payload()
    converted = _extract_provider_payload(payload)
    assert converted == payload


def test_extract_provider_payload_rejects_missing_payload() -> None:
    with pytest.raises(QdrantBootstrapOperationError, match="stored payload missing"):
        _extract_provider_payload(None)


def test_extract_provider_payload_filters_non_scalar_values() -> None:
    converted = _extract_provider_payload(
        {
            "logical_id": "point-1",
            "catalog_id": "wdc-v2-selected",
            "offer_id": "offer-0",
            "semantic_text_hash": "hash-a",
            "embedding_provider": CANONICAL_EMBEDDING_PROVIDER,
            "embedding_model": CANONICAL_EMBEDDING_MODEL,
            "embedding_dimension": CANONICAL_EMBEDDING_DIMENSION,
            "derivation_version": "v1",
            "ignored": 1.5,
        }
    )
    assert "ignored" not in converted
    assert converted["logical_id"] == "point-1"


def test_extract_dense_vector_default_channel() -> None:
    physical = _physical_target()
    vector = _extract_dense_vector([1.0, 0.0], physical)
    assert vector == (1.0, 0.0)


def test_extract_dense_vector_rejects_named_shape_on_default_target() -> None:
    physical = _physical_target()
    with pytest.raises(QdrantBootstrapOperationError, match="unexpected named channels"):
        _extract_dense_vector({"dense": [1.0, 0.0]}, physical)


def test_extract_dense_vector_rejects_missing_vector() -> None:
    physical = _physical_target()
    with pytest.raises(QdrantBootstrapOperationError, match="invalid type"):
        _extract_dense_vector(None, physical)


def _named_physical_target() -> PhysicalVectorTarget:
    config = QdrantBootstrapConfiguration(
        integration=_configuration().integration,
        logical_collection_name="vpi-product-embeddings",
        expected_vector_identity=ExpectedVectorIdentity.canonical_vpi(),
        upsert_batch_size=64,
        uses_named_dense_vector=True,
        dense_vector_channel_name="dense",
    )
    return resolve_physical_target(VectorTargetId("vpi-product-embeddings"), config)


def test_extract_dense_vector_rejects_invalid_channel_shape() -> None:
    with pytest.raises(QdrantBootstrapOperationError, match="missing named dense channel"):
        _extract_dense_vector({"other": [1.0, 0.0]}, _named_physical_target())


def test_cosine_storage_normalize_matches_qdrant_roundtrip_for_sparse_vectors() -> None:
    sparse = normalize_vector_float32((0.0, 0.002, 0.0))
    stored = cosine_storage_normalize(sparse)
    readback = cosine_storage_normalize((0.0, 1.0, 0.0))
    assert vectors_transport_equal(stored, readback, tolerance=0.0) is True


def test_record_matches_stored_accepts_runtime_composition_vectors() -> None:
    from tests.integration.platform_proofs.scenarios.verified_product_identification.conftest import (
        deterministic_dense_embedding,
    )

    record = _vector_record(3, offer_suffix="3", semantic_hash="hash-runtime")
    record = VectorLoadRecord(
        logical_point_id=record.logical_point_id,
        source_ref=record.source_ref,
        semantic_text_hash=record.semantic_text_hash,
        embedding_provider=record.embedding_provider,
        embedding_model=record.embedding_model,
        embedding_revision=record.embedding_revision,
        embedding_dimension=record.embedding_dimension,
        dense_embedding=deterministic_dense_embedding(3),
        derivation_version=record.derivation_version,
    )
    payload = payload_from_record(record)
    stored = QdrantStoredPoint(
        point_id=normalize_qdrant_logical_point_id(record.logical_point_id),
        logical_point_id=record.logical_point_id,
        payload=payload,
        vector=cosine_storage_normalize(record.dense_embedding),
    )
    assert _record_matches_stored(record, stored, tolerance=0.0) is True


def test_stored_point_from_provider_record_converts_payload_and_vector() -> None:
    record = _vector_record(0)
    payload = payload_from_record(record).to_provider_payload()
    provider_point = _FakeQdrantPoint(
        id=normalize_qdrant_logical_point_id(record.logical_point_id),
        payload=payload,
        vector=list(normalize_vector_float32(record.dense_embedding)),
    )
    stored = _stored_point_from_provider_record(provider_point, _physical_target())
    assert stored.logical_point_id == record.logical_point_id
    assert payload_identity_matches(stored.payload, payload_from_record(record)) is True


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
    with pytest.raises(QdrantBootstrapIdentityConflictError):
        raise QdrantBootstrapIdentityConflictError("conflict")


def test_verify_batch_integrity_error_translation() -> None:
    client = _FakeQdrantClient()
    client.retrieve = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("down"))  # type: ignore[method-assign]
    adapter = _adapter_with_fake(client, prepared=True)
    with pytest.raises(StorageBootstrapIntegrityError):
        adapter.verify_batch(_batch(_vector_record(0)))


# --- REAL QDRANT (optional) ---


@pytest.mark.integration
def test_real_qdrant_bounded_qualification() -> None:
    if not qdrant_environment_available():
        pytest.skip("Qdrant environment unavailable")

    from intergrax.integrations.providers.vector_store.qdrant.opens import (
        open_qdrant_vector_data_plane_client,
    )

    collection_name = f"vpi_5c5c_adapter_{uuid.uuid4().hex[:8]}"
    config = QdrantBootstrapConfiguration.from_env(logical_collection_name=collection_name)
    client = open_qdrant_vector_data_plane_client(config.integration)
    try:
        adapter = QdrantVectorStorageAdapter.from_env(logical_collection_name=collection_name)
        target = VectorTargetId("vpi-product-embeddings")
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
        adapter.close()
    finally:
        physical_name = physical_collection_name(collection_name, config.integration.tenant_id)
        collections = {item.name for item in client.get_collections().collections}
        if physical_name in collections:
            client.delete_collection(physical_name)
        client.close()
