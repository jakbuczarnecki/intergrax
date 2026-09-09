"""Bounded Qdrant qualification for vector Data Pack storage bootstrap."""

from __future__ import annotations

import importlib.metadata
import os

import pytest

from intergrax.integrations.providers.vector_store.qdrant.opens import _build_qdrant_client
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    qdrant_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.adapter import (
    QdrantVectorStorageAdapter,
    _collection_vector_shape,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    CANONICAL_EMBEDDING_DIMENSION,
    QdrantBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.target_mapping import (
    physical_collection_name,
    resolve_physical_target,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    VectorLoadRecord,
    VectorTargetId,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapWriteError,
)
from tests.integration.platform_proofs.scenarios.verified_product_identification.conftest import (
    runtime_qualification_target_name,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_qdrant_vector_storage_adapter import (
    _batch,
    _vector_record,
)

pytestmark = [pytest.mark.integration]


def _adapter() -> tuple[QdrantVectorStorageAdapter, QdrantBootstrapConfiguration, str]:
    tenant_id = runtime_qualification_target_name()
    previous_tenant = os.environ.get("INTERGRAX_QDRANT_TENANT_ID")
    os.environ["INTERGRAX_QDRANT_TENANT_ID"] = tenant_id
    configuration = QdrantBootstrapConfiguration.from_env(
        logical_collection_name="vpi-product-embeddings",
    )
    adapter = QdrantVectorStorageAdapter.from_env(
        logical_collection_name="vpi-product-embeddings",
    )
    return adapter, configuration, previous_tenant


def _restore_tenant(previous_tenant: str | None) -> None:
    if previous_tenant is None:
        os.environ.pop("INTERGRAX_QDRANT_TENANT_ID", None)
    else:
        os.environ["INTERGRAX_QDRANT_TENANT_ID"] = previous_tenant


def _cleanup(
    adapter: QdrantVectorStorageAdapter,
    configuration: QdrantBootstrapConfiguration,
    previous_tenant: str | None,
) -> None:
    client = _build_qdrant_client(configuration.integration)
    try:
        physical_name = physical_collection_name(
            configuration.logical_collection_name,
            configuration.integration.tenant_id,
        )
        collections = {item.name for item in client.get_collections().collections}
        if physical_name in collections:
            client.delete_collection(physical_name)
    finally:
        client.close()
        adapter.close()
        _restore_tenant(previous_tenant)


@pytest.mark.skipif(not qdrant_environment_available(), reason="Qdrant not configured locally")
def test_qdrant_connectivity_and_collection_configuration() -> None:
    adapter, configuration, previous_tenant = _adapter()
    try:
        target = VectorTargetId("vpi-product-embeddings")
        adapter.prepare_target(target)
        physical = resolve_physical_target(target, configuration)
        shape = _collection_vector_shape(
            adapter._client.get_collection(physical.collection_name),
            physical,
        )
        assert shape.dimension == CANONICAL_EMBEDDING_DIMENSION
        assert shape.distance.lower().startswith("cos")
    finally:
        _cleanup(adapter, configuration, previous_tenant)


@pytest.mark.skipif(not qdrant_environment_available(), reason="Qdrant not configured locally")
def test_qdrant_initial_write_readback_and_idempotency() -> None:
    adapter, configuration, previous_tenant = _adapter()
    try:
        target = VectorTargetId("vpi-product-embeddings")
        adapter.prepare_target(target)
        records = tuple(_vector_record(index) for index in range(5))
        write = adapter.write_batch(_batch(*records))
        assert write.written_count == 5
        assert write.failed_count == 0
        retry = adapter.write_batch(_batch(*records))
        assert retry.skipped_count == 5
        verify = adapter.verify_batch(_batch(*records))
        assert verify.failed_count == 0
    finally:
        _cleanup(adapter, configuration, previous_tenant)


@pytest.mark.skipif(not qdrant_environment_available(), reason="Qdrant not configured locally")
def test_qdrant_payload_conflict_fails_closed() -> None:
    adapter, configuration, previous_tenant = _adapter()
    try:
        target = VectorTargetId("vpi-product-embeddings")
        adapter.prepare_target(target)
        record = _vector_record(0, semantic_hash="hash-a")
        adapter.write_batch(_batch(record))
        with pytest.raises(StorageBootstrapWriteError, match="VECTOR_CONTENT_CONFLICT"):
            adapter.write_batch(_batch(_vector_record(0, semantic_hash="hash-b")))
        verify = adapter.verify_batch(_batch(record))
        assert verify.failed_count == 0
    finally:
        _cleanup(adapter, configuration, previous_tenant)


@pytest.mark.skipif(not qdrant_environment_available(), reason="Qdrant not configured locally")
def test_qdrant_vector_conflict_fails_closed_without_overwrite() -> None:
    adapter, configuration, previous_tenant = _adapter()
    try:
        target = VectorTargetId("vpi-product-embeddings")
        adapter.prepare_target(target)
        record = _vector_record(0)
        adapter.write_batch(_batch(record))
        vector = list(record.dense_embedding)
        vector[0] = 0.5
        conflict = VectorLoadRecord(
            logical_point_id=record.logical_point_id,
            source_ref=record.source_ref,
            semantic_text_hash=record.semantic_text_hash,
            embedding_provider=record.embedding_provider,
            embedding_model=record.embedding_model,
            embedding_revision=record.embedding_revision,
            embedding_dimension=record.embedding_dimension,
            dense_embedding=tuple(vector),
            derivation_version=record.derivation_version,
        )
        with pytest.raises(StorageBootstrapWriteError):
            adapter.write_batch(_batch(conflict))
        verify = adapter.verify_batch(_batch(record))
        assert verify.failed_count == 0
    finally:
        _cleanup(adapter, configuration, previous_tenant)


@pytest.mark.skipif(not qdrant_environment_available(), reason="Qdrant not configured locally")
def test_qdrant_version_evidence_recorded() -> None:
    adapter, configuration, previous_tenant = _adapter()
    try:
        client_version = importlib.metadata.version("qdrant-client")
        assert client_version
        collections = adapter._client.get_collections()
        assert collections is not None
    finally:
        _cleanup(adapter, configuration, previous_tenant)
