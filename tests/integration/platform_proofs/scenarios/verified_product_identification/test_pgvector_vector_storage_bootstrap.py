"""Bounded pgvector qualification for vector Data Pack storage bootstrap."""

from __future__ import annotations

import uuid

import pytest

from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    pgvector_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.adapter import (
    PgVectorStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    VectorBatch,
    VectorLoadRecord,
    VectorTargetId,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapWriteError,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_pgvector_vector_storage_adapter import (
    _batch,
    _vector_record,
)

pytestmark = [pytest.mark.integration]


def _adapter() -> PgVectorStorageAdapter:
    schema_name = f"vpi_pg_bootstrap_{uuid.uuid4().hex[:10]}"
    adapter = PgVectorStorageAdapter.from_env(schema_name=schema_name)
    adapter.prepare_target(VectorTargetId("vpi-product-embeddings"))
    return adapter


@pytest.mark.skipif(not pgvector_environment_available(), reason="pgvector not configured locally")
def test_pgvector_extension_and_target_preparation() -> None:
    adapter = _adapter()
    assert "vpi-product-embeddings" in adapter._prepared_targets


@pytest.mark.skipif(not pgvector_environment_available(), reason="pgvector not configured locally")
def test_pgvector_initial_write_readback_and_idempotency() -> None:
    adapter = _adapter()
    record = _vector_record(0)
    write = adapter.write_batch(_batch(record))
    assert write.written_count == 1
    retry = adapter.write_batch(_batch(record))
    assert retry.skipped_count == 1
    verify = adapter.verify_batch(_batch(record))
    assert verify.failed_count == 0


@pytest.mark.skipif(not pgvector_environment_available(), reason="pgvector not configured locally")
def test_pgvector_metadata_conflict_fails_closed() -> None:
    adapter = _adapter()
    record = _vector_record(0, semantic_hash="hash-a")
    adapter.write_batch(_batch(record))
    with pytest.raises(StorageBootstrapWriteError, match="VECTOR_CONTENT_CONFLICT"):
        adapter.write_batch(_batch(_vector_record(0, semantic_hash="hash-b")))


@pytest.mark.skipif(not pgvector_environment_available(), reason="pgvector not configured locally")
def test_pgvector_vector_conflict_fails_closed() -> None:
    adapter = _adapter()
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
