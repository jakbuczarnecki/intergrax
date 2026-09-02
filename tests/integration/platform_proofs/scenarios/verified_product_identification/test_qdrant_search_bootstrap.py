"""Qdrant search bootstrap integration — skipped when provider or Gate 0 unavailable."""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.opens import (
    open_qdrant_vector_index_administration,
    open_qdrant_vector_store,
)

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.platform_bootstrap_adapter import (
    PlatformSearchIndexBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    SearchIndexIngestBatch,
    SearchIndexIngestRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    BootstrapState,
    VpiBootstrapManifest,
)

pytestmark = [pytest.mark.integration]


def _qdrant_available() -> bool:
    return bool(os.getenv("INTERGRAX_QDRANT_URL", "").strip() or os.getenv("INTERGRAX_QDRANT_HOST", "").strip())


def _open_adapter(collection_name: str) -> PlatformSearchIndexBootstrapAdapter:
    config = QdrantIntegrationConfig.from_env(
        collection_name=collection_name,
        enable_sparse_vectors=True,
    )
    return PlatformSearchIndexBootstrapAdapter(
        _index_admin=open_qdrant_vector_index_administration(config),
        _vector_store=open_qdrant_vector_store(config),
        _index_identity=VectorIndexIdentity(
            logical_name=collection_name,
            tenant_id=config.tenant_id,
        ),
        _dense_channel_name="dense",
        _sparse_channel_name="sparse",
        _sparse_required=config.enable_sparse_vectors,
    )


@pytest.mark.skipif(not _qdrant_available(), reason="Qdrant not configured locally")
def test_qdrant_search_bootstrap_idempotent() -> None:
    collection_name = f"vpi_test_{uuid.uuid4().hex[:8]}"
    adapter = _open_adapter(collection_name)
    source_ref = SourceRecordRef(
        offer_id=ProductOfferId("offer-1"),
        catalog_id="wdc-v2-selected",
        source_revision=None,
    )
    manifest = VpiBootstrapManifest(
        state=BootstrapState.INITIALIZING,
        dataset_path="/data/selected_offers.parquet",
        dataset_checksum="integration-checksum",
        dataset_record_count=1,
        search_representation_derivation_version="v2",
        embedding_configuration_version="v1",
        embedding_provider="fake",
        embedding_model="fake-model",
        embedding_dimension=8,
        catalog_schema_version="v1",
        search_index_schema_version="v1",
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        checkpoint_batch_ordinal=None,
        checkpoint_rows_processed=0,
        target_max_records=1,
        catalog_source_offer_count=0,
        catalog_identifier_count=0,
        catalog_structured_attribute_count=0,
        search_point_count=0,
    )

    adapter.prepare(manifest)
    adapter.prepare(manifest)

    vector = tuple(float(value) for value in np.full(8, 1.0, dtype=np.float32))
    record = SearchIndexIngestRecord(
        logical_point_id=search_representation_point_id(
            catalog_id=source_ref.catalog_id,
            offer_id=source_ref.offer_id.value,
            derivation_version="v2",
        ),
        dense_embedding=vector,
        lexical_text="relay module 24V",
        source_ref=source_ref,
        derivation_version="v2",
        dataset_checksum="integration-checksum",
        embedding_provider="fake",
        embedding_model="fake-model",
        embedding_dimension=8,
    )
    batch = SearchIndexIngestBatch(batch_ordinal=0, records=(record,))
    adapter.ingest_batch(batch)
    adapter.ingest_batch(batch)

    assert adapter.count_points() == 1
    adapter.close()
