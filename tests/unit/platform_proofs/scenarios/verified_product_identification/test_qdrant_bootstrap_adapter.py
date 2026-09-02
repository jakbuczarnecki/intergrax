"""Unit tests for Qdrant bootstrap adapter compatibility enforcement."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from platform_proofs.scenarios.verified_product_identification.integrations.search_store.qdrant.adapter import (
    QdrantSearchIndexBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapCompatibilityError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    BootstrapState,
    VpiBootstrapManifest,
)

pytestmark = pytest.mark.unit


def _manifest(*, dimension: int = 8) -> VpiBootstrapManifest:
    return VpiBootstrapManifest(
        state=BootstrapState.INITIALIZING,
        dataset_path="/data/selected_offers.parquet",
        dataset_checksum="abc123",
        dataset_record_count=10,
        search_representation_derivation_version="v2",
        embedding_configuration_version="v1",
        embedding_provider="hf",
        embedding_model="fake-model",
        embedding_dimension=dimension,
        catalog_schema_version="v1",
        search_index_schema_version="v1",
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        checkpoint_batch_ordinal=None,
        checkpoint_rows_processed=0,
        target_max_records=10,
        catalog_source_offer_count=0,
        catalog_identifier_count=0,
        catalog_structured_attribute_count=0,
        search_point_count=0,
    )


class _FakeQdrantClient:
    def __init__(self, collection_info: SimpleNamespace | None) -> None:
        self._collection_info = collection_info

    def get_collections(self) -> object:
        return SimpleNamespace(collections=[])

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        if self._collection_info is None:
            raise RuntimeError("404 collection not found")
        return self._collection_info

    def create_collection(self, *args, **kwargs) -> bool:
        return True

    def upsert(self, *args, **kwargs) -> object:
        return None

    def close(self) -> None:
        return None


def _collection_info(*, dense_size: int, sparse_names: tuple[str, ...]) -> SimpleNamespace:
    dense = SimpleNamespace(size=dense_size)
    sparse_vectors = {name: SimpleNamespace() for name in sparse_names}
    return SimpleNamespace(
        points_count=0,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": dense},
                sparse_vectors=sparse_vectors,
            )
        ),
    )


def test_qdrant_dimension_mismatch_rejected() -> None:
    adapter = QdrantSearchIndexBootstrapAdapter(
        _client=_FakeQdrantClient(_collection_info(dense_size=512, sparse_names=("sparse",))),
        _collection_name="vpi_test",
        _tenant_id="default",
        _sparse_enabled=True,
        _sparse_encoder=SimpleNamespace(encode=lambda text: SimpleNamespace(indices=[1], values=[1.0])),
    )
    with pytest.raises(VpiBootstrapCompatibilityError, match="dimension"):
        adapter.prepare(_manifest(dimension=8))


def test_qdrant_dense_only_collection_rejected_when_sparse_required() -> None:
    adapter = QdrantSearchIndexBootstrapAdapter(
        _client=_FakeQdrantClient(_collection_info(dense_size=8, sparse_names=())),
        _collection_name="vpi_test",
        _tenant_id="default",
        _sparse_enabled=True,
        _sparse_encoder=SimpleNamespace(encode=lambda text: SimpleNamespace(indices=[1], values=[1.0])),
    )
    with pytest.raises(VpiBootstrapCompatibilityError, match="dense-only"):
        adapter.prepare(_manifest(dimension=8))
