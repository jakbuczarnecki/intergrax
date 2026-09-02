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
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    BootstrapState,
    VpiBootstrapManifest,
)

pytestmark = pytest.mark.unit


def _manifest(*, dimension: int = 8, target_max_records: int = 10) -> VpiBootstrapManifest:
    return VpiBootstrapManifest(
        state=BootstrapState.READY,
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
        checkpoint_batch_ordinal=0,
        checkpoint_rows_processed=target_max_records,
        target_max_records=target_max_records,
        catalog_source_offer_count=target_max_records,
        catalog_identifier_count=target_max_records,
        catalog_structured_attribute_count=target_max_records,
        search_point_count=target_max_records,
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


def _collection_info(
    *,
    dense_size: int,
    sparse_names: tuple[str, ...],
    points_count: int = 0,
) -> SimpleNamespace:
    dense = SimpleNamespace(size=dense_size)
    sparse_vectors = {name: SimpleNamespace() for name in sparse_names}
    return SimpleNamespace(
        points_count=points_count,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": dense},
                sparse_vectors=sparse_vectors,
            )
        ),
    )


def _adapter(
    *,
    collection_info: SimpleNamespace | None,
) -> QdrantSearchIndexBootstrapAdapter:
    return QdrantSearchIndexBootstrapAdapter(
        _client=_FakeQdrantClient(collection_info),
        _collection_name="vpi_test",
        _tenant_id="default",
        _sparse_enabled=True,
        _sparse_encoder=SimpleNamespace(encode=lambda text: SimpleNamespace(indices=[1], values=[1.0])),
    )


def test_qdrant_dimension_mismatch_rejected() -> None:
    adapter = _adapter(
        collection_info=_collection_info(dense_size=512, sparse_names=("sparse",)),
    )
    with pytest.raises(VpiBootstrapCompatibilityError, match="dimension"):
        adapter.prepare(_manifest(dimension=8))


def test_qdrant_dense_only_collection_rejected_when_sparse_required() -> None:
    adapter = _adapter(
        collection_info=_collection_info(dense_size=8, sparse_names=()),
    )
    with pytest.raises(VpiBootstrapCompatibilityError, match="dense-only"):
        adapter.prepare(_manifest(dimension=8))


def test_validate_passes_after_restart_without_prepare() -> None:
    shared_client = _FakeQdrantClient(
        _collection_info(dense_size=1024, sparse_names=("sparse",), points_count=10)
    )
    process_a = QdrantSearchIndexBootstrapAdapter(
        _client=shared_client,
        _collection_name="vpi_test",
        _tenant_id="default",
        _sparse_enabled=True,
        _sparse_encoder=SimpleNamespace(encode=lambda text: SimpleNamespace(indices=[1], values=[1.0])),
    )
    process_a.prepare(_manifest(dimension=1024, target_max_records=10))

    process_b = QdrantSearchIndexBootstrapAdapter(
        _client=shared_client,
        _collection_name="vpi_test",
        _tenant_id="default",
        _sparse_enabled=True,
        _sparse_encoder=SimpleNamespace(encode=lambda text: SimpleNamespace(indices=[1], values=[1.0])),
    )
    assert process_b._dimension is None

    report = process_b.validate(_manifest(dimension=1024, target_max_records=10))

    assert report.status is ValidationStatus.PASS
    assert any(check.name == "qdrant_dimension" and check.status is ValidationStatus.PASS for check in report.checks)
    assert any(
        check.name == "qdrant_sparse_channel" and check.status is ValidationStatus.PASS
        for check in report.checks
    )


def test_validate_fails_on_persisted_dimension_mismatch() -> None:
    adapter = _adapter(
        collection_info=_collection_info(dense_size=2560, sparse_names=("sparse",), points_count=10),
    )
    report = adapter.validate(_manifest(dimension=1024, target_max_records=10))
    assert report.status is ValidationStatus.FAIL
    dimension_check = next(check for check in report.checks if check.name == "qdrant_dimension")
    assert dimension_check.status is ValidationStatus.FAIL
    assert "2560" in dimension_check.detail


def test_validate_fails_when_sparse_channel_missing() -> None:
    adapter = _adapter(
        collection_info=_collection_info(dense_size=1024, sparse_names=(), points_count=10),
    )
    report = adapter.validate(_manifest(dimension=1024, target_max_records=10))
    assert report.status is ValidationStatus.FAIL
    sparse_check = next(check for check in report.checks if check.name == "qdrant_sparse_channel")
    assert sparse_check.status is ValidationStatus.FAIL


def test_validate_fails_when_collection_absent() -> None:
    adapter = _adapter(collection_info=None)
    report = adapter.validate(_manifest(dimension=1024, target_max_records=10))
    assert report.status is ValidationStatus.FAIL
    assert any(check.name == "qdrant_collection_exists" for check in report.checks)


def test_validate_fails_when_point_count_below_target() -> None:
    adapter = _adapter(
        collection_info=_collection_info(
            dense_size=1024,
            sparse_names=("sparse",),
            points_count=3,
        ),
    )
    report = adapter.validate(_manifest(dimension=1024, target_max_records=10))
    assert report.status is ValidationStatus.FAIL
    point_check = next(check for check in report.checks if check.name == "qdrant_point_count")
    assert point_check.status is ValidationStatus.FAIL
