"""Unit tests for Qdrant bootstrap adapter compatibility enforcement."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.vector_index_administration import (
    VectorIndexCompatibilityError,
    VectorIndexDescription,
    VectorIndexIdentity,
    VectorIndexPrepareOutcome,
    VectorIndexPrepareResult,
    VectorIndexSpec,
    VectorSearchCapability,
    validate_spec_against_description,
)
from intergrax.integrations.contracts.vector_store import VectorStoreScope
from intergrax.integrations.providers.vector_store.qdrant.index_administration import (
    build_qdrant_index_spec,
)

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


def _identity() -> VectorIndexIdentity:
    return VectorIndexIdentity(logical_name="vpi_test", tenant_id="default")


def _description(
    *,
    exists: bool,
    dense_dimension: int | None,
    sparse_present: bool,
    point_count: int = 0,
) -> VectorIndexDescription:
    capabilities: set[VectorSearchCapability] = set()
    if dense_dimension is not None:
        capabilities.add(VectorSearchCapability.DENSE)
    if sparse_present:
        capabilities.add(VectorSearchCapability.SPARSE_LEXICAL)
    return VectorIndexDescription(
        identity=_identity(),
        exists=exists,
        reachable=True,
        point_count=point_count,
        dense_dimension=dense_dimension,
        present_capabilities=frozenset(capabilities),
        dense_channel_name="dense" if dense_dimension is not None else None,
        sparse_lexical_channel_name="sparse" if sparse_present else None,
    )


@dataclass(slots=True)
class _FakeIndexAdmin:
    description: VectorIndexDescription
    prepare_error: VectorIndexCompatibilityError | None = None
    prepare_outcome: VectorIndexPrepareOutcome = VectorIndexPrepareOutcome.ALREADY_COMPATIBLE

    def probe(self) -> HealthStatus:
        return HealthStatus(slug="qdrant", healthy=True)

    def describe_index(self, identity: VectorIndexIdentity) -> VectorIndexDescription:
        return self.description

    def prepare_index(self, spec: VectorIndexSpec) -> VectorIndexPrepareResult:
        if self.prepare_error is not None:
            raise self.prepare_error
        description = self.description
        validate_spec_against_description(spec, description)
        return VectorIndexPrepareResult(outcome=self.prepare_outcome, description=description)

    def close(self) -> None:
        return None


@dataclass(slots=True)
class _FakeVectorStore:
    point_count: int = 0

    def add_records(self, records, *, scope: VectorStoreScope):
        self.point_count += len(records)
        return [record.vector_id for record in records]

    def query(self, *args, **kwargs):
        raise NotImplementedError

    def delete(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def count(self, *, scope: VectorStoreScope) -> int:
        return self.point_count


def _adapter(
    *,
    description: VectorIndexDescription,
    prepare_error: VectorIndexCompatibilityError | None = None,
) -> QdrantSearchIndexBootstrapAdapter:
    return QdrantSearchIndexBootstrapAdapter(
        _index_admin=_FakeIndexAdmin(description=description, prepare_error=prepare_error),
        _vector_store=_FakeVectorStore(point_count=description.point_count),
        _index_identity=_identity(),
        _sparse_required=True,
    )


def test_qdrant_dimension_mismatch_rejected() -> None:
    adapter = _adapter(
        description=_description(exists=True, dense_dimension=512, sparse_present=True),
        prepare_error=VectorIndexCompatibilityError("vector index dense dimension 512 != expected 8"),
    )
    with pytest.raises(VpiBootstrapCompatibilityError, match="dimension"):
        adapter.prepare(_manifest(dimension=8))


def test_qdrant_dense_only_collection_rejected_when_sparse_required() -> None:
    adapter = _adapter(
        description=_description(exists=True, dense_dimension=8, sparse_present=False),
        prepare_error=VectorIndexCompatibilityError(
            "vector index missing required capabilities: sparse_lexical"
        ),
    )
    with pytest.raises(VpiBootstrapCompatibilityError, match="sparse_lexical"):
        adapter.prepare(_manifest(dimension=8))


def test_validate_passes_after_restart_without_prepare() -> None:
    description = _description(
        exists=True,
        dense_dimension=1024,
        sparse_present=True,
        point_count=10,
    )
    process_a = _adapter(description=description)
    process_a.prepare(_manifest(dimension=1024, target_max_records=10))

    process_b = QdrantSearchIndexBootstrapAdapter(
        _index_admin=_FakeIndexAdmin(description=description),
        _vector_store=_FakeVectorStore(point_count=10),
        _index_identity=_identity(),
        _sparse_required=True,
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
        description=_description(
            exists=True,
            dense_dimension=2560,
            sparse_present=True,
            point_count=10,
        ),
    )
    report = adapter.validate(_manifest(dimension=1024, target_max_records=10))
    assert report.status is ValidationStatus.FAIL
    dimension_check = next(check for check in report.checks if check.name == "qdrant_dimension")
    assert dimension_check.status is ValidationStatus.FAIL
    assert "2560" in dimension_check.detail


def test_validate_fails_when_sparse_channel_missing() -> None:
    adapter = _adapter(
        description=_description(
            exists=True,
            dense_dimension=1024,
            sparse_present=False,
            point_count=10,
        ),
    )
    report = adapter.validate(_manifest(dimension=1024, target_max_records=10))
    assert report.status is ValidationStatus.FAIL
    sparse_check = next(check for check in report.checks if check.name == "qdrant_sparse_channel")
    assert sparse_check.status is ValidationStatus.FAIL


def test_validate_fails_when_collection_absent() -> None:
    adapter = _adapter(description=_description(exists=False, dense_dimension=None, sparse_present=False))
    report = adapter.validate(_manifest(dimension=1024, target_max_records=10))
    assert report.status is ValidationStatus.FAIL
    assert any(check.name == "qdrant_collection_exists" for check in report.checks)


def test_validate_fails_when_point_count_below_target() -> None:
    adapter = _adapter(
        description=_description(
            exists=True,
            dense_dimension=1024,
            sparse_present=True,
            point_count=3,
        ),
    )
    report = adapter.validate(_manifest(dimension=1024, target_max_records=10))
    assert report.status is ValidationStatus.FAIL
    point_check = next(check for check in report.checks if check.name == "qdrant_point_count")
    assert point_check.status is ValidationStatus.FAIL


def test_build_qdrant_index_spec_requires_sparse_spec_when_capability_requested() -> None:
    with pytest.raises(ValueError, match="sparse_lexical"):
        VectorIndexSpec(
            identity=_identity(),
            dense=build_qdrant_index_spec(
                identity=_identity(),
                dimension=8,
                enable_sparse_lexical=False,
            ).dense,
            required_capabilities=frozenset({VectorSearchCapability.SPARSE_LEXICAL}),
            sparse_lexical=None,
        )
