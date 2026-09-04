"""Alignment, resume, and artifact-boundary tests for storage bootstrap."""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("pyarrow")

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactState,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    DatasetVerificationMode,
    VpiBootstrapConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    CATALOG_SCHEMA_VERSION,
    SEARCH_INDEX_SCHEMA_VERSION,
    BootstrapState,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.orchestration.aligned_input import (
    AlignedBootstrapInputIterator,
    validate_source_artifact_alignment,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.orchestration.orchestrator import (
    VpiBootstrapDependencies,
    VpiBootstrapOrchestrator,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapDataError,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.bootstrap_artifact_test_support import (
    artifact_record_for_wdc_row,
    reader_from_records,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_bootstrap_orchestrator import (
    FakeCatalogPort,
    FakeSearchPort,
    _write_tiny_parquet,
)
from intergrax.rag.embedding.registry.profile import EmbeddingProfile

pytestmark = pytest.mark.unit


def _bootstrap_config(tmp_path: Path, *, max_records: int, row_count: int) -> VpiBootstrapConfig:
    dataset_path = tmp_path / "selected_offers.parquet"
    _write_tiny_parquet(dataset_path, row_count=row_count)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        f'{{"output_sha256":"testchecksum","selected_record_count":{row_count}}}',
        encoding="utf-8",
    )
    return VpiBootstrapConfig(
        dataset_path=dataset_path,
        dataset_manifest_path=manifest_path,
        dataset_verification_mode=DatasetVerificationMode.FAST,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        max_records=max_records,
        source_batch_size=32,
        vector_batch_size=32,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        search_index_schema_version=SEARCH_INDEX_SCHEMA_VERSION,
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        postgresql_schema="vpi",
        qdrant_collection_name="vpi_offers",
        artifact_root_dir=tmp_path / "artifacts",
        embedding_configuration=VpiEmbeddingConfiguration(
            profile=EmbeddingProfile(provider="hf", model="fake-model"),
            expected_dimension=8,
        ),
    )


def test_alignment_mismatch_fails_closed(tmp_path: Path) -> None:
    records = (
        artifact_record_for_wdc_row(0),
        artifact_record_for_wdc_row(1, offer_id="wrong-offer"),
    )
    reader = reader_from_records(records, dataset_record_count=5)
    aligned = AlignedBootstrapInputIterator(
        dataset_path=_bootstrap_config(tmp_path, max_records=2, row_count=5).dataset_path,
        artifact_reader=reader,
        artifact_manifest=reader.manifest,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        source_batch_size=2,
        start_row_index=0,
        start_batch_ordinal=0,
        max_records=2,
    )
    with pytest.raises(VpiBootstrapDataError, match="offer_id mismatch"):
        list(aligned)


def test_derivation_mismatch_fails_closed(tmp_path: Path) -> None:
    records = (
        artifact_record_for_wdc_row(0, derivation_version="v0"),
        artifact_record_for_wdc_row(1, derivation_version="v0"),
    )
    reader = reader_from_records(records, dataset_record_count=5)
    aligned = AlignedBootstrapInputIterator(
        dataset_path=_bootstrap_config(tmp_path, max_records=2, row_count=5).dataset_path,
        artifact_reader=reader,
        artifact_manifest=reader.manifest,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        source_batch_size=2,
        start_row_index=0,
        start_batch_ordinal=0,
        max_records=2,
    )
    with pytest.raises(VpiBootstrapDataError, match="derivation_version mismatch"):
        list(aligned)


def test_shard_boundary_alignment_with_source_batch_size(tmp_path: Path) -> None:
    row_count = 100
    dataset_path = tmp_path / "rows.parquet"
    _write_tiny_parquet(dataset_path, row_count=row_count)
    records = tuple(artifact_record_for_wdc_row(index) for index in range(row_count))
    reader = reader_from_records(records, dataset_record_count=row_count, shard_size=50)
    aligned = AlignedBootstrapInputIterator(
        dataset_path=dataset_path,
        artifact_reader=reader,
        artifact_manifest=reader.manifest,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        source_batch_size=32,
        start_row_index=0,
        start_batch_ordinal=0,
        max_records=row_count,
    )
    total_rows = 0
    for _, batch in aligned:
        total_rows += len(batch)
        for record in batch:
            validate_source_artifact_alignment(record.catalog_record, record.artifact_record)
    assert total_rows == row_count


def test_storage_resume_loads_delta_only(tmp_path: Path) -> None:
    row_count = 100
    config = _bootstrap_config(tmp_path, max_records=100, row_count=row_count)
    records = tuple(artifact_record_for_wdc_row(index) for index in range(row_count))
    reader = reader_from_records(records, dataset_record_count=row_count)
    catalog = FakeCatalogPort()
    search = FakeSearchPort(point_count=40)
    catalog.manifest = catalog.manifest or None
    from tests.unit.platform_proofs.scenarios.verified_product_identification.test_bootstrap_orchestrator import (
        _sample_manifest,
    )

    catalog.manifest = _sample_manifest(
        state=BootstrapState.INGESTING,
        dataset_record_count=row_count,
        target_max_records=100,
        checkpoint_batch_ordinal=1,
        checkpoint_rows_processed=40,
        catalog_source_offer_count=40,
        catalog_identifier_count=40,
        catalog_structured_attribute_count=40,
        search_point_count=40,
        dataset_checksum="testchecksum",
    )
    catalog.source_offer_count = 40
    catalog.identifier_count = 40
    catalog.structured_attribute_count = 40
    catalog._ingested_offer_ids = {str(1000 + index) for index in range(40)}
    orchestrator = VpiBootstrapOrchestrator(
        config=config,
        dependencies=VpiBootstrapDependencies(
            catalog=catalog,
            search=search,
            embedding_artifact=reader,
        ),
    )
    report = orchestrator.run()
    assert report.final_state is BootstrapState.READY
    assert catalog.manifest is not None
    assert catalog.manifest.checkpoint_rows_processed == 100
    assert len(catalog.batches) == 2


def test_target_extension_resumes_from_storage_checkpoint(tmp_path: Path) -> None:
    row_count = 200
    config = _bootstrap_config(tmp_path, max_records=200, row_count=row_count)
    records = tuple(artifact_record_for_wdc_row(index) for index in range(row_count))
    reader = reader_from_records(records, dataset_record_count=row_count)
    catalog = FakeCatalogPort()
    search = FakeSearchPort(point_count=100)
    from tests.unit.platform_proofs.scenarios.verified_product_identification.test_bootstrap_orchestrator import (
        _sample_manifest,
    )

    catalog.manifest = _sample_manifest(
        state=BootstrapState.READY,
        dataset_record_count=row_count,
        target_max_records=100,
        checkpoint_batch_ordinal=3,
        checkpoint_rows_processed=100,
        catalog_source_offer_count=100,
        catalog_identifier_count=100,
        catalog_structured_attribute_count=100,
        search_point_count=100,
        dataset_checksum="testchecksum",
    )
    catalog.source_offer_count = 100
    catalog.identifier_count = 100
    catalog.structured_attribute_count = 100
    catalog._ingested_offer_ids = {str(1000 + index) for index in range(100)}
    orchestrator = VpiBootstrapOrchestrator(
        config=config,
        dependencies=VpiBootstrapDependencies(
            catalog=catalog,
            search=search,
            embedding_artifact=reader,
        ),
    )
    report = orchestrator.run()
    assert report.final_state is BootstrapState.READY
    assert catalog.manifest is not None
    assert catalog.manifest.checkpoint_rows_processed == 200


def test_identity_mismatch_fails_before_mutation(tmp_path: Path) -> None:
    records = tuple(artifact_record_for_wdc_row(index) for index in range(5))
    reader = reader_from_records(
        records,
        dataset_record_count=5,
        embedding_model="other-model",
    )
    orchestrator = VpiBootstrapOrchestrator(
        config=_bootstrap_config(tmp_path, max_records=2, row_count=5),
        dependencies=VpiBootstrapDependencies(
            catalog=FakeCatalogPort(),
            search=FakeSearchPort(),
            embedding_artifact=reader,
        ),
    )
    report = orchestrator.run()
    assert report.final_state is BootstrapState.FAILED
    assert report.failure_stage == "VpiBootstrapCompatibilityError"


def test_ready_artifact_100_rows_bootstrap_100(tmp_path: Path) -> None:
    row_count = 100
    records = tuple(artifact_record_for_wdc_row(index) for index in range(row_count))
    reader = reader_from_records(records, dataset_record_count=row_count)
    catalog = FakeCatalogPort()
    search = FakeSearchPort()
    config = _bootstrap_config(tmp_path, max_records=row_count, row_count=row_count)
    config = replace(config, source_batch_size=25)
    orchestrator = VpiBootstrapOrchestrator(
        config=config,
        dependencies=VpiBootstrapDependencies(
            catalog=catalog,
            search=search,
            embedding_artifact=reader,
        ),
    )
    report = orchestrator.run()
    assert report.final_state is BootstrapState.READY
    assert catalog.manifest is not None
    assert catalog.manifest.checkpoint_rows_processed == row_count
    assert catalog.manifest.search_point_count == row_count
