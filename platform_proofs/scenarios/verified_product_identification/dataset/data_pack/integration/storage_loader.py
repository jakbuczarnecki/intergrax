"""Load universal data pack into reference storage without re-embedding."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    derive_search_representation,
    flatten_lexical_text,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    source_ref_key,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.ports import (
    DataPackReaderPort,
)
from platform_proofs.scenarios.verified_product_identification.ingest.pipeline.derive_batch import (
    build_catalog_ingest_batch,
)
from platform_proofs.scenarios.verified_product_identification.ingest.source_reader.parquet_dataset import (
    DatasetRow,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    CatalogBootstrapPort,
    CatalogIngestBatch,
    SearchIndexBootstrapPort,
    SearchIndexIngestBatch,
    SearchIndexIngestRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BootstrapState,
    VpiBootstrapManifest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.orchestration.search_from_artifact import (
    search_ingest_record_from_artifact,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)


@dataclass(frozen=True, slots=True)
class DataPackStorageLoadResult:
    catalog_source_rows: int
    search_point_count: int
    embedding_calls: int
    catalog_refs: frozenset[tuple[str, str, str | None]]
    search_refs: frozenset[tuple[str, str, str | None]]


def _catalog_batch_from_data_pack(
    reader: DataPackReaderPort,
    *,
    catalog_id: str,
    source_revision: str | None,
) -> CatalogIngestBatch:
    relational_records = reader.read_relational_records()
    rows = tuple(
        DatasetRow(global_row_index=record.global_row_index, record_json=record.record_json)
        for record in relational_records
    )
    return build_catalog_ingest_batch(
        batch_ordinal=0,
        rows=rows,
        catalog_id=catalog_id,
        source_revision=source_revision,
    )


def _search_batch_from_data_pack(
    reader: DataPackReaderPort,
    *,
    dataset_checksum: str,
) -> SearchIndexIngestBatch:
    relational_records = reader.read_relational_records()
    embedding_records = reader.read_embedding_records()
    embedding_by_ref = {source_ref_key(record.source_ref): record for record in embedding_records}
    search_records: list[SearchIndexIngestRecord] = []
    for relational_record in relational_records:
        embedding_record = embedding_by_ref.get(source_ref_key(relational_record.source_ref))
        if embedding_record is None:
            raise VpiDataPackBuildError(
                f"missing embedding row for {relational_record.source_ref.offer_id.value}"
            )
        source_offer = parse_wdc_source_offer_json(relational_record.record_json)
        representation = derive_search_representation(
            source_offer,
            source_ref=relational_record.source_ref,
            derivation_version=relational_record.derivation_version,
        )
        artifact_record = EmbeddingArtifactRecord(
            global_row_index=relational_record.global_row_index,
            logical_point_id=embedding_record.logical_point_id,
            catalog_id=relational_record.source_ref.catalog_id,
            offer_id=relational_record.source_ref.offer_id.value,
            source_revision=relational_record.source_ref.source_revision,
            derivation_version=relational_record.derivation_version,
            semantic_text=relational_record.semantic_text,
            lexical_text=flatten_lexical_text(representation.lexical),
            embedding_provider=embedding_record.embedding_provider,
            embedding_model=embedding_record.embedding_model,
            embedding_dimension=embedding_record.embedding_dimension,
            dense_embedding=embedding_record.dense_embedding,
        )
        search_records.append(
            search_ingest_record_from_artifact(
                artifact_record,
                dataset_checksum=dataset_checksum,
            )
        )
    return SearchIndexIngestBatch(batch_ordinal=0, records=tuple(search_records))


def load_data_pack_into_reference_storage(
    *,
    reader: DataPackReaderPort,
    catalog: CatalogBootstrapPort,
    search: SearchIndexBootstrapPort,
    manifest: VpiBootstrapManifest,
) -> DataPackStorageLoadResult:
    catalog.prepare(manifest)
    search.prepare(manifest)
    catalog_batch = _catalog_batch_from_data_pack(
        reader,
        catalog_id=manifest.catalog_id,
        source_revision=manifest.source_revision,
    )
    search_batch = _search_batch_from_data_pack(
        reader,
        dataset_checksum=manifest.dataset_checksum,
    )
    catalog_result = catalog.ingest_batch(catalog_batch)
    search_result = search.ingest_batch(search_batch)
    catalog_refs = frozenset(
        source_ref_key(record.representation.source_ref) for record in catalog_batch.records
    )
    search_refs = frozenset(source_ref_key(record.source_ref) for record in search_batch.records)
    return DataPackStorageLoadResult(
        catalog_source_rows=catalog_result.source_offer_count,
        search_point_count=search_result.point_count,
        embedding_calls=0,
        catalog_refs=catalog_refs,
        search_refs=search_refs,
    )


def validate_storage_load_result(
    result: DataPackStorageLoadResult,
    *,
    expected_count: int,
) -> ValidationReport:
    checks = (
        ValidationCheck(
            name="relational_load_count",
            status=ValidationStatus.PASS
            if result.catalog_source_rows == expected_count
            else ValidationStatus.FAIL,
            detail=f"count={result.catalog_source_rows} expected={expected_count}",
        ),
        ValidationCheck(
            name="vector_load_count",
            status=ValidationStatus.PASS
            if result.search_point_count == expected_count
            else ValidationStatus.FAIL,
            detail=f"count={result.search_point_count} expected={expected_count}",
        ),
        ValidationCheck(
            name="zero_embedding_calls",
            status=ValidationStatus.PASS if result.embedding_calls == 0 else ValidationStatus.FAIL,
            detail=f"embedding_calls={result.embedding_calls}",
        ),
        ValidationCheck(
            name="catalog_search_ref_equality",
            status=ValidationStatus.PASS
            if result.catalog_refs == result.search_refs
            else ValidationStatus.FAIL,
            detail="catalog and search source refs match",
        ),
    )
    return ValidationReport.from_checks(checks)
