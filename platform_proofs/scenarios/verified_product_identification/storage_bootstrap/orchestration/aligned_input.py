"""Streaming WDC source rows aligned with READY embedding artifact records."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.ports import (
    EmbeddingArtifactReaderPort,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
)
from platform_proofs.scenarios.verified_product_identification.ingest.pipeline.derive_batch import (
    build_catalog_ingest_batch,
)
from platform_proofs.scenarios.verified_product_identification.ingest.source_reader.parquet_dataset import (
    DatasetRow,
    iter_dataset_rows,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapDataError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    CatalogIngestRecord,
)


@dataclass(frozen=True, slots=True)
class AlignedBootstrapInputRecord:
    """One catalog source row paired with its materialized search artifact row."""

    source_row: DatasetRow
    catalog_record: CatalogIngestRecord
    artifact_record: EmbeddingArtifactRecord


def validate_source_artifact_alignment(
    catalog_record: CatalogIngestRecord,
    artifact_record: EmbeddingArtifactRecord,
) -> None:
    if catalog_record.global_row_index != artifact_record.global_row_index:
        raise VpiBootstrapDataError(
            "source/artifact global_row_index mismatch at row "
            f"{catalog_record.global_row_index} vs {artifact_record.global_row_index}"
        )
    source_ref = catalog_record.representation.source_ref
    if source_ref.catalog_id != artifact_record.catalog_id:
        raise VpiBootstrapDataError(
            f"catalog_id mismatch at row {catalog_record.global_row_index}: "
            f"{source_ref.catalog_id} vs {artifact_record.catalog_id}"
        )
    if source_ref.offer_id.value != artifact_record.offer_id:
        raise VpiBootstrapDataError(
            f"offer_id mismatch at row {catalog_record.global_row_index}: "
            f"{source_ref.offer_id.value} vs {artifact_record.offer_id}"
        )
    if source_ref.source_revision != artifact_record.source_revision:
        raise VpiBootstrapDataError(
            f"source_revision mismatch at row {catalog_record.global_row_index}"
        )
    if catalog_record.representation.derivation_version != artifact_record.derivation_version:
        raise VpiBootstrapDataError(
            f"derivation_version mismatch at row {catalog_record.global_row_index}: "
            f"{catalog_record.representation.derivation_version} "
            f"vs {artifact_record.derivation_version}"
        )


def iter_artifact_records(
    reader: EmbeddingArtifactReaderPort,
    manifest: EmbeddingArtifactManifest,
    *,
    start_row_index: int,
    max_records: int,
) -> Iterator[EmbeddingArtifactRecord]:
    rows_emitted = 0
    for descriptor in manifest.committed_shards:
        if descriptor.last_global_row_index < start_row_index:
            continue
        for record in reader.iterate_shard_records(descriptor):
            if record.global_row_index < start_row_index:
                continue
            if rows_emitted >= max_records:
                return
            yield record
            rows_emitted += 1


@dataclass(slots=True)
class AlignedBootstrapInputIterator:
    """Bounded-memory iterator pairing WDC source batches with artifact rows."""

    dataset_path: Path
    artifact_reader: EmbeddingArtifactReaderPort
    artifact_manifest: EmbeddingArtifactManifest
    catalog_id: str
    source_revision: str | None
    source_batch_size: int
    start_row_index: int
    start_batch_ordinal: int
    max_records: int

    def __iter__(self) -> Iterator[tuple[int, tuple[AlignedBootstrapInputRecord, ...]]]:
        artifact_iter = iter_artifact_records(
            self.artifact_reader,
            self.artifact_manifest,
            start_row_index=self.start_row_index,
            max_records=self.max_records,
        )
        for batch_ordinal, source_rows in iter_dataset_rows(
            self.dataset_path,
            batch_size=self.source_batch_size,
            start_row_index=self.start_row_index,
            start_batch_ordinal=self.start_batch_ordinal,
            max_records=self.max_records,
        ):
            catalog_batch = build_catalog_ingest_batch(
                batch_ordinal=batch_ordinal,
                rows=source_rows,
                catalog_id=self.catalog_id,
                source_revision=self.source_revision,
            )
            aligned_records: list[AlignedBootstrapInputRecord] = []
            for catalog_record in catalog_batch.records:
                try:
                    artifact_record = next(artifact_iter)
                except StopIteration:
                    raise VpiBootstrapDataError(
                        f"artifact ended before source row {catalog_record.global_row_index}"
                    ) from None
                validate_source_artifact_alignment(catalog_record, artifact_record)
                aligned_records.append(
                    AlignedBootstrapInputRecord(
                        source_row=DatasetRow(
                            global_row_index=catalog_record.global_row_index,
                            record_json=catalog_record.record_json,
                        ),
                        catalog_record=catalog_record,
                        artifact_record=artifact_record,
                    )
                )
            yield batch_ordinal, tuple(aligned_records)
