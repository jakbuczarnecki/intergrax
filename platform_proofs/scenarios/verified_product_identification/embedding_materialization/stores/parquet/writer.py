"""Filesystem Parquet embedding artifact writer."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactWriteError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
    EmbeddingArtifactShardDescriptor,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.manifest_io import (
    read_manifest_file,
    write_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.parquet_codec import (
    write_records_parquet,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.paths import (
    manifest_path,
    shard_file_name,
    shard_path,
    temp_shard_path,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.reconciliation import (
    reconcile_orphan_shards,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.shard_validation import (
    build_shard_descriptor,
    validate_shard_descriptor_continuity,
    validate_shard_file,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)


class ParquetFilesystemArtifactWriter:
    """Reference ``EmbeddingArtifactWriterPort`` — atomic shard rename on POSIX/Windows."""

    def __init__(self, artifact_dir: Path) -> None:
        self._artifact_dir = artifact_dir

    def prepare(self, manifest: EmbeddingArtifactManifest) -> None:
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_path(self._artifact_dir)
        if not manifest_file.is_file():
            created = manifest
            if manifest.created_at_utc is None:
                created = EmbeddingArtifactManifest(
                    state=manifest.state,
                    artifact_schema_version=manifest.artifact_schema_version,
                    dataset_path=manifest.dataset_path,
                    dataset_checksum=manifest.dataset_checksum,
                    dataset_record_count=manifest.dataset_record_count,
                    search_representation_derivation_version=manifest.search_representation_derivation_version,
                    embedding_configuration_version=manifest.embedding_configuration_version,
                    embedding_provider=manifest.embedding_provider,
                    embedding_model=manifest.embedding_model,
                    embedding_dimension=manifest.embedding_dimension,
                    catalog_id=manifest.catalog_id,
                    source_revision=manifest.source_revision,
                    checkpoint_shard_ordinal=manifest.checkpoint_shard_ordinal,
                    checkpoint_rows_materialized=manifest.checkpoint_rows_materialized,
                    target_max_records=manifest.target_max_records,
                    total_artifact_record_count=manifest.total_artifact_record_count,
                    shard_count=manifest.shard_count,
                    committed_shards=manifest.committed_shards,
                    created_at_utc=datetime.now(UTC).isoformat(),
                    finalized_at_utc=manifest.finalized_at_utc,
                    failure_stage=manifest.failure_stage,
                    failure_detail=manifest.failure_detail,
                )
            write_manifest_file(manifest_file, created)

    def write_shard(
        self,
        shard_ordinal: int,
        records: Sequence[EmbeddingArtifactRecord],
    ) -> EmbeddingArtifactShardDescriptor:
        if not records:
            raise ArtifactWriteError("cannot write empty shard")
        embedding_dimension = records[0].embedding_dimension
        temp_path = temp_shard_path(self._artifact_dir, shard_ordinal)
        final_path = shard_path(self._artifact_dir, shard_ordinal)
        if temp_path.exists():
            temp_path.unlink()
        write_records_parquet(temp_path, records, embedding_dimension=embedding_dimension)
        descriptor = build_shard_descriptor(
            shard_ordinal=shard_ordinal,
            file_name=shard_file_name(shard_ordinal),
            records=records,
            file_path=temp_path,
        )
        validate_shard_file(
            descriptor=descriptor,
            file_path=temp_path,
            expected_dimension=embedding_dimension,
        )
        temp_path.replace(final_path)
        return descriptor

    def read_manifest(self) -> EmbeddingArtifactManifest | None:
        manifest_file = manifest_path(self._artifact_dir)
        if not manifest_file.is_file():
            return None
        return read_manifest_file(manifest_file)

    def write_manifest(self, manifest: EmbeddingArtifactManifest) -> None:
        write_manifest_file(manifest_path(self._artifact_dir), manifest)

    def reconcile_orphan_shards(
        self,
        manifest: EmbeddingArtifactManifest,
    ) -> EmbeddingArtifactManifest:
        return reconcile_orphan_shards(artifact_dir=self._artifact_dir, manifest=manifest)

    def validate(self, manifest: EmbeddingArtifactManifest) -> ValidationReport:
        checks: list[ValidationCheck] = []
        try:
            validate_shard_descriptor_continuity(manifest.committed_shards)
            for descriptor in manifest.committed_shards:
                validate_shard_file(
                    descriptor=descriptor,
                    file_path=self._artifact_dir / descriptor.file_name,
                    expected_dimension=manifest.embedding_dimension,
                )
            row_total = sum(descriptor.record_count for descriptor in manifest.committed_shards)
            if row_total != manifest.checkpoint_rows_materialized:
                checks.append(
                    ValidationCheck(
                        name="checkpoint_row_total",
                        status=ValidationStatus.FAIL,
                        detail=(
                            f"checkpoint_rows={manifest.checkpoint_rows_materialized} "
                            f"shard_rows={row_total}"
                        ),
                    )
                )
            else:
                checks.append(
                    ValidationCheck(
                        name="checkpoint_row_total",
                        status=ValidationStatus.PASS,
                        detail=f"rows={row_total}",
                    )
                )
            checks.append(
                ValidationCheck(
                    name="shard_continuity",
                    status=ValidationStatus.PASS,
                    detail=f"shards={manifest.shard_count}",
                )
            )
        except Exception as exc:
            checks.append(
                ValidationCheck(
                    name="artifact_integrity",
                    status=ValidationStatus.FAIL,
                    detail=str(exc),
                )
            )
        return ValidationReport.from_checks(tuple(checks))

    def close(self) -> None:
        return None
