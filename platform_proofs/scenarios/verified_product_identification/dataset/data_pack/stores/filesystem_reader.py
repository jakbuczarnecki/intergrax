"""Filesystem data pack reader."""

from __future__ import annotations

from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    verify_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.compatibility import (
    validate_shard_index_contract,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.validation import (
    assert_validation_pass,
    validate_cross_artifact_identity,
    validate_embedding_records,
    validate_relational_records,
    validate_semantic_text_hashes,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    read_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DataPackPaths,
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardIndex,
    read_shard_index_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    read_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    read_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)


class FilesystemDataPackReader:
    def __init__(self, root: Path) -> None:
        self._paths = resolve_data_pack_paths(root)
        self._manifest: DataPackManifest | None = None
        self._shard_index: ShardIndex | None = None
        self._relational_records: tuple[RelationalDataPackRecord, ...] | None = None
        self._embedding_records: tuple[EmbeddingDataPackRecord, ...] | None = None

    @property
    def paths(self) -> DataPackPaths:
        return self._paths

    def read_manifest(self) -> DataPackManifest:
        if self._manifest is None:
            self._manifest = read_manifest_file(self._paths.manifest_file)
        return self._manifest

    def read_shard_index(self) -> ShardIndex:
        if self._shard_index is None:
            manifest = self.read_manifest()
            shard_index_path = self._paths.root / manifest.shards_index_path
            self._shard_index = read_shard_index_file(shard_index_path)
        return self._shard_index

    def _sorted_relational_records(
        self,
        records: tuple[RelationalDataPackRecord, ...],
    ) -> tuple[RelationalDataPackRecord, ...]:
        return tuple(sorted(records, key=lambda record: record.global_row_index))

    def read_relational_records(self) -> tuple[RelationalDataPackRecord, ...]:
        if self._relational_records is None:
            shard_index = self.read_shard_index()
            records: list[RelationalDataPackRecord] = []
            for descriptor in shard_index.relational_shards:
                shard_path = self._paths.root / descriptor.relative_path
                records.extend(read_relational_parquet(shard_path))
            self._relational_records = self._sorted_relational_records(tuple(records))
        return self._relational_records

    def read_embedding_records(self) -> tuple[EmbeddingDataPackRecord, ...]:
        if self._embedding_records is None:
            manifest = self.read_manifest()
            shard_index = self.read_shard_index()
            records: list[EmbeddingDataPackRecord] = []
            for descriptor in shard_index.embedding_shards:
                shard_path = self._paths.root / descriptor.relative_path
                records.extend(
                    read_embedding_parquet(
                        shard_path,
                        expected_dimension=manifest.embedding_identity.dimension,
                    )
                )
            records_by_ref = {
                (
                    record.source_ref.catalog_id,
                    record.source_ref.offer_id.value,
                    record.source_ref.source_revision,
                ): record
                for record in records
            }
            ordered_records: list[EmbeddingDataPackRecord] = []
            for relational_record in self.read_relational_records():
                key = (
                    relational_record.source_ref.catalog_id,
                    relational_record.source_ref.offer_id.value,
                    relational_record.source_ref.source_revision,
                )
                embedding_record = records_by_ref.get(key)
                if embedding_record is None:
                    msg = f"missing embedding row for {relational_record.source_ref.offer_id.value}"
                    raise ValueError(msg)
                ordered_records.append(embedding_record)
            self._embedding_records = tuple(ordered_records)
        return self._embedding_records

    def validate_integrity(self) -> ValidationReport:
        manifest = self.read_manifest()
        shard_index = self.read_shard_index()
        verify_sha256sums(self._paths.checksums_file, self._paths.root)
        relational_records = self.read_relational_records()
        embedding_records = self.read_embedding_records()
        reports = [
            validate_shard_index_contract(shard_index),
            validate_relational_records(
                relational_records,
                expected_count=manifest.record_count,
            ),
            validate_embedding_records(
                embedding_records,
                expected_count=manifest.record_count,
                expected_dimension=manifest.embedding_identity.dimension,
            ),
            validate_cross_artifact_identity(relational_records, embedding_records),
            validate_semantic_text_hashes(relational_records, embedding_records),
            ValidationReport.from_checks(
                (
                    ValidationCheck(
                        name="checksum_validation",
                        status=ValidationStatus.PASS,
                        detail="SHA256SUMS verified",
                    ),
                )
            ),
        ]
        checks: list[ValidationCheck] = []
        for report in reports:
            checks.extend(report.checks)
        return ValidationReport.from_checks(tuple(checks))

    def close(self) -> None:
        return None
