"""Filesystem data pack reader."""

from __future__ import annotations

from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    verify_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.validation import (
    assert_validation_pass,
    validate_cross_artifact_identity,
    validate_embedding_records,
    validate_relational_records,
    validate_semantic_text_hashes,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    read_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DataPackPaths,
    resolve_data_pack_paths,
    shard_file_name,
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
        self._relational_records: tuple | None = None
        self._embedding_records: tuple | None = None

    @property
    def paths(self) -> DataPackPaths:
        return self._paths

    def read_manifest(self) -> DataPackManifest:
        if self._manifest is None:
            self._manifest = read_manifest_file(self._paths.manifest_file)
        return self._manifest

    def read_relational_records(self):
        from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
            RelationalDataPackRecord,
        )

        if self._relational_records is None:
            manifest = self.read_manifest()
            path = self._paths.relational_dir / manifest.relational_shard_file
            self._relational_records = read_relational_parquet(path)
        return self._relational_records

    def read_embedding_records(self):
        from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
            EmbeddingDataPackRecord,
        )

        if self._embedding_records is None:
            manifest = self.read_manifest()
            path = self._paths.embeddings_dir / manifest.embedding_shard_file
            self._embedding_records = read_embedding_parquet(
                path,
                expected_dimension=manifest.embedding_identity.dimension,
            )
        return self._embedding_records

    def validate_integrity(self) -> ValidationReport:
        manifest = self.read_manifest()
        verify_sha256sums(self._paths.checksums_file, self._paths.root)
        relational_records = self.read_relational_records()
        embedding_records = self.read_embedding_records()
        reports = [
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
