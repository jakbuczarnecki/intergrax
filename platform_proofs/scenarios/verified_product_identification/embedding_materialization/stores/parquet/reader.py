"""Filesystem Parquet embedding artifact reader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.compatibility import (
    EmbeddingArtifactCompatibilityIdentity,
    assert_manifest_compatible,
    compatibility_identity_from_manifest,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
    EmbeddingArtifactShardDescriptor,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.manifest_io import (
    read_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.parquet_codec import (
    read_records_parquet,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.paths import (
    manifest_path,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.shard_validation import (
    validate_shard_descriptor_continuity,
    validate_shard_file,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)


class ParquetFilesystemArtifactReader:
    """Reference ``EmbeddingArtifactReaderPort`` — no embedding provider dependency."""

    def __init__(self, artifact_dir: Path) -> None:
        self._artifact_dir = artifact_dir

    def read_manifest(self) -> EmbeddingArtifactManifest:
        manifest_file = manifest_path(self._artifact_dir)
        if not manifest_file.is_file():
            raise ArtifactIntegrityError(f"artifact manifest missing: {manifest_file}")
        return read_manifest_file(manifest_file)

    def iterate_shard_records(
        self,
        descriptor: EmbeddingArtifactShardDescriptor,
    ) -> Iterator[EmbeddingArtifactRecord]:
        manifest = self.read_manifest()
        records = read_records_parquet(
            self._artifact_dir / descriptor.file_name,
            expected_dimension=manifest.embedding_dimension,
        )
        for record in records:
            yield record

    def validate_identity(
        self,
        expected: EmbeddingArtifactCompatibilityIdentity,
    ) -> ValidationReport:
        manifest = self.read_manifest()
        try:
            assert_manifest_compatible(existing=manifest, expected=expected)
            validate_shard_descriptor_continuity(manifest.committed_shards)
            for descriptor in manifest.committed_shards:
                validate_shard_file(
                    descriptor=descriptor,
                    file_path=self._artifact_dir / descriptor.file_name,
                    expected_dimension=manifest.embedding_dimension,
                )
            identity = compatibility_identity_from_manifest(manifest)
            detail = (
                f"provider={identity.embedding_provider} "
                f"model={identity.embedding_model} "
                f"dimension={identity.embedding_dimension}"
            )
            return ValidationReport.from_checks(
                (
                    ValidationCheck(
                        name="artifact_identity",
                        status=ValidationStatus.PASS,
                        detail=detail,
                    ),
                )
            )
        except Exception as exc:
            return ValidationReport.from_checks(
                (
                    ValidationCheck(
                        name="artifact_identity",
                        status=ValidationStatus.FAIL,
                        detail=str(exc),
                    ),
                )
            )

    def close(self) -> None:
        return None
