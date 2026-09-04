"""Provider-neutral embedding artifact ports."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.compatibility import (
    EmbeddingArtifactCompatibilityIdentity,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
    EmbeddingArtifactShardDescriptor,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationReport,
)


class EmbeddingArtifactWriterPort(Protocol):
    def prepare(self, manifest: EmbeddingArtifactManifest) -> None: ...

    def write_shard(
        self,
        shard_ordinal: int,
        records: Sequence[EmbeddingArtifactRecord],
    ) -> EmbeddingArtifactShardDescriptor: ...

    def read_manifest(self) -> EmbeddingArtifactManifest | None: ...

    def write_manifest(self, manifest: EmbeddingArtifactManifest) -> None: ...

    def reconcile_orphan_shards(
        self,
        manifest: EmbeddingArtifactManifest,
    ) -> EmbeddingArtifactManifest: ...

    def validate(self, manifest: EmbeddingArtifactManifest) -> ValidationReport: ...

    def close(self) -> None: ...


class EmbeddingArtifactReaderPort(Protocol):
    def read_manifest(self) -> EmbeddingArtifactManifest: ...

    def iterate_shard_records(
        self,
        descriptor: EmbeddingArtifactShardDescriptor,
    ) -> Iterator[EmbeddingArtifactRecord]: ...

    def validate_identity(
        self,
        expected: EmbeddingArtifactCompatibilityIdentity,
    ) -> ValidationReport: ...

    def close(self) -> None: ...
