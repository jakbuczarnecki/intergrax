"""Scenario-owned embedding artifact manifest contract and lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EmbeddingArtifactState(str, Enum):
    INITIALIZING = "INITIALIZING"
    MATERIALIZING = "MATERIALIZING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"


EMBEDDING_ARTIFACT_SCHEMA_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactShardDescriptor:
    shard_ordinal: int
    file_name: str
    first_global_row_index: int
    last_global_row_index: int
    record_count: int
    sha256_checksum: str


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactManifest:
    """Source of embedding artifact compatibility identity."""

    state: EmbeddingArtifactState
    artifact_schema_version: str
    dataset_path: str
    dataset_checksum: str
    dataset_record_count: int
    search_representation_derivation_version: str
    embedding_configuration_version: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    catalog_id: str
    source_revision: str | None
    checkpoint_shard_ordinal: int | None
    checkpoint_rows_materialized: int
    target_max_records: int | None
    total_artifact_record_count: int
    shard_count: int
    committed_shards: tuple[EmbeddingArtifactShardDescriptor, ...]
    created_at_utc: str | None = None
    finalized_at_utc: str | None = None
    failure_stage: str | None = None
    failure_detail: str | None = None

    def with_state(
        self,
        state: EmbeddingArtifactState,
        *,
        failure_stage: str | None = None,
        failure_detail: str | None = None,
    ) -> EmbeddingArtifactManifest:
        return EmbeddingArtifactManifest(
            state=state,
            artifact_schema_version=self.artifact_schema_version,
            dataset_path=self.dataset_path,
            dataset_checksum=self.dataset_checksum,
            dataset_record_count=self.dataset_record_count,
            search_representation_derivation_version=self.search_representation_derivation_version,
            embedding_configuration_version=self.embedding_configuration_version,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            catalog_id=self.catalog_id,
            source_revision=self.source_revision,
            checkpoint_shard_ordinal=self.checkpoint_shard_ordinal,
            checkpoint_rows_materialized=self.checkpoint_rows_materialized,
            target_max_records=self.target_max_records,
            total_artifact_record_count=self.total_artifact_record_count,
            shard_count=self.shard_count,
            committed_shards=self.committed_shards,
            created_at_utc=self.created_at_utc,
            finalized_at_utc=self.finalized_at_utc,
            failure_stage=failure_stage,
            failure_detail=failure_detail,
        )

    def with_run_target(self, target_max_records: int | None) -> EmbeddingArtifactManifest:
        return EmbeddingArtifactManifest(
            state=self.state,
            artifact_schema_version=self.artifact_schema_version,
            dataset_path=self.dataset_path,
            dataset_checksum=self.dataset_checksum,
            dataset_record_count=self.dataset_record_count,
            search_representation_derivation_version=self.search_representation_derivation_version,
            embedding_configuration_version=self.embedding_configuration_version,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            catalog_id=self.catalog_id,
            source_revision=self.source_revision,
            checkpoint_shard_ordinal=self.checkpoint_shard_ordinal,
            checkpoint_rows_materialized=self.checkpoint_rows_materialized,
            target_max_records=target_max_records,
            total_artifact_record_count=self.total_artifact_record_count,
            shard_count=self.shard_count,
            committed_shards=self.committed_shards,
            created_at_utc=self.created_at_utc,
            finalized_at_utc=self.finalized_at_utc,
            failure_stage=self.failure_stage,
            failure_detail=self.failure_detail,
        )

    def with_checkpoint(
        self,
        *,
        shard_ordinal: int,
        rows_materialized: int,
        committed_shards: tuple[EmbeddingArtifactShardDescriptor, ...],
    ) -> EmbeddingArtifactManifest:
        return EmbeddingArtifactManifest(
            state=self.state,
            artifact_schema_version=self.artifact_schema_version,
            dataset_path=self.dataset_path,
            dataset_checksum=self.dataset_checksum,
            dataset_record_count=self.dataset_record_count,
            search_representation_derivation_version=self.search_representation_derivation_version,
            embedding_configuration_version=self.embedding_configuration_version,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            catalog_id=self.catalog_id,
            source_revision=self.source_revision,
            checkpoint_shard_ordinal=shard_ordinal,
            checkpoint_rows_materialized=rows_materialized,
            target_max_records=self.target_max_records,
            total_artifact_record_count=rows_materialized,
            shard_count=len(committed_shards),
            committed_shards=committed_shards,
            created_at_utc=self.created_at_utc,
            finalized_at_utc=self.finalized_at_utc,
            failure_stage=self.failure_stage,
            failure_detail=self.failure_detail,
        )
