"""Provider-neutral VPI embedding artifact materialization orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    flatten_lexical_text,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    VpiEmbeddingMaterializationConfig,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    EmbeddingMaterializationProviderError,
    VpiEmbeddingMaterializationError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.ports import (
    EmbeddingArtifactWriterPort,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.results import (
    MaterializationRunReport,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.compatibility import (
    EmbeddingArtifactCompatibilityIdentity,
    assert_manifest_compatible,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EmbeddingArtifactManifest,
    EmbeddingArtifactState,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.run_target import (
    assert_requested_target_not_below_checkpoint,
    checkpoint_meets_target,
    manifest_run_target_value,
    resolve_requested_target_rows,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.orchestration.embedding_batches import (
    iter_embedding_slices,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.validation.ready_gate import (
    evaluate_artifact_ready_gate,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.validation.vectors import (
    validate_embedding_batch_vectors,
)
from platform_proofs.scenarios.verified_product_identification.ingest.pipeline.derive_batch import (
    build_catalog_ingest_batch,
)
from platform_proofs.scenarios.verified_product_identification.ingest.source_reader.parquet_dataset import (
    count_rows_to_ingest,
    iter_dataset_rows,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    EmbeddingExecutionPort,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    EmbeddingProbeResult,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.identity import (
    DatasetIdentity,
    resolve_dataset_identity,
)


@dataclass(frozen=True, slots=True)
class VpiEmbeddingMaterializationDependencies:
    artifact_writer: EmbeddingArtifactWriterPort
    embedding: EmbeddingExecutionPort


@dataclass(slots=True)
class EmbeddingMaterializationOrchestrator:
    config: VpiEmbeddingMaterializationConfig
    dependencies: VpiEmbeddingMaterializationDependencies

    def run(self, *, validate_only: bool = False) -> MaterializationRunReport:
        started_total = time.perf_counter()
        embedding_probe_result: EmbeddingProbeResult | None = None
        manifest: EmbeddingArtifactManifest | None = None
        rows_materialized = 0
        embedding_batches = 0
        shards_committed = 0
        embedding_calls = 0
        elapsed_embedding = 0.0
        elapsed_derive = 0.0
        elapsed_artifact_write = 0.0

        try:
            dataset_identity = resolve_dataset_identity(
                dataset_path=self.config.dataset_path,
                dataset_manifest_path=self.config.dataset_manifest_path,
                verification_mode=self.config.dataset_verification_mode,
            )
            expected_identity = self._expected_compatibility_identity(dataset_identity)
            requested_target_rows = resolve_requested_target_rows(
                max_records=self.config.max_records,
                dataset_record_count=dataset_identity.dataset_record_count,
            )
            persisted_run_target = manifest_run_target_value(
                requested_target_rows,
                dataset_identity.dataset_record_count,
            )
            manifest = self._initial_manifest(
                dataset_identity,
                expected_identity,
                target_max_records=persisted_run_target,
            )

            existing = self.dependencies.artifact_writer.read_manifest()
            if existing is not None:
                assert_manifest_compatible(existing=existing, expected=expected_identity)
                assert_requested_target_not_below_checkpoint(
                    requested_target_rows=requested_target_rows,
                    checkpoint_rows_materialized=existing.checkpoint_rows_materialized,
                )
                manifest = existing.with_run_target(persisted_run_target)
                manifest = self.dependencies.artifact_writer.reconcile_orphan_shards(manifest)
                if manifest is not existing:
                    self.dependencies.artifact_writer.write_manifest(manifest)
                rows_materialized = manifest.checkpoint_rows_materialized
                shards_committed = manifest.shard_count

                if (
                    existing.state is EmbeddingArtifactState.READY
                    and checkpoint_meets_target(
                        checkpoint_rows_materialized=existing.checkpoint_rows_materialized,
                        requested_target_rows=requested_target_rows,
                    )
                ):
                    artifact_validation = self.dependencies.artifact_writer.validate(manifest)
                    ready_validation = evaluate_artifact_ready_gate(
                        manifest=manifest,
                        artifact_report=artifact_validation,
                        checkpoint_complete=True,
                    )
                    if validate_only or ready_validation.status is ValidationStatus.PASS:
                        return MaterializationRunReport(
                            final_state=manifest.state,
                            manifest=manifest,
                            validation=ready_validation,
                            embedding_probe=None,
                            rows_materialized=rows_materialized,
                            embedding_batches=0,
                            shards_committed=shards_committed,
                            embedding_calls=0,
                            elapsed_total_seconds=time.perf_counter() - started_total,
                            elapsed_embedding_seconds=0.0,
                            elapsed_derive_seconds=0.0,
                            elapsed_artifact_write_seconds=0.0,
                            failure_stage=None,
                            failure_detail=None,
                        )
                    raise EmbeddingMaterializationProviderError(
                        ready_validation.checks[-1].detail
                        if ready_validation.checks
                        else "READY validation failed"
                    )

            if validate_only:
                artifact_validation = self.dependencies.artifact_writer.validate(manifest)
                return MaterializationRunReport(
                    final_state=manifest.state,
                    manifest=manifest,
                    validation=artifact_validation,
                    embedding_probe=None,
                    rows_materialized=rows_materialized,
                    embedding_batches=0,
                    shards_committed=shards_committed,
                    embedding_calls=0,
                    elapsed_total_seconds=time.perf_counter() - started_total,
                    elapsed_embedding_seconds=0.0,
                    elapsed_derive_seconds=0.0,
                    elapsed_artifact_write_seconds=0.0,
                    failure_stage=None,
                    failure_detail=None,
                )

            embedding_probe_result = self.dependencies.embedding.probe()
            if embedding_probe_result.status is not ValidationStatus.PASS:
                raise EmbeddingMaterializationProviderError(embedding_probe_result.detail)

            manifest = manifest.with_state(EmbeddingArtifactState.INITIALIZING)
            self.dependencies.artifact_writer.prepare(manifest)
            self.dependencies.artifact_writer.write_manifest(manifest)

            manifest = manifest.with_state(EmbeddingArtifactState.MATERIALIZING)
            self.dependencies.artifact_writer.write_manifest(manifest)

            start_row = manifest.checkpoint_rows_materialized
            remaining_to_materialize = count_rows_to_ingest(
                start_row_index=start_row,
                max_records=self.config.max_records,
                dataset_record_count=dataset_identity.dataset_record_count,
            )
            embedding_model = self.config.embedding_configuration.model
            if embedding_model is None:
                raise EmbeddingMaterializationProviderError("embedding model is required")

            shard_buffer: list[EmbeddingArtifactRecord] = []
            next_shard_ordinal = len(manifest.committed_shards)
            committed_shards = list(manifest.committed_shards)

            for _, rows in iter_dataset_rows(
                self.config.dataset_path,
                batch_size=self.config.source_read_batch_size,
                start_row_index=start_row,
                start_batch_ordinal=0,
                max_records=remaining_to_materialize,
            ):
                derive_started = time.perf_counter()
                catalog_batch = build_catalog_ingest_batch(
                    batch_ordinal=0,
                    rows=rows,
                    catalog_id=self.config.catalog_id,
                    source_revision=self.config.source_revision,
                )
                elapsed_derive += time.perf_counter() - derive_started

                semantic_texts = [
                    record.representation.semantic.semantic_text
                    for record in catalog_batch.records
                ]
                batch_vectors: list[tuple[float, ...]] = []
                for _, text_slice in iter_embedding_slices(
                    semantic_texts,
                    batch_size=self.config.embedding_batch_size,
                ):
                    embed_started = time.perf_counter()
                    try:
                        vectors = self.dependencies.embedding.embed_batch(text_slice)
                    except EmbeddingMaterializationProviderError:
                        raise
                    except Exception as exc:
                        raise EmbeddingMaterializationProviderError(
                            "embedding batch failed"
                        ) from exc
                    elapsed_embedding += time.perf_counter() - embed_started
                    embedding_calls += 1
                    embedding_batches += 1
                    validate_embedding_batch_vectors(
                        vectors=vectors,
                        expected_count=len(text_slice),
                        expected_dimension=self.config.embedding_configuration.expected_dimension,
                    )
                    batch_vectors.extend(vectors)

                for record_index, record in enumerate(catalog_batch.records):
                    source_ref = record.representation.source_ref
                    artifact_record = EmbeddingArtifactRecord(
                        global_row_index=record.global_row_index,
                        logical_point_id=search_representation_point_id(
                            catalog_id=source_ref.catalog_id,
                            offer_id=source_ref.offer_id.value,
                            derivation_version=record.representation.derivation_version,
                        ),
                        catalog_id=source_ref.catalog_id,
                        offer_id=source_ref.offer_id.value,
                        source_revision=source_ref.source_revision,
                        derivation_version=record.representation.derivation_version,
                        semantic_text=record.representation.semantic.semantic_text,
                        lexical_text=flatten_lexical_text(record.representation.lexical),
                        embedding_provider=self.config.embedding_configuration.provider,
                        embedding_model=embedding_model,
                        embedding_dimension=self.config.embedding_configuration.expected_dimension,
                        dense_embedding=batch_vectors[record_index],
                    )
                    shard_buffer.append(artifact_record)

                    while len(shard_buffer) >= self.config.artifact_shard_size:
                        shard_records = tuple(shard_buffer[: self.config.artifact_shard_size])
                        shard_buffer = shard_buffer[self.config.artifact_shard_size :]
                        write_started = time.perf_counter()
                        descriptor = self.dependencies.artifact_writer.write_shard(
                            next_shard_ordinal,
                            shard_records,
                        )
                        elapsed_artifact_write += time.perf_counter() - write_started
                        committed_shards.append(descriptor)
                        rows_materialized = descriptor.last_global_row_index + 1
                        manifest = manifest.with_checkpoint(
                            shard_ordinal=descriptor.shard_ordinal,
                            rows_materialized=rows_materialized,
                            committed_shards=tuple(committed_shards),
                        ).with_state(EmbeddingArtifactState.MATERIALIZING)
                        self.dependencies.artifact_writer.write_manifest(manifest)
                        shards_committed += 1
                        next_shard_ordinal += 1

            if shard_buffer:
                write_started = time.perf_counter()
                descriptor = self.dependencies.artifact_writer.write_shard(
                    next_shard_ordinal,
                    tuple(shard_buffer),
                )
                elapsed_artifact_write += time.perf_counter() - write_started
                committed_shards.append(descriptor)
                rows_materialized = descriptor.last_global_row_index + 1
                manifest = manifest.with_checkpoint(
                    shard_ordinal=descriptor.shard_ordinal,
                    rows_materialized=rows_materialized,
                    committed_shards=tuple(committed_shards),
                ).with_state(EmbeddingArtifactState.MATERIALIZING)
                self.dependencies.artifact_writer.write_manifest(manifest)
                shards_committed += 1

            manifest = manifest.with_state(EmbeddingArtifactState.VALIDATING)
            self.dependencies.artifact_writer.write_manifest(manifest)

            artifact_validation = self.dependencies.artifact_writer.validate(manifest)
            checkpoint_complete = checkpoint_meets_target(
                checkpoint_rows_materialized=manifest.checkpoint_rows_materialized,
                requested_target_rows=requested_target_rows,
            )
            ready_validation = evaluate_artifact_ready_gate(
                manifest=manifest,
                artifact_report=artifact_validation,
                checkpoint_complete=checkpoint_complete,
            )

            if ready_validation.status is ValidationStatus.PASS:
                manifest = EmbeddingArtifactManifest(
                    state=EmbeddingArtifactState.READY,
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
                    created_at_utc=manifest.created_at_utc,
                    finalized_at_utc=datetime.now(UTC).isoformat(),
                    failure_stage=None,
                    failure_detail=None,
                )
            else:
                manifest = manifest.with_state(
                    EmbeddingArtifactState.FAILED,
                    failure_stage="ready_gate",
                    failure_detail=ready_validation.checks[-1].detail,
                )
            self.dependencies.artifact_writer.write_manifest(manifest)

            return MaterializationRunReport(
                final_state=manifest.state,
                manifest=manifest,
                validation=ready_validation,
                embedding_probe=embedding_probe_result,
                rows_materialized=rows_materialized,
                embedding_batches=embedding_batches,
                shards_committed=shards_committed,
                embedding_calls=embedding_calls,
                elapsed_total_seconds=time.perf_counter() - started_total,
                elapsed_embedding_seconds=elapsed_embedding,
                elapsed_derive_seconds=elapsed_derive,
                elapsed_artifact_write_seconds=elapsed_artifact_write,
                failure_stage=manifest.failure_stage,
                failure_detail=manifest.failure_detail,
            )
        except VpiEmbeddingMaterializationError as exc:
            if manifest is not None:
                failed = manifest.with_state(
                    EmbeddingArtifactState.FAILED,
                    failure_stage=exc.__class__.__name__,
                    failure_detail=str(exc),
                )
                try:
                    self.dependencies.artifact_writer.write_manifest(failed)
                except Exception:
                    pass
                manifest = failed
            return MaterializationRunReport(
                final_state=EmbeddingArtifactState.FAILED,
                manifest=manifest,
                validation=None,
                embedding_probe=embedding_probe_result,
                rows_materialized=rows_materialized,
                embedding_batches=embedding_batches,
                shards_committed=shards_committed,
                embedding_calls=embedding_calls,
                elapsed_total_seconds=time.perf_counter() - started_total,
                elapsed_embedding_seconds=elapsed_embedding,
                elapsed_derive_seconds=elapsed_derive,
                elapsed_artifact_write_seconds=elapsed_artifact_write,
                failure_stage=exc.__class__.__name__,
                failure_detail=str(exc),
            )

    def _expected_compatibility_identity(
        self,
        dataset_identity: DatasetIdentity,
    ) -> EmbeddingArtifactCompatibilityIdentity:
        embedding = self.config.embedding_configuration
        model = embedding.model
        if model is None:
            raise EmbeddingMaterializationProviderError("embedding model is required")
        return EmbeddingArtifactCompatibilityIdentity(
            dataset_checksum=dataset_identity.dataset_checksum,
            dataset_record_count=dataset_identity.dataset_record_count,
            search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
            embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
            embedding_provider=embedding.provider,
            embedding_model=model,
            embedding_dimension=embedding.expected_dimension,
            artifact_schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
            catalog_id=self.config.catalog_id,
            source_revision=self.config.source_revision,
        )

    def _initial_manifest(
        self,
        dataset_identity: DatasetIdentity,
        expected: EmbeddingArtifactCompatibilityIdentity,
        *,
        target_max_records: int | None,
    ) -> EmbeddingArtifactManifest:
        return EmbeddingArtifactManifest(
            state=EmbeddingArtifactState.INITIALIZING,
            artifact_schema_version=expected.artifact_schema_version,
            dataset_path=dataset_identity.dataset_path,
            dataset_checksum=expected.dataset_checksum,
            dataset_record_count=expected.dataset_record_count,
            search_representation_derivation_version=expected.search_representation_derivation_version,
            embedding_configuration_version=expected.embedding_configuration_version,
            embedding_provider=expected.embedding_provider,
            embedding_model=expected.embedding_model,
            embedding_dimension=expected.embedding_dimension,
            catalog_id=expected.catalog_id,
            source_revision=self.config.source_revision,
            checkpoint_shard_ordinal=None,
            checkpoint_rows_materialized=0,
            target_max_records=target_max_records,
            total_artifact_record_count=0,
            shard_count=0,
            committed_shards=(),
        )
