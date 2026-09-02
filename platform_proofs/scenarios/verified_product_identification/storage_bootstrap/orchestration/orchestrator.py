"""Provider-neutral VPI bootstrap orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass

from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_registry

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    flatten_lexical_text,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    VpiBootstrapConfig,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapCompatibilityError,
    VpiBootstrapError,
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    CatalogBootstrapPort,
    EmbeddingReadinessProbe,
    SearchIndexBootstrapPort,
    SearchIndexIngestBatch,
    SearchIndexIngestRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    BootstrapRunReport,
    EmbeddingProbeResult,
    ValidationReport,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.compatibility import (
    BootstrapCompatibilityIdentity,
    assert_manifest_compatible,
    compatibility_identity_from_manifest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.identity import (
    resolve_dataset_identity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BootstrapState,
    VpiBootstrapManifest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.validation.ready_gate import (
    evaluate_ready_gate,
)
from platform_proofs.scenarios.verified_product_identification.ingest.pipeline.derive_batch import (
    build_catalog_ingest_batch,
)
from platform_proofs.scenarios.verified_product_identification.ingest.source_reader.parquet_dataset import (
    count_rows_to_ingest,
    iter_dataset_rows,
)


@dataclass(frozen=True, slots=True)
class VpiBootstrapDependencies:
    catalog: CatalogBootstrapPort
    search: SearchIndexBootstrapPort
    embedding_probe: EmbeddingReadinessProbe


@dataclass(slots=True)
class VpiBootstrapOrchestrator:
    config: VpiBootstrapConfig
    dependencies: VpiBootstrapDependencies

    def run(self) -> BootstrapRunReport:
        embedding_probe_result: EmbeddingProbeResult | None = None
        manifest: VpiBootstrapManifest | None = None
        batches_completed = 0
        rows_processed = 0

        try:
            dataset_identity = resolve_dataset_identity(
                dataset_path=self.config.dataset_path,
                dataset_manifest_path=self.config.dataset_manifest_path,
                verification_mode=self.config.dataset_verification_mode,
            )
            expected_identity = self._expected_compatibility_identity(dataset_identity)
            manifest = self._initial_manifest(dataset_identity, expected_identity)

            existing = self.dependencies.catalog.read_manifest()
            if existing is not None:
                assert_manifest_compatible(existing=existing, expected=expected_identity)
                if existing.state is BootstrapState.READY and self._checkpoint_complete(existing):
                    validation = self._validate_all(existing)
                    if validation.status is ValidationStatus.PASS:
                        return BootstrapRunReport(
                            final_state=BootstrapState.READY,
                            manifest=existing,
                            validation=validation,
                            embedding_probe=None,
                            batches_completed=existing.checkpoint_batch_ordinal or 0,
                            rows_processed=existing.checkpoint_rows_processed,
                            failure_stage=None,
                            failure_detail=None,
                        )
                manifest = existing

            embedding_report = self.dependencies.embedding_probe.probe()
            if embedding_report.status is not ValidationStatus.PASS:
                raise VpiBootstrapProviderError(embedding_report.checks[0].detail)

            catalog_ready = self.dependencies.catalog.probe_readiness()
            search_ready = self.dependencies.search.probe_readiness()
            if catalog_ready.status is not ValidationStatus.PASS:
                raise VpiBootstrapProviderError(catalog_ready.checks[0].detail)
            if search_ready.status is not ValidationStatus.PASS:
                raise VpiBootstrapProviderError(search_ready.checks[0].detail)

            manifest = manifest.with_state(BootstrapState.INITIALIZING)
            self.dependencies.catalog.prepare(manifest)
            self.dependencies.catalog.write_manifest(manifest)
            self.dependencies.search.prepare(manifest)

            manifest = manifest.with_state(BootstrapState.INGESTING)
            self.dependencies.catalog.write_manifest(manifest)

            start_row = manifest.checkpoint_rows_processed
            start_batch_ordinal = (
                manifest.checkpoint_batch_ordinal + 1
                if manifest.checkpoint_batch_ordinal is not None
                else 0
            )
            remaining_to_ingest = count_rows_to_ingest(
                start_row_index=start_row,
                max_records=self.config.max_records,
                dataset_record_count=dataset_identity.dataset_record_count,
            )
            cumulative_offers = manifest.catalog_source_offer_count
            cumulative_identifiers = manifest.catalog_identifier_count
            cumulative_structured = manifest.catalog_structured_attribute_count
            cumulative_points = manifest.search_point_count

            embedding_model = self.config.embedding_configuration.model
            if embedding_model is None:
                raise VpiBootstrapProviderError("embedding model is required for ingest")
            registry = create_default_registry(embedding_model=embedding_model)
            provider = registry.get(self.config.embedding_configuration.provider)

            for batch_ordinal, rows in iter_dataset_rows(
                self.config.dataset_path,
                batch_size=self.config.source_batch_size,
                start_row_index=start_row,
                start_batch_ordinal=start_batch_ordinal,
                max_records=remaining_to_ingest,
            ):
                started = time.perf_counter()
                catalog_batch = build_catalog_ingest_batch(
                    batch_ordinal=batch_ordinal,
                    rows=rows,
                    catalog_id=self.config.catalog_id,
                    source_revision=self.config.source_revision,
                )
                catalog_result = self.dependencies.catalog.ingest_batch(catalog_batch)

                semantic_texts = [
                    record.representation.semantic.semantic_text
                    for record in catalog_batch.records
                ]
                vectors = provider.embed(semantic_texts)
                search_records: list[SearchIndexIngestRecord] = []
                for record_index, record in enumerate(catalog_batch.records):
                    source_ref = record.representation.source_ref
                    search_records.append(
                        SearchIndexIngestRecord(
                            logical_point_id=search_representation_point_id(
                                catalog_id=source_ref.catalog_id,
                                offer_id=source_ref.offer_id.value,
                                derivation_version=record.representation.derivation_version,
                            ),
                            dense_embedding=tuple(float(value) for value in vectors[record_index]),
                            lexical_text=flatten_lexical_text(record.representation.lexical),
                            source_ref=source_ref,
                            derivation_version=record.representation.derivation_version,
                            dataset_checksum=dataset_identity.dataset_checksum,
                            embedding_provider=self.config.embedding_configuration.provider,
                            embedding_model=embedding_model,
                            embedding_dimension=self.config.embedding_configuration.expected_dimension,
                        )
                    )
                search_batch = SearchIndexIngestBatch(
                    batch_ordinal=batch_ordinal,
                    records=tuple(search_records),
                )
                search_result = self.dependencies.search.ingest_batch(search_batch)

                rows_processed = manifest.checkpoint_rows_processed + len(rows)
                cumulative_offers += catalog_result.source_offers_ingested
                cumulative_identifiers += catalog_result.identifiers_ingested
                cumulative_structured += catalog_result.structured_attributes_ingested
                cumulative_points += search_result.points_ingested
                batches_completed = batch_ordinal + 1

                manifest = manifest.with_checkpoint(
                    batch_ordinal=batch_ordinal,
                    rows_processed=rows_processed,
                    catalog_source_offer_count=cumulative_offers,
                    catalog_identifier_count=cumulative_identifiers,
                    catalog_structured_attribute_count=cumulative_structured,
                    search_point_count=cumulative_points,
                ).with_state(BootstrapState.INGESTING)
                self.dependencies.catalog.write_manifest(manifest)
                _ = time.perf_counter() - started

            manifest = manifest.with_state(BootstrapState.VALIDATING)
            self.dependencies.catalog.write_manifest(manifest)

            catalog_validation = self.dependencies.catalog.validate(manifest)
            search_validation = self.dependencies.search.validate(manifest)
            checkpoint_complete = self._checkpoint_complete(manifest)
            ready_validation = evaluate_ready_gate(
                manifest=manifest,
                embedding_report=embedding_report,
                catalog_report=catalog_validation,
                search_report=search_validation,
                checkpoint_complete=checkpoint_complete,
            )

            if ready_validation.status is ValidationStatus.PASS:
                manifest = manifest.with_state(BootstrapState.READY)
            else:
                manifest = manifest.with_state(
                    BootstrapState.FAILED,
                    failure_stage="ready_gate",
                    failure_detail=ready_validation.checks[-1].detail,
                )
            self.dependencies.catalog.write_manifest(manifest)

            return BootstrapRunReport(
                final_state=manifest.state,
                manifest=manifest,
                validation=ready_validation,
                embedding_probe=embedding_probe_result,
                batches_completed=batches_completed,
                rows_processed=rows_processed,
                failure_stage=manifest.failure_stage,
                failure_detail=manifest.failure_detail,
            )
        except VpiBootstrapError as exc:
            if manifest is not None:
                failed = manifest.with_state(
                    BootstrapState.FAILED,
                    failure_stage=exc.__class__.__name__,
                    failure_detail=str(exc),
                )
                try:
                    self.dependencies.catalog.write_manifest(failed)
                except Exception:
                    pass
                manifest = failed
            return BootstrapRunReport(
                final_state=BootstrapState.FAILED,
                manifest=manifest,
                validation=None,
                embedding_probe=embedding_probe_result,
                batches_completed=batches_completed,
                rows_processed=rows_processed,
                failure_stage=exc.__class__.__name__,
                failure_detail=str(exc),
            )

    def _expected_compatibility_identity(
        self,
        dataset_identity: object,
    ) -> BootstrapCompatibilityIdentity:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.identity import (
            DatasetIdentity,
        )

        if not isinstance(dataset_identity, DatasetIdentity):
            raise VpiBootstrapError("internal dataset identity type mismatch")
        embedding = self.config.embedding_configuration
        model = embedding.model
        if model is None:
            raise VpiBootstrapProviderError("embedding model is required")
        return BootstrapCompatibilityIdentity(
            dataset_checksum=dataset_identity.dataset_checksum,
            dataset_record_count=dataset_identity.dataset_record_count,
            search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
            embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
            embedding_provider=embedding.provider,
            embedding_model=model,
            embedding_dimension=embedding.expected_dimension,
            catalog_schema_version=self.config.catalog_schema_version,
            search_index_schema_version=self.config.search_index_schema_version,
            bootstrap_implementation_version=self.config.bootstrap_implementation_version,
            catalog_id=self.config.catalog_id,
        )

    def _initial_manifest(
        self,
        dataset_identity: object,
        expected: BootstrapCompatibilityIdentity,
    ) -> VpiBootstrapManifest:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.identity import (
            DatasetIdentity,
        )

        if not isinstance(dataset_identity, DatasetIdentity):
            raise VpiBootstrapError("internal dataset identity type mismatch")
        return VpiBootstrapManifest(
            state=BootstrapState.INITIALIZING,
            dataset_path=dataset_identity.dataset_path,
            dataset_checksum=expected.dataset_checksum,
            dataset_record_count=expected.dataset_record_count,
            search_representation_derivation_version=expected.search_representation_derivation_version,
            embedding_configuration_version=expected.embedding_configuration_version,
            embedding_provider=expected.embedding_provider,
            embedding_model=expected.embedding_model,
            embedding_dimension=expected.embedding_dimension,
            catalog_schema_version=expected.catalog_schema_version,
            search_index_schema_version=expected.search_index_schema_version,
            bootstrap_implementation_version=expected.bootstrap_implementation_version,
            catalog_id=expected.catalog_id,
            source_revision=self.config.source_revision,
            checkpoint_batch_ordinal=None,
            checkpoint_rows_processed=0,
            target_max_records=self.config.max_records,
            catalog_source_offer_count=0,
            catalog_identifier_count=0,
            catalog_structured_attribute_count=0,
            search_point_count=0,
        )

    def _checkpoint_complete(self, manifest: VpiBootstrapManifest) -> bool:
        target = manifest.target_max_records
        if target is None:
            return manifest.checkpoint_rows_processed >= manifest.dataset_record_count
        return manifest.checkpoint_rows_processed >= target

    def _validate_all(self, manifest: VpiBootstrapManifest) -> ValidationReport:
        catalog_validation = self.dependencies.catalog.validate(manifest)
        search_validation = self.dependencies.search.validate(manifest)
        return evaluate_ready_gate(
            manifest=manifest,
            embedding_report=ValidationReport.from_checks(()),
            catalog_report=catalog_validation,
            search_report=search_validation,
            checkpoint_complete=self._checkpoint_complete(manifest),
        )
