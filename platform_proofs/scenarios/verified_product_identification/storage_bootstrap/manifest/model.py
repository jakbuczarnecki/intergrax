"""Scenario-owned bootstrap manifest contract and lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BootstrapState(str, Enum):
    INITIALIZING = "INITIALIZING"
    INGESTING = "INGESTING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"


CATALOG_SCHEMA_VERSION = "v1"
SEARCH_INDEX_SCHEMA_VERSION = "v1"
BOOTSTRAP_IMPLEMENTATION_VERSION = "vpi-bootstrap/1.0.0"


@dataclass(frozen=True, slots=True)
class VpiBootstrapManifest:
    """Source of bootstrap compatibility identity for catalog and search stores."""

    state: BootstrapState
    dataset_path: str
    dataset_checksum: str
    dataset_record_count: int
    search_representation_derivation_version: str
    embedding_configuration_version: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    catalog_schema_version: str
    search_index_schema_version: str
    bootstrap_implementation_version: str
    catalog_id: str
    source_revision: str | None
    checkpoint_batch_ordinal: int | None
    checkpoint_rows_processed: int
    target_max_records: int | None
    catalog_source_offer_count: int
    catalog_identifier_count: int
    catalog_structured_attribute_count: int
    search_point_count: int
    failure_stage: str | None = None
    failure_detail: str | None = None

    def with_state(
        self,
        state: BootstrapState,
        *,
        failure_stage: str | None = None,
        failure_detail: str | None = None,
    ) -> VpiBootstrapManifest:
        return VpiBootstrapManifest(
            state=state,
            dataset_path=self.dataset_path,
            dataset_checksum=self.dataset_checksum,
            dataset_record_count=self.dataset_record_count,
            search_representation_derivation_version=self.search_representation_derivation_version,
            embedding_configuration_version=self.embedding_configuration_version,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            catalog_schema_version=self.catalog_schema_version,
            search_index_schema_version=self.search_index_schema_version,
            bootstrap_implementation_version=self.bootstrap_implementation_version,
            catalog_id=self.catalog_id,
            source_revision=self.source_revision,
            checkpoint_batch_ordinal=self.checkpoint_batch_ordinal,
            checkpoint_rows_processed=self.checkpoint_rows_processed,
            target_max_records=self.target_max_records,
            catalog_source_offer_count=self.catalog_source_offer_count,
            catalog_identifier_count=self.catalog_identifier_count,
            catalog_structured_attribute_count=self.catalog_structured_attribute_count,
            search_point_count=self.search_point_count,
            failure_stage=failure_stage,
            failure_detail=failure_detail,
        )

    def with_run_target(self, target_max_records: int | None) -> VpiBootstrapManifest:
        return VpiBootstrapManifest(
            state=self.state,
            dataset_path=self.dataset_path,
            dataset_checksum=self.dataset_checksum,
            dataset_record_count=self.dataset_record_count,
            search_representation_derivation_version=self.search_representation_derivation_version,
            embedding_configuration_version=self.embedding_configuration_version,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            catalog_schema_version=self.catalog_schema_version,
            search_index_schema_version=self.search_index_schema_version,
            bootstrap_implementation_version=self.bootstrap_implementation_version,
            catalog_id=self.catalog_id,
            source_revision=self.source_revision,
            checkpoint_batch_ordinal=self.checkpoint_batch_ordinal,
            checkpoint_rows_processed=self.checkpoint_rows_processed,
            target_max_records=target_max_records,
            catalog_source_offer_count=self.catalog_source_offer_count,
            catalog_identifier_count=self.catalog_identifier_count,
            catalog_structured_attribute_count=self.catalog_structured_attribute_count,
            search_point_count=self.search_point_count,
            failure_stage=self.failure_stage,
            failure_detail=self.failure_detail,
        )

    def with_checkpoint(
        self,
        *,
        batch_ordinal: int,
        rows_processed: int,
        catalog_source_offer_count: int,
        catalog_identifier_count: int,
        catalog_structured_attribute_count: int,
        search_point_count: int,
    ) -> VpiBootstrapManifest:
        return VpiBootstrapManifest(
            state=self.state,
            dataset_path=self.dataset_path,
            dataset_checksum=self.dataset_checksum,
            dataset_record_count=self.dataset_record_count,
            search_representation_derivation_version=self.search_representation_derivation_version,
            embedding_configuration_version=self.embedding_configuration_version,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
            catalog_schema_version=self.catalog_schema_version,
            search_index_schema_version=self.search_index_schema_version,
            bootstrap_implementation_version=self.bootstrap_implementation_version,
            catalog_id=self.catalog_id,
            source_revision=self.source_revision,
            checkpoint_batch_ordinal=batch_ordinal,
            checkpoint_rows_processed=rows_processed,
            target_max_records=self.target_max_records,
            catalog_source_offer_count=catalog_source_offer_count,
            catalog_identifier_count=catalog_identifier_count,
            catalog_structured_attribute_count=catalog_structured_attribute_count,
            search_point_count=search_point_count,
            failure_stage=self.failure_stage,
            failure_detail=self.failure_detail,
        )
