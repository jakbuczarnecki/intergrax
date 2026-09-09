"""Canonical validation expectations for VPI Data Pack v1."""

from __future__ import annotations

import math
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.data_package.identity import (
    CANONICAL_DATASET_CHECKSUM,
    CANONICAL_SELECTED_RECORD_COUNT,
    CANONICAL_SOURCE_DATASET_NAME,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    VPI_CANONICAL_EMBEDDING_DIMENSION,
    VPI_CANONICAL_EMBEDDING_MODEL,
    VPI_CANONICAL_EMBEDDING_PROVIDER,
    VPI_CANONICAL_EMBEDDING_REVISION,
)


@dataclass(frozen=True, slots=True)
class DataPackValidationExpectations:
    record_count: int
    shard_size: int
    shard_count: int
    data_pack_version: str
    derivation_version: str
    semantic_text_version: str
    relational_schema_version: str
    embedding_schema_version: str
    embedding_provider: str
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int
    embedding_configuration_version: str
    input_policy_version: str
    source_dataset_name: str
    source_dataset_sha256: str
    require_proof_report: bool

    def expected_shard_record_count(self, ordinal: int) -> int:
        if ordinal < 1 or ordinal > self.shard_count:
            msg = f"ordinal out of range: {ordinal}"
            raise ValueError(msg)
        if ordinal < self.shard_count:
            return self.shard_size
        return self.record_count - (self.shard_count - 1) * self.shard_size

    def expected_global_row_range(self, ordinal: int) -> tuple[int, int]:
        start = (ordinal - 1) * self.shard_size
        end_exclusive = min(ordinal * self.shard_size, self.record_count)
        return start, end_exclusive


def canonical_v1_validation_expectations() -> DataPackValidationExpectations:
    record_count = CANONICAL_SELECTED_RECORD_COUNT
    shard_size = 1_000
    shard_count = math.ceil(record_count / shard_size)
    return DataPackValidationExpectations(
        record_count=record_count,
        shard_size=shard_size,
        shard_count=shard_count,
        data_pack_version=DATA_PACK_VERSION,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
        embedding_provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
        embedding_model=VPI_CANONICAL_EMBEDDING_MODEL,
        embedding_model_revision=VPI_CANONICAL_EMBEDDING_REVISION,
        embedding_dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        input_policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        source_dataset_name=CANONICAL_SOURCE_DATASET_NAME,
        source_dataset_sha256=CANONICAL_DATASET_CHECKSUM,
        require_proof_report=True,
    )


def fixture_validation_expectations(
    *,
    record_count: int,
    shard_size: int,
    require_proof_report: bool = True,
) -> DataPackValidationExpectations:
    shard_count = math.ceil(record_count / shard_size)
    return DataPackValidationExpectations(
        record_count=record_count,
        shard_size=shard_size,
        shard_count=shard_count,
        data_pack_version=DATA_PACK_VERSION,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
        embedding_provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
        embedding_model=VPI_CANONICAL_EMBEDDING_MODEL,
        embedding_model_revision=VPI_CANONICAL_EMBEDDING_REVISION,
        embedding_dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        input_policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        source_dataset_name=CANONICAL_SOURCE_DATASET_NAME,
        source_dataset_sha256="a" * 64,
        require_proof_report=require_proof_report,
    )
