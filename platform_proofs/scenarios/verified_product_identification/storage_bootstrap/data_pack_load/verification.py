"""Batch verification helpers for cross-store consistency."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalBatch,
    StorageLoadBatchResult,
    VectorBatch,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    BootstrapFailure,
    BootstrapFailureCategory,
    StorageBootstrapIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.mapping import (
    identity_key,
)


def assert_batch_identity_parity(batch: RelationalBatch, vector_batch: VectorBatch) -> None:
    if len(batch.records) != len(vector_batch.records):
        raise StorageBootstrapIntegrityError(
            f"batch {batch.batch_number} record count mismatch: "
            f"relational={len(batch.records)} vector={len(vector_batch.records)}"
        )
    for relational_record, vector_record in zip(batch.records, vector_batch.records, strict=True):
        if relational_record.source_ref != vector_record.source_ref:
            raise StorageBootstrapIntegrityError(
                BootstrapFailure(
                    category=BootstrapFailureCategory.IDENTITY_MISMATCH,
                    detail="relational and vector source_ref differ within batch",
                    batch_number=batch.batch_number,
                    first_failed_identity=identity_key(relational_record.source_ref),
                ).detail
            )


def assert_verification_complete(
    *,
    batch_number: int,
    relational_result: StorageLoadBatchResult,
    vector_result: StorageLoadBatchResult,
) -> None:
    if not relational_result.is_complete_success:
        raise StorageBootstrapIntegrityError(
            BootstrapFailure(
                category=BootstrapFailureCategory.INTEGRITY_FAILED,
                detail="relational verification incomplete",
                batch_number=batch_number,
                first_failed_identity=relational_result.first_failed_identity,
            ).detail
        )
    if not vector_result.is_complete_success:
        raise StorageBootstrapIntegrityError(
            BootstrapFailure(
                category=BootstrapFailureCategory.INTEGRITY_FAILED,
                detail="vector verification incomplete",
                batch_number=batch_number,
                first_failed_identity=vector_result.first_failed_identity,
            ).detail
        )
    if relational_result.successful_count != vector_result.successful_count:
        raise StorageBootstrapIntegrityError(
            BootstrapFailure(
                category=BootstrapFailureCategory.INTEGRITY_FAILED,
                detail=(
                    f"orphan side detected: relational={relational_result.successful_count} "
                    f"vector={vector_result.successful_count}"
                ),
                batch_number=batch_number,
            ).detail
        )
