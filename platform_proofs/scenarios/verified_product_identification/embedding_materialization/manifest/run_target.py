"""Run-target semantics for embedding artifact materialization."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactCompatibilityError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
)


def resolve_requested_target_rows(
    *,
    max_records: int | None,
    dataset_record_count: int,
) -> int:
    if max_records is None:
        return dataset_record_count
    return min(max_records, dataset_record_count)


def checkpoint_meets_target(
    *,
    checkpoint_rows_materialized: int,
    requested_target_rows: int,
) -> bool:
    return checkpoint_rows_materialized >= requested_target_rows


def assert_requested_target_not_below_checkpoint(
    *,
    requested_target_rows: int,
    checkpoint_rows_materialized: int,
) -> None:
    if checkpoint_rows_materialized > requested_target_rows:
        raise ArtifactCompatibilityError(
            "requested materialization target "
            f"({requested_target_rows}) is below existing checkpoint "
            f"({checkpoint_rows_materialized}); cannot shrink scope without explicit rebuild"
        )


def manifest_run_target_value(requested_target_rows: int, dataset_record_count: int) -> int | None:
    if requested_target_rows >= dataset_record_count:
        return None
    return requested_target_rows


def effective_run_target_rows(manifest: EmbeddingArtifactManifest) -> int:
    if manifest.target_max_records is None:
        return manifest.dataset_record_count
    return manifest.target_max_records
