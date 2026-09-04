"""Orphan shard reconciliation after crash between shard commit and manifest checkpoint."""

from __future__ import annotations

from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
    EmbeddingArtifactShardDescriptor,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.paths import (
    shard_file_name,
    shard_path,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.parquet_codec import (
    read_records_parquet,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.shard_validation import (
    build_shard_descriptor,
    validate_record_row_alignment,
    validate_shard_descriptor_continuity,
)


def reconcile_orphan_shards(
    *,
    artifact_dir: Path,
    manifest: EmbeddingArtifactManifest,
) -> EmbeddingArtifactManifest:
    committed = list(manifest.committed_shards)
    validate_shard_descriptor_continuity(committed)
    next_ordinal = len(committed)
    next_row_index = manifest.checkpoint_rows_materialized
    orphan_path = shard_path(artifact_dir, next_ordinal)
    if not orphan_path.is_file():
        return manifest

    records = read_records_parquet(
        orphan_path,
        expected_dimension=manifest.embedding_dimension,
    )
    validate_record_row_alignment(records)
    if records[0].global_row_index != next_row_index:
        raise ArtifactIntegrityError(
            f"orphan shard {orphan_path.name} starts at row {records[0].global_row_index}, "
            f"expected {next_row_index}"
        )

    descriptor = build_shard_descriptor(
        shard_ordinal=next_ordinal,
        file_name=shard_file_name(next_ordinal),
        records=records,
        file_path=orphan_path,
    )
    adopted = committed + [descriptor]
    validate_shard_descriptor_continuity(adopted)
    rows_materialized = descriptor.last_global_row_index + 1
    return manifest.with_checkpoint(
        shard_ordinal=descriptor.shard_ordinal,
        rows_materialized=rows_materialized,
        committed_shards=tuple(adopted),
    )
