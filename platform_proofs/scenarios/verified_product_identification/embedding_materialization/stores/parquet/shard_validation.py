"""Shard integrity validation helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactShardDescriptor,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.parquet_codec import (
    read_records_parquet,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_record_row_alignment(records: Sequence[EmbeddingArtifactRecord]) -> None:
    if not records:
        raise ArtifactIntegrityError("shard cannot be empty")
    expected_first = records[0].global_row_index
    for offset, record in enumerate(records):
        expected_row = expected_first + offset
        if record.global_row_index != expected_row:
            raise ArtifactIntegrityError(
                f"row gap or overlap at offset {offset}: expected global_row_index "
                f"{expected_row}, got {record.global_row_index}"
            )


def build_shard_descriptor(
    *,
    shard_ordinal: int,
    file_name: str,
    records: Sequence[EmbeddingArtifactRecord],
    file_path: Path,
) -> EmbeddingArtifactShardDescriptor:
    validate_record_row_alignment(records)
    return EmbeddingArtifactShardDescriptor(
        shard_ordinal=shard_ordinal,
        file_name=file_name,
        first_global_row_index=records[0].global_row_index,
        last_global_row_index=records[-1].global_row_index,
        record_count=len(records),
        sha256_checksum=sha256_file(file_path),
    )


def validate_shard_descriptor_continuity(
    committed_shards: Sequence[EmbeddingArtifactShardDescriptor],
) -> None:
    if not committed_shards:
        return
    ordered = sorted(committed_shards, key=lambda descriptor: descriptor.shard_ordinal)
    for index, descriptor in enumerate(ordered):
        if descriptor.shard_ordinal != index:
            raise ArtifactIntegrityError(
                f"shard ordinal gap: expected {index}, got {descriptor.shard_ordinal}"
            )
        if index == 0:
            continue
        previous = ordered[index - 1]
        expected_first = previous.last_global_row_index + 1
        if descriptor.first_global_row_index != expected_first:
            raise ArtifactIntegrityError(
                "shard row overlap or gap between "
                f"{previous.shard_ordinal} and {descriptor.shard_ordinal}"
            )


def validate_shard_file(
    *,
    descriptor: EmbeddingArtifactShardDescriptor,
    file_path: Path,
    expected_dimension: int,
) -> None:
    if not file_path.is_file():
        raise ArtifactIntegrityError(f"shard file missing: {file_path}")
    checksum = sha256_file(file_path)
    if checksum != descriptor.sha256_checksum:
        raise ArtifactIntegrityError(
            f"shard checksum mismatch for {descriptor.file_name}: expected "
            f"{descriptor.sha256_checksum}, got {checksum}"
        )
    records = read_records_parquet(file_path, expected_dimension=expected_dimension)
    if len(records) != descriptor.record_count:
        raise ArtifactIntegrityError(
            f"shard {descriptor.file_name} record count mismatch: "
            f"descriptor={descriptor.record_count}, file={len(records)}"
        )
    if records[0].global_row_index != descriptor.first_global_row_index:
        raise ArtifactIntegrityError(
            f"shard {descriptor.file_name} first row mismatch"
        )
    if records[-1].global_row_index != descriptor.last_global_row_index:
        raise ArtifactIntegrityError(
            f"shard {descriptor.file_name} last row mismatch"
        )
    validate_record_row_alignment(records)
