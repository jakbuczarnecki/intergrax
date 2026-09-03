"""Parquet encode/decode for embedding artifact records."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactIntegrityError,
    ArtifactWriteError,
)


def _vector_type(embedding_dimension: int) -> pa.DataType:
    return pa.list_(pa.float32(), embedding_dimension)


def write_records_parquet(
    path: Path,
    records: Sequence[EmbeddingArtifactRecord],
    *,
    embedding_dimension: int,
) -> None:
    if not records:
        raise ArtifactWriteError("cannot write empty shard")
    vector_type = _vector_type(embedding_dimension)
    table = pa.table(
        {
            "global_row_index": pa.array(
                [record.global_row_index for record in records],
                type=pa.int64(),
            ),
            "logical_point_id": pa.array(
                [record.logical_point_id for record in records],
                type=pa.string(),
            ),
            "catalog_id": pa.array(
                [record.catalog_id for record in records],
                type=pa.string(),
            ),
            "offer_id": pa.array(
                [record.offer_id for record in records],
                type=pa.string(),
            ),
            "source_revision": pa.array(
                [record.source_revision for record in records],
                type=pa.string(),
            ),
            "derivation_version": pa.array(
                [record.derivation_version for record in records],
                type=pa.string(),
            ),
            "semantic_text": pa.array(
                [record.semantic_text for record in records],
                type=pa.string(),
            ),
            "lexical_text": pa.array(
                [record.lexical_text for record in records],
                type=pa.string(),
            ),
            "embedding_provider": pa.array(
                [record.embedding_provider for record in records],
                type=pa.string(),
            ),
            "embedding_model": pa.array(
                [record.embedding_model for record in records],
                type=pa.string(),
            ),
            "embedding_dimension": pa.array(
                [record.embedding_dimension for record in records],
                type=pa.int32(),
            ),
            "dense_embedding": pa.array(
                [list(record.dense_embedding) for record in records],
                type=vector_type,
            ),
        }
    )
    try:
        pq.write_table(table, path)
    except OSError as exc:
        raise ArtifactWriteError(f"failed to write parquet shard: {path}") from exc


def read_records_parquet(
    path: Path,
    *,
    expected_dimension: int,
) -> tuple[EmbeddingArtifactRecord, ...]:
    try:
        table = pq.read_table(path)
    except OSError as exc:
        raise ArtifactIntegrityError(f"failed to read parquet shard: {path}") from exc
    required_columns = (
        "global_row_index",
        "logical_point_id",
        "catalog_id",
        "offer_id",
        "source_revision",
        "derivation_version",
        "semantic_text",
        "lexical_text",
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "dense_embedding",
    )
    for column_name in required_columns:
        if column_name not in table.column_names:
            raise ArtifactIntegrityError(f"shard missing required column: {column_name}")

    records: list[EmbeddingArtifactRecord] = []
    for row_index in range(table.num_rows):
        dimension_value = table.column("embedding_dimension")[row_index].as_py()
        if dimension_value != expected_dimension:
            raise ArtifactIntegrityError(
                f"row {row_index} embedding_dimension {dimension_value} != expected {expected_dimension}"
            )
        vector_raw = table.column("dense_embedding")[row_index].as_py()
        if not isinstance(vector_raw, list):
            raise ArtifactIntegrityError(f"row {row_index} dense_embedding is not a list")
        dense_embedding = tuple(float(value) for value in vector_raw)
        source_revision_raw = table.column("source_revision")[row_index].as_py()
        records.append(
            EmbeddingArtifactRecord(
                global_row_index=int(table.column("global_row_index")[row_index].as_py()),
                logical_point_id=str(table.column("logical_point_id")[row_index].as_py()),
                catalog_id=str(table.column("catalog_id")[row_index].as_py()),
                offer_id=str(table.column("offer_id")[row_index].as_py()),
                source_revision=(
                    str(source_revision_raw) if source_revision_raw is not None else None
                ),
                derivation_version=str(table.column("derivation_version")[row_index].as_py()),
                semantic_text=str(table.column("semantic_text")[row_index].as_py()),
                lexical_text=str(table.column("lexical_text")[row_index].as_py()),
                embedding_provider=str(table.column("embedding_provider")[row_index].as_py()),
                embedding_model=str(table.column("embedding_model")[row_index].as_py()),
                embedding_dimension=expected_dimension,
                dense_embedding=dense_embedding,
            )
        )
    return tuple(records)
