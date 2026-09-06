"""Parquet encode/decode for embedding data pack records."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    source_ref_from_columns,
)


def _vector_type(embedding_dimension: int) -> pa.DataType:
    return pa.list_(pa.float32(), embedding_dimension)


def write_embedding_parquet(
    path: Path,
    records: Sequence[EmbeddingDataPackRecord],
    *,
    embedding_dimension: int,
) -> None:
    if not records:
        raise VpiDataPackIntegrityError("cannot write empty embedding shard")
    vector_type = _vector_type(embedding_dimension)
    table = pa.table(
        {
            "logical_point_id": pa.array(
                [record.logical_point_id for record in records],
                type=pa.string(),
            ),
            "catalog_id": pa.array(
                [record.source_ref.catalog_id for record in records],
                type=pa.string(),
            ),
            "offer_id": pa.array(
                [record.source_ref.offer_id.value for record in records],
                type=pa.string(),
            ),
            "source_revision": pa.array(
                [record.source_ref.source_revision for record in records],
                type=pa.string(),
            ),
            "derivation_version": pa.array(
                [record.derivation_version for record in records],
                type=pa.string(),
            ),
            "semantic_text_hash": pa.array(
                [record.semantic_text_hash for record in records],
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
            "embedding_model_revision": pa.array(
                [record.embedding_model_revision for record in records],
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
        raise VpiDataPackIntegrityError(f"failed to write embedding parquet: {path}") from exc


def read_embedding_parquet(
    path: Path,
    *,
    expected_dimension: int,
) -> tuple[EmbeddingDataPackRecord, ...]:
    try:
        table = pq.read_table(path)
    except OSError as exc:
        raise VpiDataPackIntegrityError(f"failed to read embedding parquet: {path}") from exc
    required = (
        "logical_point_id",
        "catalog_id",
        "offer_id",
        "source_revision",
        "derivation_version",
        "semantic_text_hash",
        "embedding_provider",
        "embedding_model",
        "embedding_model_revision",
        "embedding_dimension",
        "dense_embedding",
    )
    for column_name in required:
        if column_name not in table.column_names:
            raise VpiDataPackIntegrityError(f"embedding shard missing column: {column_name}")

    records: list[EmbeddingDataPackRecord] = []
    for row_index in range(table.num_rows):
        dimension_value = table.column("embedding_dimension")[row_index].as_py()
        if dimension_value != expected_dimension:
            raise VpiDataPackIntegrityError(
                f"row {row_index} embedding_dimension {dimension_value} "
                f"!= expected {expected_dimension}"
            )
        vector_raw = table.column("dense_embedding")[row_index].as_py()
        if not isinstance(vector_raw, list):
            raise VpiDataPackIntegrityError(f"row {row_index} dense_embedding is not a list")
        dense_embedding = tuple(float(value) for value in vector_raw)
        source_revision_raw = table.column("source_revision")[row_index].as_py()
        source_revision = str(source_revision_raw) if source_revision_raw is not None else None
        model_revision_raw = table.column("embedding_model_revision")[row_index].as_py()
        model_revision = str(model_revision_raw) if model_revision_raw is not None else None
        records.append(
            EmbeddingDataPackRecord(
                logical_point_id=str(table.column("logical_point_id")[row_index].as_py()),
                source_ref=source_ref_from_columns(
                    catalog_id=str(table.column("catalog_id")[row_index].as_py()),
                    offer_id=str(table.column("offer_id")[row_index].as_py()),
                    source_revision=source_revision,
                ),
                derivation_version=str(table.column("derivation_version")[row_index].as_py()),
                semantic_text_hash=str(table.column("semantic_text_hash")[row_index].as_py()),
                embedding_provider=str(table.column("embedding_provider")[row_index].as_py()),
                embedding_model=str(table.column("embedding_model")[row_index].as_py()),
                embedding_model_revision=model_revision,
                embedding_dimension=expected_dimension,
                dense_embedding=dense_embedding,
            )
        )
    return tuple(records)
