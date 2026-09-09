"""Bounded-memory Parquet row streaming for Data Pack bootstrap."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    source_ref_from_columns,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.errors import (
    DataPackReaderIntegrityError,
    DataPackReaderSchemaError,
)

_RELATIONAL_REQUIRED_COLUMNS = (
    "global_row_index",
    "catalog_id",
    "offer_id",
    "source_revision",
    "record_json",
    "derivation_version",
    "semantic_text",
    "semantic_text_hash",
    "title",
    "brand",
    "category",
    "description",
    "has_identifiers",
    "has_spec_table",
    "has_structured_attributes",
)

_EMBEDDING_REQUIRED_COLUMNS = (
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

_DEFAULT_PARQUET_BATCH_SIZE = 256


def _open_parquet_file(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(path)
    except OSError as exc:
        raise DataPackReaderIntegrityError(f"failed to open parquet file: {path}") from exc
    except pa.ArrowException as exc:
        raise DataPackReaderIntegrityError(f"failed to read parquet metadata: {path}") from exc


def _validate_columns(
    batch: pa.RecordBatch,
    *,
    required_columns: tuple[str, ...],
    shard_path: Path,
) -> None:
    for column_name in required_columns:
        if column_name not in batch.schema.names:
            raise DataPackReaderSchemaError(
                f"{shard_path}: missing required column {column_name}"
            )


def _decode_relational_row(batch: pa.RecordBatch, row_index: int) -> RelationalDataPackRecord:
    source_revision_raw = batch.column("source_revision")[row_index].as_py()
    source_revision = str(source_revision_raw) if source_revision_raw is not None else None
    return RelationalDataPackRecord(
        global_row_index=int(batch.column("global_row_index")[row_index].as_py()),
        source_ref=source_ref_from_columns(
            catalog_id=str(batch.column("catalog_id")[row_index].as_py()),
            offer_id=str(batch.column("offer_id")[row_index].as_py()),
            source_revision=source_revision,
        ),
        record_json=str(batch.column("record_json")[row_index].as_py()),
        derivation_version=str(batch.column("derivation_version")[row_index].as_py()),
        semantic_text=str(batch.column("semantic_text")[row_index].as_py()),
        semantic_text_hash=str(batch.column("semantic_text_hash")[row_index].as_py()),
        title=batch.column("title")[row_index].as_py(),
        brand=batch.column("brand")[row_index].as_py(),
        category=batch.column("category")[row_index].as_py(),
        description=batch.column("description")[row_index].as_py(),
        has_identifiers=bool(batch.column("has_identifiers")[row_index].as_py()),
        has_spec_table=bool(batch.column("has_spec_table")[row_index].as_py()),
        has_structured_attributes=bool(
            batch.column("has_structured_attributes")[row_index].as_py()
        ),
    )


def _decode_embedding_row(
    batch: pa.RecordBatch,
    row_index: int,
    *,
    expected_dimension: int,
    shard_path: Path,
) -> EmbeddingDataPackRecord:
    dimension_value = batch.column("embedding_dimension")[row_index].as_py()
    if dimension_value != expected_dimension:
        raise DataPackReaderSchemaError(
            f"{shard_path}: row {row_index} embedding_dimension {dimension_value} "
            f"!= expected {expected_dimension}"
        )
    vector_raw = batch.column("dense_embedding")[row_index].as_py()
    if not isinstance(vector_raw, list):
        raise DataPackReaderSchemaError(
            f"{shard_path}: row {row_index} dense_embedding is not a list"
        )
    dense_embedding = tuple(float(value) for value in vector_raw)
    source_revision_raw = batch.column("source_revision")[row_index].as_py()
    source_revision = str(source_revision_raw) if source_revision_raw is not None else None
    model_revision_raw = batch.column("embedding_model_revision")[row_index].as_py()
    model_revision = str(model_revision_raw) if model_revision_raw is not None else None
    return EmbeddingDataPackRecord(
        logical_point_id=str(batch.column("logical_point_id")[row_index].as_py()),
        source_ref=source_ref_from_columns(
            catalog_id=str(batch.column("catalog_id")[row_index].as_py()),
            offer_id=str(batch.column("offer_id")[row_index].as_py()),
            source_revision=source_revision,
        ),
        derivation_version=str(batch.column("derivation_version")[row_index].as_py()),
        semantic_text_hash=str(batch.column("semantic_text_hash")[row_index].as_py()),
        embedding_provider=str(batch.column("embedding_provider")[row_index].as_py()),
        embedding_model=str(batch.column("embedding_model")[row_index].as_py()),
        embedding_model_revision=model_revision,
        embedding_dimension=expected_dimension,
        dense_embedding=dense_embedding,
    )


def iter_relational_records(
    path: Path,
    *,
    batch_size: int = _DEFAULT_PARQUET_BATCH_SIZE,
) -> Iterator[RelationalDataPackRecord]:
    parquet_file = _open_parquet_file(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        _validate_columns(batch, required_columns=_RELATIONAL_REQUIRED_COLUMNS, shard_path=path)
        for row_index in range(batch.num_rows):
            yield _decode_relational_row(batch, row_index)


def iter_embedding_records(
    path: Path,
    *,
    expected_dimension: int,
    batch_size: int = _DEFAULT_PARQUET_BATCH_SIZE,
) -> Iterator[EmbeddingDataPackRecord]:
    parquet_file = _open_parquet_file(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        _validate_columns(batch, required_columns=_EMBEDDING_REQUIRED_COLUMNS, shard_path=path)
        for row_index in range(batch.num_rows):
            yield _decode_embedding_row(
                batch,
                row_index,
                expected_dimension=expected_dimension,
                shard_path=path,
            )
