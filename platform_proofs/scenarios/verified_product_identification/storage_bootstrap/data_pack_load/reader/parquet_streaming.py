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
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.schema import (
    embedding_parquet_field_types,
    embedding_parquet_schema,
    relational_parquet_field_types,
    relational_parquet_schema,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.errors import (
    DataPackReaderIntegrityError,
    DataPackReaderSchemaError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.parquet_row_decode import (
    ParquetScalar,
    extract_parquet_scalar,
    require_bool,
    require_float_vector,
    require_int,
    require_optional_string,
    require_string,
)

_DEFAULT_PARQUET_BATCH_SIZE = 256


def _open_parquet_file(path: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(path)
    except OSError as exc:
        raise DataPackReaderIntegrityError(f"failed to open parquet file: {path}") from exc
    except pa.ArrowException as exc:
        raise DataPackReaderIntegrityError(f"failed to read parquet metadata: {path}") from exc


def _validate_arrow_schema(
    schema: pa.Schema,
    *,
    expected_field_types: dict[str, pa.DataType],
    shard_path: Path,
) -> None:
    for column_name, expected_type in expected_field_types.items():
        if column_name not in schema.names:
            raise DataPackReaderSchemaError(
                f"{shard_path}: missing required column {column_name}"
            )
        actual_type = schema.field(column_name).type
        if not actual_type.equals(expected_type):
            raise DataPackReaderSchemaError(
                f"{shard_path}: column {column_name}: expected Arrow type "
                f"{expected_type}, got {actual_type}"
            )


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


def _cell_scalar(
    batch: pa.RecordBatch,
    column: str,
    batch_row_index: int,
    *,
    shard_path: Path,
    shard_row_index: int,
) -> ParquetScalar:
    return extract_parquet_scalar(
        batch.column(column)[batch_row_index],
        shard_path=shard_path,
        row_index=shard_row_index,
        column=column,
    )


def _decode_relational_row(
    batch: pa.RecordBatch,
    batch_row_index: int,
    *,
    shard_path: Path,
    shard_row_index: int,
) -> RelationalDataPackRecord:
    source_revision = require_optional_string(
        _cell_scalar(
            batch,
            "source_revision",
            batch_row_index,
            shard_path=shard_path,
            shard_row_index=shard_row_index,
        ),
        shard_path=shard_path,
        row_index=shard_row_index,
        column="source_revision",
    )
    return RelationalDataPackRecord(
        global_row_index=require_int(
            _cell_scalar(
                batch,
                "global_row_index",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="global_row_index",
        ),
        source_ref=source_ref_from_columns(
            catalog_id=require_string(
                _cell_scalar(
                    batch,
                    "catalog_id",
                    batch_row_index,
                    shard_path=shard_path,
                    shard_row_index=shard_row_index,
                ),
                shard_path=shard_path,
                row_index=shard_row_index,
                column="catalog_id",
            ),
            offer_id=require_string(
                _cell_scalar(
                    batch,
                    "offer_id",
                    batch_row_index,
                    shard_path=shard_path,
                    shard_row_index=shard_row_index,
                ),
                shard_path=shard_path,
                row_index=shard_row_index,
                column="offer_id",
            ),
            source_revision=source_revision,
        ),
        record_json=require_string(
            _cell_scalar(
                batch,
                "record_json",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="record_json",
        ),
        derivation_version=require_string(
            _cell_scalar(
                batch,
                "derivation_version",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="derivation_version",
        ),
        semantic_text=require_string(
            _cell_scalar(
                batch,
                "semantic_text",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="semantic_text",
        ),
        semantic_text_hash=require_string(
            _cell_scalar(
                batch,
                "semantic_text_hash",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="semantic_text_hash",
        ),
        title=require_optional_string(
            _cell_scalar(
                batch,
                "title",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="title",
        ),
        brand=require_optional_string(
            _cell_scalar(
                batch,
                "brand",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="brand",
        ),
        category=require_optional_string(
            _cell_scalar(
                batch,
                "category",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="category",
        ),
        description=require_optional_string(
            _cell_scalar(
                batch,
                "description",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="description",
        ),
        has_identifiers=require_bool(
            _cell_scalar(
                batch,
                "has_identifiers",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="has_identifiers",
        ),
        has_spec_table=require_bool(
            _cell_scalar(
                batch,
                "has_spec_table",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="has_spec_table",
        ),
        has_structured_attributes=require_bool(
            _cell_scalar(
                batch,
                "has_structured_attributes",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="has_structured_attributes",
        ),
    )


def _decode_embedding_row(
    batch: pa.RecordBatch,
    batch_row_index: int,
    *,
    expected_dimension: int,
    shard_path: Path,
    shard_row_index: int,
) -> EmbeddingDataPackRecord:
    embedding_dimension = require_int(
        _cell_scalar(
            batch,
            "embedding_dimension",
            batch_row_index,
            shard_path=shard_path,
            shard_row_index=shard_row_index,
        ),
        shard_path=shard_path,
        row_index=shard_row_index,
        column="embedding_dimension",
        minimum=1,
    )
    if embedding_dimension != expected_dimension:
        raise DataPackReaderSchemaError(
            f"{shard_path}: row {shard_row_index} column embedding_dimension: "
            f"expected {expected_dimension}, got {embedding_dimension}"
        )
    dense_embedding = require_float_vector(
        _cell_scalar(
            batch,
            "dense_embedding",
            batch_row_index,
            shard_path=shard_path,
            shard_row_index=shard_row_index,
        ),
        shard_path=shard_path,
        row_index=shard_row_index,
        column="dense_embedding",
        expected_length=expected_dimension,
    )
    source_revision = require_optional_string(
        _cell_scalar(
            batch,
            "source_revision",
            batch_row_index,
            shard_path=shard_path,
            shard_row_index=shard_row_index,
        ),
        shard_path=shard_path,
        row_index=shard_row_index,
        column="source_revision",
    )
    model_revision = require_optional_string(
        _cell_scalar(
            batch,
            "embedding_model_revision",
            batch_row_index,
            shard_path=shard_path,
            shard_row_index=shard_row_index,
        ),
        shard_path=shard_path,
        row_index=shard_row_index,
        column="embedding_model_revision",
    )
    return EmbeddingDataPackRecord(
        logical_point_id=require_string(
            _cell_scalar(
                batch,
                "logical_point_id",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="logical_point_id",
        ),
        source_ref=source_ref_from_columns(
            catalog_id=require_string(
                _cell_scalar(
                    batch,
                    "catalog_id",
                    batch_row_index,
                    shard_path=shard_path,
                    shard_row_index=shard_row_index,
                ),
                shard_path=shard_path,
                row_index=shard_row_index,
                column="catalog_id",
            ),
            offer_id=require_string(
                _cell_scalar(
                    batch,
                    "offer_id",
                    batch_row_index,
                    shard_path=shard_path,
                    shard_row_index=shard_row_index,
                ),
                shard_path=shard_path,
                row_index=shard_row_index,
                column="offer_id",
            ),
            source_revision=source_revision,
        ),
        derivation_version=require_string(
            _cell_scalar(
                batch,
                "derivation_version",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="derivation_version",
        ),
        semantic_text_hash=require_string(
            _cell_scalar(
                batch,
                "semantic_text_hash",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="semantic_text_hash",
        ),
        embedding_provider=require_string(
            _cell_scalar(
                batch,
                "embedding_provider",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="embedding_provider",
        ),
        embedding_model=require_string(
            _cell_scalar(
                batch,
                "embedding_model",
                batch_row_index,
                shard_path=shard_path,
                shard_row_index=shard_row_index,
            ),
            shard_path=shard_path,
            row_index=shard_row_index,
            column="embedding_model",
        ),
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
    _validate_arrow_schema(
        parquet_file.schema_arrow,
        expected_field_types=relational_parquet_field_types(),
        shard_path=path,
    )
    required_columns = tuple(relational_parquet_schema().names)
    shard_row_offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        _validate_columns(batch, required_columns=required_columns, shard_path=path)
        for batch_row_index in range(batch.num_rows):
            yield _decode_relational_row(
                batch,
                batch_row_index,
                shard_path=path,
                shard_row_index=shard_row_offset + batch_row_index,
            )
        shard_row_offset += batch.num_rows


def iter_embedding_records(
    path: Path,
    *,
    expected_dimension: int,
    batch_size: int = _DEFAULT_PARQUET_BATCH_SIZE,
) -> Iterator[EmbeddingDataPackRecord]:
    parquet_file = _open_parquet_file(path)
    _validate_arrow_schema(
        parquet_file.schema_arrow,
        expected_field_types=embedding_parquet_field_types(expected_dimension),
        shard_path=path,
    )
    required_columns = tuple(embedding_parquet_schema(expected_dimension).names)
    shard_row_offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        _validate_columns(batch, required_columns=required_columns, shard_path=path)
        for batch_row_index in range(batch.num_rows):
            yield _decode_embedding_row(
                batch,
                batch_row_index,
                expected_dimension=expected_dimension,
                shard_path=path,
                shard_row_index=shard_row_offset + batch_row_index,
            )
        shard_row_offset += batch.num_rows
