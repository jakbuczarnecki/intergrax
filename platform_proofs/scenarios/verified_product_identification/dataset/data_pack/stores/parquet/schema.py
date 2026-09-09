"""Canonical Arrow Parquet field types for Data Pack shards."""

from __future__ import annotations

import pyarrow as pa


def relational_parquet_field_types() -> dict[str, pa.DataType]:
    """Return canonical relational shard column Arrow types."""
    return {
        "global_row_index": pa.int64(),
        "catalog_id": pa.string(),
        "offer_id": pa.string(),
        "source_revision": pa.string(),
        "record_json": pa.string(),
        "derivation_version": pa.string(),
        "semantic_text": pa.string(),
        "semantic_text_hash": pa.string(),
        "title": pa.string(),
        "brand": pa.string(),
        "category": pa.string(),
        "description": pa.string(),
        "has_identifiers": pa.bool_(),
        "has_spec_table": pa.bool_(),
        "has_structured_attributes": pa.bool_(),
    }


def relational_parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field(name, field_type)
            for name, field_type in relational_parquet_field_types().items()
        ]
    )


def embedding_vector_type(embedding_dimension: int) -> pa.DataType:
    return pa.list_(pa.float32(), embedding_dimension)


def embedding_parquet_field_types(embedding_dimension: int) -> dict[str, pa.DataType]:
    """Return canonical embedding shard column Arrow types."""
    return {
        "logical_point_id": pa.string(),
        "catalog_id": pa.string(),
        "offer_id": pa.string(),
        "source_revision": pa.string(),
        "derivation_version": pa.string(),
        "semantic_text_hash": pa.string(),
        "embedding_provider": pa.string(),
        "embedding_model": pa.string(),
        "embedding_model_revision": pa.string(),
        "embedding_dimension": pa.int32(),
        "dense_embedding": embedding_vector_type(embedding_dimension),
    }


def embedding_parquet_schema(embedding_dimension: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field(name, field_type)
            for name, field_type in embedding_parquet_field_types(embedding_dimension).items()
        ]
    )
