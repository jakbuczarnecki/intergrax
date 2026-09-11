"""Adapter-owned typed row contract for pgvector bootstrap reads."""

from __future__ import annotations

import array
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pgvector import Vector as PgVectorProviderVector

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.configuration import (
    ExpectedVectorIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.errors import (
    PgVectorBootstrapOperationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    VectorLoadRecord,
)

_PgVectorFetchedScalar = str | int | float | None
_PgVectorFetchedValue = _PgVectorFetchedScalar | Sequence[float] | PgVectorProviderVector


@dataclass(frozen=True, slots=True)
class StoredPgVectorRow:
    logical_point_id: str
    catalog_id: str
    offer_id: str
    source_revision_norm: str
    source_revision: str | None
    semantic_text_hash: str
    embedding_provider: str
    embedding_model: str
    embedding_revision: str | None
    embedding_dimension: int
    derivation_version: str
    dense_embedding: tuple[float, ...]


def source_revision_norm(source_revision: str | None) -> str:
    return source_revision or ""


def normalize_vector_float32(values: tuple[float, ...]) -> tuple[float, ...]:
    packed = array.array("f", values)
    return tuple(packed)


def vectors_transport_equal(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    tolerance: float,
) -> bool:
    if len(left) != len(right):
        return False
    left_norm = normalize_vector_float32(left)
    right_norm = normalize_vector_float32(right)
    if tolerance == 0.0:
        return left_norm == right_norm
    return all(abs(a - b) <= tolerance for a, b in zip(left_norm, right_norm, strict=True))


def embedding_identity_matches(
    record: VectorLoadRecord,
    expected: ExpectedVectorIdentity,
) -> bool:
    if record.embedding_provider != expected.provider:
        return False
    if record.embedding_model != expected.model:
        return False
    if record.embedding_revision != expected.revision:
        return False
    return record.embedding_dimension == expected.dimension


def stored_row_from_record(record: VectorLoadRecord) -> StoredPgVectorRow:
    source_ref = record.source_ref
    return StoredPgVectorRow(
        logical_point_id=record.logical_point_id,
        catalog_id=source_ref.catalog_id,
        offer_id=source_ref.offer_id.value,
        source_revision_norm=source_revision_norm(source_ref.source_revision),
        source_revision=source_ref.source_revision,
        semantic_text_hash=record.semantic_text_hash,
        embedding_provider=record.embedding_provider,
        embedding_model=record.embedding_model,
        embedding_revision=record.embedding_revision,
        embedding_dimension=record.embedding_dimension,
        derivation_version=record.derivation_version,
        dense_embedding=normalize_vector_float32(record.dense_embedding),
    )


def stored_row_identity_matches(
    stored: StoredPgVectorRow,
    expected: StoredPgVectorRow,
) -> bool:
    return (
        stored.logical_point_id == expected.logical_point_id
        and stored.catalog_id == expected.catalog_id
        and stored.offer_id == expected.offer_id
        and stored.source_revision_norm == expected.source_revision_norm
        and stored.source_revision == expected.source_revision
        and stored.semantic_text_hash == expected.semantic_text_hash
        and stored.embedding_provider == expected.embedding_provider
        and stored.embedding_model == expected.embedding_model
        and stored.embedding_revision == expected.embedding_revision
        and stored.embedding_dimension == expected.embedding_dimension
        and stored.derivation_version == expected.derivation_version
    )


def record_matches_stored(
    record: VectorLoadRecord,
    stored: StoredPgVectorRow,
    *,
    tolerance: float,
) -> bool:
    expected = stored_row_from_record(record)
    if not stored_row_identity_matches(stored, expected):
        return False
    return vectors_transport_equal(
        expected.dense_embedding,
        stored.dense_embedding,
        tolerance=tolerance,
    )


def _require_str(
    row: Mapping[str, _PgVectorFetchedValue],
    field: str,
) -> str:
    if field not in row:
        raise PgVectorBootstrapOperationError(
            f"malformed pgvector row: missing {field}"
        )
    value = row[field]
    if not isinstance(value, str):
        raise PgVectorBootstrapOperationError(
            f"malformed pgvector row: {field} must be str"
        )
    return value


def _require_int(
    row: Mapping[str, _PgVectorFetchedValue],
    field: str,
) -> int:
    if field not in row:
        raise PgVectorBootstrapOperationError(
            f"malformed pgvector row: missing {field}"
        )
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PgVectorBootstrapOperationError(
            f"malformed pgvector row: {field} must be int"
        )
    return value


def _optional_str(
    row: Mapping[str, _PgVectorFetchedValue],
    field: str,
) -> str | None:
    if field not in row:
        raise PgVectorBootstrapOperationError(
            f"malformed pgvector row: missing {field}"
        )
    value = row[field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise PgVectorBootstrapOperationError(
            f"malformed pgvector row: {field} must be str or null"
        )
    return value


def _provider_vector_components(raw: PgVectorProviderVector) -> list[float]:
    return raw.to_list()


def _embedding_values(raw: _PgVectorFetchedValue, *, expected_dimension: int) -> tuple[float, ...]:
    if raw is None:
        raise PgVectorBootstrapOperationError("malformed pgvector row: dense_embedding missing")
    if isinstance(raw, PgVectorProviderVector):
        components = _provider_vector_components(raw)
    elif isinstance(raw, (str, int, float)):
        raise PgVectorBootstrapOperationError(
            "malformed pgvector row: dense_embedding must be a vector sequence"
        )
    else:
        components = [float(value) for value in raw]
    if len(components) != expected_dimension:
        raise PgVectorBootstrapOperationError(
            "malformed pgvector row: dense_embedding dimension mismatch"
        )
    for value in components:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PgVectorBootstrapOperationError(
                "malformed pgvector row: dense_embedding must contain numeric values"
            )
        if not math.isfinite(float(value)):
            raise PgVectorBootstrapOperationError(
                "malformed pgvector row: dense_embedding must be finite"
            )
    return normalize_vector_float32(tuple(float(value) for value in components))


def stored_pgvector_row_from_fetched_row(
    row: Mapping[str, _PgVectorFetchedValue],
) -> StoredPgVectorRow:
    embedding_dimension = _require_int(row, "embedding_dimension")
    return StoredPgVectorRow(
        logical_point_id=_require_str(row, "logical_point_id"),
        catalog_id=_require_str(row, "catalog_id"),
        offer_id=_require_str(row, "offer_id"),
        source_revision_norm=_require_str(row, "source_revision_norm"),
        source_revision=_optional_str(row, "source_revision"),
        semantic_text_hash=_require_str(row, "semantic_text_hash"),
        embedding_provider=_require_str(row, "embedding_provider"),
        embedding_model=_require_str(row, "embedding_model"),
        embedding_revision=_optional_str(row, "embedding_revision"),
        embedding_dimension=embedding_dimension,
        derivation_version=_require_str(row, "derivation_version"),
        dense_embedding=_embedding_values(
            row["dense_embedding"],
            expected_dimension=embedding_dimension,
        ),
    )
