"""Typed Qdrant vector payload contracts for VPI storage bootstrap."""

from __future__ import annotations

import array
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.errors import (
    QdrantBootstrapOperationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.configuration import (
    ExpectedVectorIdentity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    VectorLoadRecord,
)

LOGICAL_POINT_ID_PAYLOAD_KEY = "logical_id"
SOURCE_CATALOG_PAYLOAD_KEY = "catalog_id"
SOURCE_OFFER_PAYLOAD_KEY = "offer_id"
SOURCE_REVISION_PAYLOAD_KEY = "source_revision"
SEMANTIC_TEXT_HASH_PAYLOAD_KEY = "semantic_text_hash"
EMBEDDING_PROVIDER_PAYLOAD_KEY = "embedding_provider"
EMBEDDING_MODEL_PAYLOAD_KEY = "embedding_model"
EMBEDDING_REVISION_PAYLOAD_KEY = "embedding_revision"
EMBEDDING_DIMENSION_PAYLOAD_KEY = "embedding_dimension"
DERIVATION_VERSION_PAYLOAD_KEY = "derivation_version"


@dataclass(frozen=True, slots=True)
class QdrantVectorPayload:
    logical_point_id: str
    catalog_id: str
    offer_id: str
    source_revision: str | None
    semantic_text_hash: str
    embedding_provider: str
    embedding_model: str
    embedding_revision: str | None
    embedding_dimension: int
    derivation_version: str

    def to_provider_payload(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = {
            LOGICAL_POINT_ID_PAYLOAD_KEY: self.logical_point_id,
            SOURCE_CATALOG_PAYLOAD_KEY: self.catalog_id,
            SOURCE_OFFER_PAYLOAD_KEY: self.offer_id,
            SEMANTIC_TEXT_HASH_PAYLOAD_KEY: self.semantic_text_hash,
            EMBEDDING_PROVIDER_PAYLOAD_KEY: self.embedding_provider,
            EMBEDDING_MODEL_PAYLOAD_KEY: self.embedding_model,
            EMBEDDING_DIMENSION_PAYLOAD_KEY: self.embedding_dimension,
            DERIVATION_VERSION_PAYLOAD_KEY: self.derivation_version,
        }
        if self.source_revision is not None:
            payload[SOURCE_REVISION_PAYLOAD_KEY] = self.source_revision
        if self.embedding_revision is not None:
            payload[EMBEDDING_REVISION_PAYLOAD_KEY] = self.embedding_revision
        return payload


@dataclass(frozen=True, slots=True)
class QdrantStoredPoint:
    point_id: str | int
    logical_point_id: str
    payload: QdrantVectorPayload
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class QdrantUpsertPoint:
    id: str | int
    vector: list[float] | dict[str, list[float]]
    payload: dict[str, str | int]


def payload_from_record(record: VectorLoadRecord) -> QdrantVectorPayload:
    source_ref = record.source_ref
    return QdrantVectorPayload(
        logical_point_id=record.logical_point_id,
        catalog_id=source_ref.catalog_id,
        offer_id=source_ref.offer_id.value,
        source_revision=source_ref.source_revision,
        semantic_text_hash=record.semantic_text_hash,
        embedding_provider=record.embedding_provider,
        embedding_model=record.embedding_model,
        embedding_revision=record.embedding_revision,
        embedding_dimension=record.embedding_dimension,
        derivation_version=record.derivation_version,
    )


def source_ref_from_payload(payload: QdrantVectorPayload) -> SourceRecordRef:
    from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
        ProductOfferId,
    )

    return SourceRecordRef(
        offer_id=ProductOfferId(payload.offer_id),
        catalog_id=payload.catalog_id,
        source_revision=payload.source_revision,
    )


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


def payload_identity_matches(
    stored: QdrantVectorPayload,
    expected: QdrantVectorPayload,
) -> bool:
    return (
        stored.logical_point_id == expected.logical_point_id
        and stored.catalog_id == expected.catalog_id
        and stored.offer_id == expected.offer_id
        and stored.source_revision == expected.source_revision
        and stored.semantic_text_hash == expected.semantic_text_hash
        and stored.embedding_provider == expected.embedding_provider
        and stored.embedding_model == expected.embedding_model
        and stored.embedding_revision == expected.embedding_revision
        and stored.embedding_dimension == expected.embedding_dimension
        and stored.derivation_version == expected.derivation_version
    )


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


def _require_str_field(raw_payload: dict[str, str | int], key: str) -> str:
    value = raw_payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise QdrantBootstrapOperationError(f"stored payload missing {key}")
    return value


def _optional_str_field(raw_payload: dict[str, str | int], key: str) -> str | None:
    value = raw_payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise QdrantBootstrapOperationError(f"stored payload field {key} is not a string")
    return value


def _require_int_field(raw_payload: dict[str, str | int], key: str) -> int:
    value = raw_payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise QdrantBootstrapOperationError(f"stored payload missing {key}")
    return value


def payload_from_provider_dict(raw_payload: dict[str, str | int]) -> QdrantVectorPayload:
    return QdrantVectorPayload(
        logical_point_id=_require_str_field(raw_payload, LOGICAL_POINT_ID_PAYLOAD_KEY),
        catalog_id=_require_str_field(raw_payload, SOURCE_CATALOG_PAYLOAD_KEY),
        offer_id=_require_str_field(raw_payload, SOURCE_OFFER_PAYLOAD_KEY),
        source_revision=_optional_str_field(raw_payload, SOURCE_REVISION_PAYLOAD_KEY),
        semantic_text_hash=_require_str_field(raw_payload, SEMANTIC_TEXT_HASH_PAYLOAD_KEY),
        embedding_provider=_require_str_field(raw_payload, EMBEDDING_PROVIDER_PAYLOAD_KEY),
        embedding_model=_require_str_field(raw_payload, EMBEDDING_MODEL_PAYLOAD_KEY),
        embedding_revision=_optional_str_field(raw_payload, EMBEDDING_REVISION_PAYLOAD_KEY),
        embedding_dimension=_require_int_field(raw_payload, EMBEDDING_DIMENSION_PAYLOAD_KEY),
        derivation_version=_require_str_field(raw_payload, DERIVATION_VERSION_PAYLOAD_KEY),
    )
