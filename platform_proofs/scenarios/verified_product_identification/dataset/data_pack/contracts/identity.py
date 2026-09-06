"""Canonical source identity helpers for universal data packs."""

from __future__ import annotations

import hashlib
from typing import TypeAlias

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)

SEMANTIC_TEXT_HASH_ALGORITHM = "sha256"
DATA_PACK_VERSION = "vpi.data_pack/1.0.0"
DATA_PACK_FORMAT_MAJOR = "1"
RELATIONAL_SCHEMA_VERSION = "vpi.relational/1.0.0"
EMBEDDING_SCHEMA_VERSION = "vpi.embedding/1.0.0"
PARQUET_FILE_FORMAT = "parquet"
PROOF_50_SAMPLE_VERSION = "proof-50/1.0.0"
PROOF_50_RECORD_COUNT = 50
SCENARIO_ID = "verified_product_identification"
VPI_CANONICAL_EMBEDDING_PROVIDER = "hf"
VPI_CANONICAL_EMBEDDING_MODEL = "BAAI/bge-m3"
VPI_CANONICAL_EMBEDDING_DIMENSION = 1024
VPI_CANONICAL_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
SOURCE_REF_IDENTITY_ENCODING_VERSION = "vpi.source-ref/1"

SourceRefKey: TypeAlias = tuple[str, str, str | None]

_REVISION_ABSENT = b"\x00"
_REVISION_PRESENT = b"\x01"


def semantic_text_hash(semantic_text: str) -> str:
    if not isinstance(semantic_text, str):
        msg = "semantic_text must be a string"
        raise TypeError(msg)
    return hashlib.sha256(semantic_text.encode("utf-8")).hexdigest()


def source_ref_key(source_ref: SourceRecordRef) -> SourceRefKey:
    return (
        source_ref.catalog_id,
        source_ref.offer_id.value,
        source_ref.source_revision,
    )


def source_ref_sort_key(source_ref: SourceRecordRef) -> tuple[str, str, str]:
    revision = source_ref.source_revision or ""
    return (source_ref.catalog_id, source_ref.offer_id.value, revision)


def _encode_length_prefixed_utf8(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def encode_source_ref_identity(
    *,
    catalog_id: str,
    offer_id: str,
    source_revision: str | None,
) -> bytes:
    payload = bytearray()
    payload.extend(_encode_length_prefixed_utf8(catalog_id))
    payload.extend(_encode_length_prefixed_utf8(offer_id))
    if source_revision is None:
        payload.extend(_REVISION_ABSENT)
    else:
        payload.extend(_REVISION_PRESENT)
        payload.extend(_encode_length_prefixed_utf8(source_revision))
    return bytes(payload)


def encode_source_ref_identity_from_ref(source_ref: SourceRecordRef) -> bytes:
    return encode_source_ref_identity(
        catalog_id=source_ref.catalog_id,
        offer_id=source_ref.offer_id.value,
        source_revision=source_ref.source_revision,
    )


def source_ref_set_sha256_from_keys(
    keys: tuple[SourceRefKey, ...],
) -> str:
    encoded_records = tuple(
        encode_source_ref_identity(
            catalog_id=catalog_id,
            offer_id=offer_id,
            source_revision=source_revision,
        )
        for catalog_id, offer_id, source_revision in keys
    )
    ordered = sorted(encoded_records)
    digest = hashlib.sha256()
    for encoded in ordered:
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def source_ref_set_sha256(source_refs: tuple[SourceRecordRef, ...]) -> str:
    return source_ref_set_sha256_from_keys(
        tuple(source_ref_key(source_ref) for source_ref in source_refs)
    )


def source_ref_from_columns(
    *,
    catalog_id: str,
    offer_id: str,
    source_revision: str | None,
) -> SourceRecordRef:
    from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
        ProductOfferId,
    )

    return SourceRecordRef(
        offer_id=ProductOfferId(offer_id),
        catalog_id=catalog_id,
        source_revision=source_revision,
    )
