"""Canonical source identity helpers for universal data packs."""

from __future__ import annotations

import hashlib

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


def semantic_text_hash(semantic_text: str) -> str:
    if not isinstance(semantic_text, str):
        msg = "semantic_text must be a string"
        raise TypeError(msg)
    return hashlib.sha256(semantic_text.encode("utf-8")).hexdigest()


def source_ref_key(source_ref: SourceRecordRef) -> tuple[str, str, str | None]:
    return (
        source_ref.catalog_id,
        source_ref.offer_id.value,
        source_ref.source_revision,
    )


def source_ref_sort_key(source_ref: SourceRecordRef) -> tuple[str, str, str]:
    revision = source_ref.source_revision or ""
    return (source_ref.catalog_id, source_ref.offer_id.value, revision)


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
