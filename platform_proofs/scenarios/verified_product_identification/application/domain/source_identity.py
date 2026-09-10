"""Canonical source-reference identity encoding — domain-level, provider-neutral."""

from __future__ import annotations

import hashlib

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)

SOURCE_REF_IDENTITY_ENCODING_VERSION = "vpi.source-ref/1"
HYPOTHESIS_IDENTITY_ENCODING_VERSION = "vpi.hypothesis/1"

_REVISION_ABSENT = b"\x00"
_REVISION_PRESENT = b"\x01"


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


def source_ref_set_sha256(source_refs: tuple[SourceRecordRef, ...]) -> str:
    """Deterministic digest over a sorted member set — used for hypothesis IDs."""

    if not source_refs:
        raise ValueError("source_refs must be non-empty")
    encoded_records = tuple(
        encode_source_ref_identity_from_ref(source_ref)
        for source_ref in sorted(source_refs, key=source_ref_sort_key)
    )
    digest = hashlib.sha256()
    digest.update(HYPOTHESIS_IDENTITY_ENCODING_VERSION.encode("utf-8"))
    digest.update(len(encoded_records).to_bytes(4, "big"))
    for encoded in encoded_records:
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()
