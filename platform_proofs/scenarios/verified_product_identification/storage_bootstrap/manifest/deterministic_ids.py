"""Deterministic persisted identity helpers for catalog and search stores."""

from __future__ import annotations

import hashlib

from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    StructuredAttribute,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)


def structured_attribute_identity(attribute: StructuredAttribute) -> str:
    canonical = attribute.canonical_key or ""
    raw = f"{canonical}|{attribute.source_key}|{attribute.source_field}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def search_representation_point_id(
    *,
    catalog_id: str,
    offer_id: str,
    derivation_version: str,
) -> str:
    return f"vpi:{catalog_id}:{offer_id}:semantic:{derivation_version}"


def source_ref_payload(source_ref: SourceRecordRef) -> dict[str, str]:
    payload: dict[str, str] = {
        "offer_id": source_ref.offer_id.value,
        "catalog_id": source_ref.catalog_id,
    }
    if source_ref.source_revision is not None:
        payload["source_revision"] = source_ref.source_revision
    return payload
