"""Typed vector hit identity decoder — storage payload parity at provider boundary."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.knowledge.contracts.validation import JsonValue

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.payload import (
    SOURCE_CATALOG_PAYLOAD_KEY,
    SOURCE_OFFER_PAYLOAD_KEY,
    SOURCE_REVISION_PAYLOAD_KEY,
)


class VectorHitIdentityDecodeError(ValueError):
    """Raised when a vector hit payload lacks required source identity."""


@dataclass(frozen=True, slots=True)
class VectorHitIdentity:
    catalog_id: str
    offer_id: ProductOfferId
    source_revision: str | None

    def to_source_record_ref(self) -> SourceRecordRef:
        return SourceRecordRef(
            offer_id=self.offer_id,
            catalog_id=self.catalog_id,
            source_revision=self.source_revision,
        )

    @property
    def source_revision_norm(self) -> str:
        return self.source_revision or ""


def _require_str_field(raw_payload: dict[str, str | int], key: str) -> str:
    value = raw_payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VectorHitIdentityDecodeError(f"vector hit missing {key}")
    return value


def _optional_str_field(raw_payload: dict[str, str | int], key: str) -> str | None:
    value = raw_payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise VectorHitIdentityDecodeError(f"vector hit field {key} is not a string")
    return value


def decode_vector_hit_identity_from_storage_payload(
    raw_payload: dict[str, str | int],
) -> VectorHitIdentity:
    """Decode identity using the same keys written by Qdrant storage bootstrap."""
    catalog_id = _require_str_field(raw_payload, SOURCE_CATALOG_PAYLOAD_KEY)
    offer_id_raw = _require_str_field(raw_payload, SOURCE_OFFER_PAYLOAD_KEY)
    source_revision = _optional_str_field(raw_payload, SOURCE_REVISION_PAYLOAD_KEY)
    return VectorHitIdentity(
        catalog_id=catalog_id,
        offer_id=ProductOfferId(offer_id_raw),
        source_revision=source_revision,
    )


def _coerce_storage_scalar(value: JsonValue, *, field_name: str) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise VectorHitIdentityDecodeError(f"vector hit field {field_name} has unsupported type")


def storage_payload_from_metadata(metadata: dict[str, JsonValue]) -> dict[str, str | int]:
    """Project provider metadata to the typed storage payload shape for identity decode."""
    projected: dict[str, str | int] = {}
    for key in (
        SOURCE_CATALOG_PAYLOAD_KEY,
        SOURCE_OFFER_PAYLOAD_KEY,
        SOURCE_REVISION_PAYLOAD_KEY,
    ):
        if key not in metadata:
            continue
        coerced = _coerce_storage_scalar(metadata[key], field_name=key)
        if coerced is not None:
            projected[key] = coerced
    return projected


def decode_vector_hit_identity_from_metadata(metadata: dict[str, JsonValue]) -> VectorHitIdentity:
    return decode_vector_hit_identity_from_storage_payload(
        storage_payload_from_metadata(metadata)
    )
