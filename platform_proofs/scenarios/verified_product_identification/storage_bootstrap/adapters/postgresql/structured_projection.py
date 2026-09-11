"""Scenario-owned structured attribute projection for indexed retrieval persistence."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    derive_search_representation,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalLoadRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    structured_attribute_identity,
)


def _source_revision_norm(source_revision: str | None) -> str:
    return source_revision or ""


@dataclass(frozen=True, slots=True)
class ProjectedStructuredAttributeRow:
    """One derived structured attribute row bound to a source offer identity."""

    catalog_id: str
    offer_id: str
    source_revision_norm: str
    source_revision: str | None
    attr_identity: str
    canonical_key: str | None
    source_key: str
    source_value: str
    normalized_text_value: str
    typed_value_text: str | None
    source_field: str


def project_structured_from_load_record(
    record: RelationalLoadRecord,
) -> tuple[ProjectedStructuredAttributeRow, ...]:
    """Project typed structured attribute index rows from one relational load record."""
    source_offer = parse_wdc_source_offer_json(record.record_json)
    if source_offer.offer_id != record.source_ref.offer_id.value:
        msg = "record_json offer id must match source_ref.offer_id"
        raise ValueError(msg)

    representation = derive_search_representation(
        source_offer,
        source_ref=record.source_ref,
        derivation_version=record.derivation_version,
    )
    revision_norm = _source_revision_norm(record.source_ref.source_revision)
    projected: list[ProjectedStructuredAttributeRow] = []
    for attribute in representation.structured.attributes:
        typed_value_text = (
            str(attribute.typed_value) if attribute.typed_value is not None else None
        )
        projected.append(
            ProjectedStructuredAttributeRow(
                catalog_id=record.source_ref.catalog_id,
                offer_id=record.source_ref.offer_id.value,
                source_revision_norm=revision_norm,
                source_revision=record.source_ref.source_revision,
                attr_identity=structured_attribute_identity(attribute),
                canonical_key=attribute.canonical_key,
                source_key=attribute.source_key,
                source_value=attribute.source_value,
                normalized_text_value=attribute.normalized_text_value,
                typed_value_text=typed_value_text,
                source_field=attribute.source_field,
            )
        )
    return tuple(projected)
