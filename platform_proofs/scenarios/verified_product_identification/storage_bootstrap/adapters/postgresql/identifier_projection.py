"""Scenario-owned WDC identifier projection for indexed exact lookup persistence."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.catalog.identifier_normalization import (
    classify_wdc_identifier_type,
    normalize_exact_lookup_value,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalLoadRecord,
)


def _source_revision_norm(source_revision: str | None) -> str:
    return source_revision or ""


@dataclass(frozen=True, slots=True)
class ProjectedIdentifierRow:
    """One derived exact-lookup identifier row bound to a source offer identity."""

    catalog_id: str
    offer_id: str
    source_revision_norm: str
    source_revision: str | None
    identifier_type: ProductIdentifierType
    source_value: str
    normalized_value: str
    source_field: str


def project_identifiers_from_load_record(
    record: RelationalLoadRecord,
) -> tuple[ProjectedIdentifierRow, ...]:
    """Project typed identifier index rows from one relational load record."""
    source_offer = parse_wdc_source_offer_json(record.record_json)
    if source_offer.offer_id != record.source_ref.offer_id.value:
        msg = "record_json offer id must match source_ref.offer_id"
        raise ValueError(msg)

    revision_norm = _source_revision_norm(record.source_ref.source_revision)
    projected: list[ProjectedIdentifierRow] = []
    for entry in source_offer.identifiers:
        identifier_type = classify_wdc_identifier_type(entry.source_key)
        if identifier_type is None:
            continue
        normalized_value = normalize_exact_lookup_value(identifier_type, entry.source_value)
        if not normalized_value:
            continue
        projected.append(
            ProjectedIdentifierRow(
                catalog_id=record.source_ref.catalog_id,
                offer_id=record.source_ref.offer_id.value,
                source_revision_norm=revision_norm,
                source_revision=record.source_ref.source_revision,
                identifier_type=identifier_type,
                source_value=entry.source_value,
                normalized_value=normalized_value,
                source_field=entry.source_key,
            )
        )
    return tuple(projected)


def project_identifiers_from_source_ref(
    *,
    record_json: str,
    source_ref: SourceRecordRef,
) -> tuple[ProjectedIdentifierRow, ...]:
    """Project identifier rows from raw record_json and an explicit source reference."""
    return project_identifiers_from_load_record(
        RelationalLoadRecord(
            source_ref=source_ref,
            global_row_index=0,
            record_json=record_json,
            semantic_text="",
            semantic_text_hash="projection-only",
            derivation_version="projection-only",
        )
    )
