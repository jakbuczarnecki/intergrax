"""Extract bounded identity profiles from immutable source offers."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.catalog.identifier_normalization import (
    classify_wdc_identifier_type,
    normalize_exact_lookup_value,
)
from platform_proofs.scenarios.verified_product_identification.application.catalog.spec_table_content_parser import (
    parse_spec_table_content,
)
from platform_proofs.scenarios.verified_product_identification.application.catalog.structured_attribute_normalization import (
    DefaultStructuredAttributeNormalizationPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcSourceOffer,
)

_STRUCTURED_POLICY = DefaultStructuredAttributeNormalizationPolicy()
_IDENTIFIER_NORMALIZATION_RULE = "identifier_normalization/v1"
_BRAND_NORMALIZATION_RULE = "structured_attribute/v1"
_STRUCTURED_NORMALIZATION_RULE = "structured_attribute/v1"


@dataclass(frozen=True, slots=True)
class IdentityStructuredAttribute:
    """One canonical structured attribute relevant to identity comparison."""

    canonical_key: str
    normalized_text_value: str
    source_field: str
    source_key: str
    source_value: str

    def __post_init__(self) -> None:
        if not self.canonical_key.strip():
            raise ValueError("canonical_key must be non-empty")
        if not self.normalized_text_value.strip():
            raise ValueError("normalized_text_value must be non-empty")
        if not self.source_field.strip():
            raise ValueError("source_field must be non-empty")
        if not self.source_key.strip():
            raise ValueError("source_key must be non-empty")
        if not self.source_value.strip():
            raise ValueError("source_value must be non-empty")


@dataclass(frozen=True, slots=True)
class IdentityTypedIdentifier:
    """One normalized identifier value grouped by canonical type."""

    identifier_type: ProductIdentifierType
    normalized_value: str
    source_field: str
    source_value: str

    def __post_init__(self) -> None:
        if not self.normalized_value.strip():
            raise ValueError("normalized_value must be non-empty")
        if not self.source_field.strip():
            raise ValueError("source_field must be non-empty")
        if not self.source_value.strip():
            raise ValueError("source_value must be non-empty")


@dataclass(frozen=True, slots=True)
class SourceOfferIdentityProfile:
    """Bounded identity-relevant projection of one source offer."""

    source_ref: SourceRecordRef
    brand: str | None
    brand_source_field: str | None
    brand_source_value: str | None
    identifiers: tuple[IdentityTypedIdentifier, ...]
    structured_attributes: tuple[IdentityStructuredAttribute, ...]

    def identifiers_by_type(
        self,
        identifier_type: ProductIdentifierType,
    ) -> tuple[IdentityTypedIdentifier, ...]:
        return tuple(
            identifier
            for identifier in self.identifiers
            if identifier.identifier_type is identifier_type
        )


def build_identity_profile(
    source_offer: WdcSourceOffer,
    *,
    source_ref: SourceRecordRef,
) -> SourceOfferIdentityProfile:
    """Pure deterministic identity projection from one typed WDC offer."""

    if source_ref.offer_id.value != source_offer.offer_id:
        raise ValueError("source_ref.offer_id must match source_offer.offer_id")

    identifiers = _extract_identifiers(source_offer)
    structured_attributes = _extract_structured_attributes(source_offer)
    brand, brand_source_field, brand_source_value = _extract_brand(source_offer)

    return SourceOfferIdentityProfile(
        source_ref=source_ref,
        brand=brand,
        brand_source_field=brand_source_field,
        brand_source_value=brand_source_value,
        identifiers=identifiers,
        structured_attributes=structured_attributes,
    )


def brand_normalization_rule() -> str:
    return _BRAND_NORMALIZATION_RULE


def identifier_normalization_rule() -> str:
    return _IDENTIFIER_NORMALIZATION_RULE


def structured_normalization_rule() -> str:
    return _STRUCTURED_NORMALIZATION_RULE


def _extract_brand(source_offer: WdcSourceOffer) -> tuple[str | None, str | None, str | None]:
    if source_offer.brand is None:
        return None, None, None
    raw_brand = source_offer.brand
    normalized = _STRUCTURED_POLICY.normalized_text_value(source_value=raw_brand)
    if not normalized:
        return None, None, None
    return normalized.casefold(), "brand", raw_brand


def _extract_identifiers(source_offer: WdcSourceOffer) -> tuple[IdentityTypedIdentifier, ...]:
    identifiers: list[IdentityTypedIdentifier] = []
    seen: set[tuple[ProductIdentifierType, str]] = set()
    for entry in source_offer.identifiers:
        identifier_type = classify_wdc_identifier_type(entry.source_key)
        if identifier_type is None:
            continue
        normalized_value = normalize_exact_lookup_value(identifier_type, entry.source_value)
        if not normalized_value:
            continue
        dedup_key = (identifier_type, normalized_value)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        identifiers.append(
            IdentityTypedIdentifier(
                identifier_type=identifier_type,
                normalized_value=normalized_value,
                source_field=entry.source_key,
                source_value=entry.source_value,
            )
        )
    return tuple(
        sorted(
            identifiers,
            key=lambda item: (
                item.identifier_type.value,
                item.normalized_value,
                item.source_field,
            ),
        )
    )


def _extract_structured_attributes(
    source_offer: WdcSourceOffer,
) -> tuple[IdentityStructuredAttribute, ...]:
    attributes: list[IdentityStructuredAttribute] = []
    seen: set[tuple[str, str]] = set()

    for pair in source_offer.key_value_pairs:
        canonical_key = _STRUCTURED_POLICY.canonical_key(
            source_key=pair.source_key,
            source_field="keyValuePairs",
        )
        if canonical_key is None:
            continue
        normalized_text_value = _STRUCTURED_POLICY.normalized_text_value(
            source_value=pair.source_value,
        )
        if not normalized_text_value:
            continue
        dedup_key = (canonical_key.casefold(), normalized_text_value.casefold())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        attributes.append(
            IdentityStructuredAttribute(
                canonical_key=canonical_key,
                normalized_text_value=normalized_text_value,
                source_field="keyValuePairs",
                source_key=pair.source_key,
                source_value=pair.source_value,
            )
        )

    if source_offer.spec_table_content is not None:
        for parsed_attribute in parse_spec_table_content(source_offer.spec_table_content):
            canonical_key = _STRUCTURED_POLICY.canonical_key(
                source_key=parsed_attribute.source_key,
                source_field="specTableContent",
            )
            if canonical_key is None:
                continue
            normalized_text_value = _STRUCTURED_POLICY.normalized_text_value(
                source_value=parsed_attribute.source_value,
            )
            if not normalized_text_value:
                continue
            dedup_key = (canonical_key.casefold(), normalized_text_value.casefold())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            attributes.append(
                IdentityStructuredAttribute(
                    canonical_key=canonical_key,
                    normalized_text_value=normalized_text_value,
                    source_field="specTableContent",
                    source_key=parsed_attribute.source_key,
                    source_value=parsed_attribute.source_value,
                )
            )

    return tuple(
        sorted(
            attributes,
            key=lambda item: (
                item.canonical_key.casefold(),
                item.normalized_text_value.casefold(),
                item.source_field,
                item.source_key,
            ),
        )
    )
