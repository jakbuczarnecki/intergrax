"""Deterministic derivation of provider-neutral search representations."""

from __future__ import annotations

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
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
    DerivedOfferSearchRepresentation,
    ExactIdentifierTerm,
    ExactSearchRepresentation,
    LexicalSearchRepresentation,
    SemanticContributingField,
    SemanticSearchRepresentation,
    StructuredAttribute,
    StructuredSearchRepresentation,
    StructuredTextFragment,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcSourceOffer,
)

_DEFAULT_STRUCTURED_POLICY = DefaultStructuredAttributeNormalizationPolicy()


def build_source_record_ref(
    source_offer: WdcSourceOffer,
    *,
    catalog_id: str,
    source_revision: str | None = None,
) -> SourceRecordRef:
    """Build the immutable source reference for one typed WDC offer."""
    return SourceRecordRef(
        offer_id=ProductOfferId(source_offer.offer_id),
        catalog_id=catalog_id,
        source_revision=source_revision,
    )


def derive_search_representation(
    source_offer: WdcSourceOffer,
    *,
    source_ref: SourceRecordRef,
    derivation_version: str = SEARCH_REPRESENTATION_DERIVATION_VERSION,
) -> DerivedOfferSearchRepresentation:
    """Pure deterministic transform from one source offer to derived search views."""
    if source_ref.offer_id.value != source_offer.offer_id:
        msg = "source_ref.offer_id must match source_offer.offer_id"
        raise ValueError(msg)

    exact = _derive_exact_representation(source_offer, source_ref=source_ref)
    lexical = _derive_lexical_representation(source_offer, source_ref=source_ref)
    structured = _derive_structured_representation(source_offer, source_ref=source_ref)
    semantic = _derive_semantic_representation(source_offer, source_ref=source_ref)

    return DerivedOfferSearchRepresentation(
        source_ref=source_ref,
        exact=exact,
        lexical=lexical,
        structured=structured,
        semantic=semantic,
        derivation_version=derivation_version,
    )


def flatten_lexical_text(representation: LexicalSearchRepresentation) -> str:
    """Deterministically flatten lexical fields for provider-specific indexing."""
    parts: list[str] = []
    if representation.title is not None:
        parts.append(representation.title)
    if representation.brand is not None:
        parts.append(representation.brand)
    if representation.description is not None:
        parts.append(representation.description)
    for fragment in representation.structured_text_fragments:
        parts.append(fragment.text)
    return "\n".join(parts)


def _derive_exact_representation(
    source_offer: WdcSourceOffer,
    *,
    source_ref: SourceRecordRef,
) -> ExactSearchRepresentation:
    terms: list[ExactIdentifierTerm] = []
    for entry in source_offer.identifiers:
        identifier_type = classify_wdc_identifier_type(entry.source_key)
        if identifier_type is None:
            continue
        lookup_value = normalize_exact_lookup_value(identifier_type, entry.source_value)
        if not lookup_value:
            continue
        terms.append(
            ExactIdentifierTerm(
                identifier_type=identifier_type,
                source_value=entry.source_value,
                lookup_value=lookup_value,
                source_field=entry.source_key,
            )
        )
    return ExactSearchRepresentation(source_ref=source_ref, terms=tuple(terms))


def _derive_lexical_representation(
    source_offer: WdcSourceOffer,
    *,
    source_ref: SourceRecordRef,
) -> LexicalSearchRepresentation:
    fragments: list[StructuredTextFragment] = []
    for pair in source_offer.key_value_pairs:
        fragments.append(
            StructuredTextFragment(
                source_field="keyValuePairs",
                text=f"{pair.source_key}: {pair.source_value}",
            )
        )
    if source_offer.spec_table_content is not None:
        fragments.append(
            StructuredTextFragment(
                source_field="specTableContent",
                text=source_offer.spec_table_content,
            )
        )
    return LexicalSearchRepresentation(
        source_ref=source_ref,
        title=source_offer.title,
        brand=source_offer.brand,
        description=source_offer.description,
        structured_text_fragments=tuple(fragments),
    )


def _derive_structured_representation(
    source_offer: WdcSourceOffer,
    *,
    source_ref: SourceRecordRef,
) -> StructuredSearchRepresentation:
    attributes: list[StructuredAttribute] = []
    seen_pairs: set[tuple[str | None, str]] = set()

    for pair in source_offer.key_value_pairs:
        attribute = StructuredAttribute(
            canonical_key=_DEFAULT_STRUCTURED_POLICY.canonical_key(
                source_key=pair.source_key,
                source_field="keyValuePairs",
            ),
            source_key=pair.source_key,
            source_value=pair.source_value,
            normalized_text_value=_DEFAULT_STRUCTURED_POLICY.normalized_text_value(
                source_value=pair.source_value,
            ),
            typed_value=_DEFAULT_STRUCTURED_POLICY.typed_value(
                raw_value=pair.raw_value,
            ),
            source_field="keyValuePairs",
        )
        attributes.append(attribute)
        seen_pairs.add((attribute.canonical_key, attribute.normalized_text_value))

    if source_offer.spec_table_content is not None:
        for parsed_attribute in parse_spec_table_content(source_offer.spec_table_content):
            attribute = StructuredAttribute(
                canonical_key=_DEFAULT_STRUCTURED_POLICY.canonical_key(
                    source_key=parsed_attribute.source_key,
                    source_field="specTableContent",
                ),
                source_key=parsed_attribute.source_key,
                source_value=parsed_attribute.source_value,
                normalized_text_value=_DEFAULT_STRUCTURED_POLICY.normalized_text_value(
                    source_value=parsed_attribute.source_value,
                ),
                typed_value=None,
                source_field="specTableContent",
            )
            dedup_key = (attribute.canonical_key, attribute.normalized_text_value)
            if dedup_key in seen_pairs:
                continue
            attributes.append(attribute)
            seen_pairs.add(dedup_key)

    return StructuredSearchRepresentation(source_ref=source_ref, attributes=tuple(attributes))


def _derive_semantic_representation(
    source_offer: WdcSourceOffer,
    *,
    source_ref: SourceRecordRef,
) -> SemanticSearchRepresentation:
    contributing_fields: list[SemanticContributingField] = []
    if source_offer.title is not None:
        contributing_fields.append(
            SemanticContributingField(source_field="title", text=source_offer.title)
        )
    if source_offer.brand is not None:
        contributing_fields.append(
            SemanticContributingField(source_field="brand", text=source_offer.brand)
        )
    if source_offer.description is not None:
        contributing_fields.append(
            SemanticContributingField(
                source_field="description",
                text=source_offer.description,
            )
        )
    for pair in source_offer.key_value_pairs:
        contributing_fields.append(
            SemanticContributingField(
                source_field="keyValuePairs",
                text=f"{pair.source_key}: {pair.source_value}",
            )
        )
    if source_offer.spec_table_content is not None:
        contributing_fields.append(
            SemanticContributingField(
                source_field="specTableContent",
                text=source_offer.spec_table_content,
            )
        )

    semantic_text = "\n".join(field.text for field in contributing_fields)
    return SemanticSearchRepresentation(
        source_ref=source_ref,
        semantic_text=semantic_text,
        contributing_fields=tuple(contributing_fields),
    )
