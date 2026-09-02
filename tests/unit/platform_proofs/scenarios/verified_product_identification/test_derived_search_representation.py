"""Unit tests for provider-neutral derived search representation."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.application.catalog import (
    build_source_record_ref,
    derive_search_representation,
    flatten_lexical_text,
    resolve_source_record,
)
from platform_proofs.scenarios.verified_product_identification.application.catalog.identifier_normalization import (
    classify_wdc_identifier_type,
    normalize_exact_lookup_value,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    SourceRecordFetchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    DerivedOfferSearchRepresentation,
    ExactIdentifierTerm,
    ExactSearchRepresentation,
    LexicalSearchRepresentation,
    ProductIdentifierType,
    ProductOfferId,
    ProductSourceProvenance,
    ProductSourceRecord,
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
    SemanticSearchRepresentation,
    SourceRecordRef,
    StructuredSearchRepresentation,
    parse_wdc_source_offer_json,
)

pytestmark = pytest.mark.unit

CATALOG_ID = "catalog-neutral"
SOURCE_REVISION = "rev-neutral"

# Minimized from real WDC vertical-tab colon alternation (sample id 9084112).
_REAL_WDC_PARSEABLE_SPEC = (
    "Wingspan:\x0b 39.3\"\x0b Overall Length:\x0b 33.1\"\x0b "
    "Wing Area:\x0b 272.8 sq inches"
)

# Minimized from real WDC single-blob newline prose pattern (~71% of sample).
_REAL_WDC_UNPARSEABLE_SPEC = (
    "Dimensions et détails M Dimension des verres 52 "
    "Dimension du pont nasal 16 Longueur des branches 140"
)


def _electronics_offer_record() -> dict[str, object]:
    return {
        "id": 1001,
        "cluster_id": 42,
        "category": "Consumer Electronics",
        "identifiers": [
            {"/gtin13": "[8806095123456]"},
            {"/mpn": "[MZ-V9P2T0BW]"},
            {"/sku": "[SKU-NEUTRAL-01]"},
            {"/productID": "[PROD-NEUTRAL-01]"},
        ],
        "title": "Neutral NVMe Storage Device 2TB",
        "description": "High-speed internal storage for general-purpose systems.",
        "brand": "NeutralBrand",
        "price": "199.99",
        "keyValuePairs": {
            "Capacity": "2TB",
            "Interface": "NVMe",
            "InStock": True,
            "WarrantyYears": 3,
        },
        "specTableContent": "Capacity 2TB\nInterface NVMe",
    }


def _minimal_offer_record() -> dict[str, object]:
    return {
        "id": "offer-minimal",
        "title": "Household Cleaning Spray",
    }


def _source_ref_for_offer_id(offer_id: str) -> SourceRecordRef:
    return SourceRecordRef(
        offer_id=ProductOfferId(offer_id),
        catalog_id=CATALOG_ID,
        source_revision=SOURCE_REVISION,
    )


def _derive_from_record(record: dict[str, object]) -> tuple[DerivedOfferSearchRepresentation, SourceRecordRef]:
    source_offer = parse_wdc_source_offer_json(json.dumps(record, ensure_ascii=False))
    source_ref = build_source_record_ref(
        source_offer,
        catalog_id=CATALOG_ID,
        source_revision=SOURCE_REVISION,
    )
    derived = derive_search_representation(source_offer, source_ref=source_ref)
    return derived, source_ref


def test_deterministic_derivation_same_source_same_representation() -> None:
    record = _electronics_offer_record()
    first, first_ref = _derive_from_record(record)
    second, second_ref = _derive_from_record(record)

    assert first == second
    assert first_ref == second_ref
    assert first.derivation_version == SEARCH_REPRESENTATION_DERIVATION_VERSION


def test_source_identity_preserved_in_all_channels() -> None:
    derived, source_ref = _derive_from_record(_electronics_offer_record())

    assert derived.source_ref == source_ref
    assert derived.exact.source_ref == source_ref
    assert derived.lexical.source_ref == source_ref
    assert derived.structured.source_ref == source_ref
    assert derived.semantic.source_ref == source_ref


def test_exact_typed_identifiers_become_lookup_terms() -> None:
    derived, _ = _derive_from_record(_electronics_offer_record())

    terms_by_type = {term.identifier_type: term for term in derived.exact.terms}
    assert set(terms_by_type) == {
        ProductIdentifierType.GTIN,
        ProductIdentifierType.MPN,
        ProductIdentifierType.SKU,
        ProductIdentifierType.PRODUCT_ID,
    }
    gtin = terms_by_type[ProductIdentifierType.GTIN]
    assert gtin.source_value == "[8806095123456]"
    assert gtin.lookup_value == "8806095123456"
    assert gtin.source_field == "/gtin13"
    mpn = terms_by_type[ProductIdentifierType.MPN]
    assert mpn.source_value == "[MZ-V9P2T0BW]"
    assert mpn.lookup_value == "MZ-V9P2T0BW"


def test_missing_identifiers_yield_empty_exact_representation() -> None:
    derived, _ = _derive_from_record(_minimal_offer_record())

    assert derived.exact.terms == ()


def test_lexical_preserves_field_boundaries() -> None:
    derived, _ = _derive_from_record(_electronics_offer_record())
    lexical = derived.lexical

    assert lexical.title == "Neutral NVMe Storage Device 2TB"
    assert lexical.brand == "NeutralBrand"
    assert lexical.description == "High-speed internal storage for general-purpose systems."
    assert [fragment.source_field for fragment in lexical.structured_text_fragments] == [
        "keyValuePairs",
        "keyValuePairs",
        "keyValuePairs",
        "keyValuePairs",
        "specTableContent",
    ]
    flattened = flatten_lexical_text(lexical)
    assert "Neutral NVMe Storage Device 2TB" in flattened
    assert "Capacity: 2TB" in flattened
    assert "Interface NVMe" in flattened
    assert "199.99" not in flattened


def test_semantic_text_is_deterministic_without_embedding() -> None:
    derived, _ = _derive_from_record(_electronics_offer_record())

    assert derived.semantic.semantic_text
    assert derived.semantic.semantic_text == "\n".join(
        field.text for field in derived.semantic.contributing_fields
    )
    assert "Neutral NVMe Storage Device 2TB" in derived.semantic.semantic_text
    assert "Capacity: 2TB" in derived.semantic.semantic_text
    assert "42" not in derived.semantic.semantic_text
    assert "199.99" not in derived.semantic.semantic_text


def test_structured_kvp_produces_bounded_attributes_without_fake_spec_blob() -> None:
    derived, _ = _derive_from_record(_electronics_offer_record())
    attributes = derived.structured.attributes

    assert len(attributes) == 4
    by_key = {attribute.source_key: attribute for attribute in attributes}
    capacity = by_key["Capacity"]
    assert capacity.source_value == "2TB"
    assert capacity.normalized_text_value == "2TB"
    assert capacity.source_field == "keyValuePairs"
    assert capacity.canonical_key == "Capacity"
    in_stock = by_key["InStock"]
    assert in_stock.typed_value is True
    warranty = by_key["WarrantyYears"]
    assert warranty.typed_value == 3
    assert "specTableContent" not in by_key


def test_missing_brand_category_description_still_derives() -> None:
    record = {
        "id": 2002,
        "cluster_id": 77,
        "category": None,
        "identifiers": None,
        "title": "Industrial Relay Module",
        "description": None,
        "brand": None,
        "keyValuePairs": {"Voltage": "24V"},
        "specTableContent": None,
    }
    derived, _ = _derive_from_record(record)

    assert derived.lexical.title == "Industrial Relay Module"
    assert derived.lexical.brand is None
    assert derived.lexical.description is None
    assert derived.exact.terms == ()
    assert derived.semantic.semantic_text == "Industrial Relay Module\nVoltage: 24V"


def test_cluster_id_does_not_establish_verified_identity() -> None:
    derived, _ = _derive_from_record(_electronics_offer_record())
    serialized = json.dumps(
        {
            "exact": [
                {
                    "type": term.identifier_type.value,
                    "lookup": term.lookup_value,
                }
                for term in derived.exact.terms
            ],
            "lexical": flatten_lexical_text(derived.lexical),
            "structured": [attribute.source_value for attribute in derived.structured.attributes],
            "semantic": derived.semantic.semantic_text,
        },
        ensure_ascii=False,
    )
    assert "42" not in serialized


def test_derived_representation_points_back_to_source_truth() -> None:
    record = _electronics_offer_record()
    source_offer = parse_wdc_source_offer_json(json.dumps(record, ensure_ascii=False))
    source_ref = build_source_record_ref(
        source_offer,
        catalog_id=CATALOG_ID,
        source_revision=SOURCE_REVISION,
    )
    derived = derive_search_representation(source_offer, source_ref=source_ref)

    @dataclass(frozen=True, slots=True)
    class InMemorySourceStore:
        def fetch(self, requested_ref: SourceRecordRef) -> SourceRecordFetchResult:
            assert requested_ref == derived.source_ref
            return SourceRecordFetchResult(
                record=ProductSourceRecord(
                    offer_id=requested_ref.offer_id,
                    record_payload_ref=f"{CATALOG_ID}:payload:{requested_ref.offer_id.value}",
                    provenance=ProductSourceProvenance(
                        catalog_id=CATALOG_ID,
                        source_revision=SOURCE_REVISION,
                    ),
                )
            )

    candidate_offer_id = ProductOfferId(source_offer.offer_id)
    from platform_proofs.scenarios.verified_product_identification.application.domain import (
        ProductCandidate,
        RetrievalChannel,
    )

    candidate = ProductCandidate(
        offer_id=candidate_offer_id,
        channel=RetrievalChannel.EXACT,
        rank=0,
        source_ref=derived.source_ref,
    )
    source_record = resolve_source_record(candidate, InMemorySourceStore())
    assert source_record.offer_id.value == source_offer.offer_id


def test_no_provider_specific_imports_in_derivation_modules() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    module_paths = [
        repo_root
        / "platform_proofs/scenarios/verified_product_identification/application/catalog/derive_search_representation.py",
        repo_root
        / "platform_proofs/scenarios/verified_product_identification/application/domain/search_representation.py",
        repo_root
        / "platform_proofs/scenarios/verified_product_identification/application/domain/wdc_source_offer.py",
    ]
    forbidden_tokens = (
        "postgres",
        "postgresql",
        "mysql",
        "qdrant",
        "pgvector",
        "elasticsearch",
        "openai",
    )
    for module_path in module_paths:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.casefold() not in forbidden_tokens
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module.casefold() not in forbidden_tokens


def test_parse_wdc_source_offer_handles_empty_optional_fields() -> None:
    record = {
        "id": 9,
        "identifiers": [],
        "title": "   ",
        "description": "",
        "brand": None,
        "keyValuePairs": {},
        "specTableContent": None,
    }
    source_offer = parse_wdc_source_offer_json(json.dumps(record, ensure_ascii=False))

    assert source_offer.offer_id == "9"
    assert source_offer.title is None
    assert source_offer.identifiers == ()
    assert source_offer.key_value_pairs == ()


def test_mismatched_channel_source_ref_rejected() -> None:
    derived, source_ref = _derive_from_record(_electronics_offer_record())
    mismatched_ref = _source_ref_for_offer_id("other-offer")
    lexical = replace(derived.lexical, source_ref=mismatched_ref)

    with pytest.raises(ValueError, match="source_ref mismatch"):
        DerivedOfferSearchRepresentation(
            source_ref=source_ref,
            exact=derived.exact,
            lexical=lexical,
            structured=derived.structured,
            semantic=derived.semantic,
            derivation_version=derived.derivation_version,
        )


def test_all_channel_source_refs_matching_passes_envelope_validation() -> None:
    derived, source_ref = _derive_from_record(_electronics_offer_record())

    envelope = DerivedOfferSearchRepresentation(
        source_ref=source_ref,
        exact=derived.exact,
        lexical=derived.lexical,
        structured=derived.structured,
        semantic=derived.semantic,
        derivation_version=derived.derivation_version,
    )
    assert envelope.source_ref == source_ref


@pytest.mark.parametrize(
    ("source_value", "expected_lookup"),
    [
        ("[8806095123456]", "8806095123456"),
        ("[12345670]", "12345670"),
        ("[123456789012]", "123456789012"),
        ("[88060951234567]", "88060951234567"),
    ],
)
def test_valid_gtin_lengths_accepted(source_value: str, expected_lookup: str) -> None:
    lookup = normalize_exact_lookup_value(ProductIdentifierType.GTIN, source_value)
    assert lookup == expected_lookup


def test_ambiguous_gtin_source_value_does_not_produce_exact_term() -> None:
    record = {
        "id": 3001,
        "identifiers": [{"/gtin13": "[7332543307227, 7332543297146]"}],
        "title": "Ambiguous GTIN offer",
    }
    derived, _ = _derive_from_record(record)
    assert derived.exact.terms == ()


def test_invalid_gtin_length_produces_no_exact_term() -> None:
    record = {
        "id": 3002,
        "identifiers": [{"/gtin13": "[12345]"}],
        "title": "Short GTIN offer",
    }
    derived, _ = _derive_from_record(record)
    assert derived.exact.terms == ()


def test_gtin_with_separators_only_when_unambiguous_single_group() -> None:
    accepted = normalize_exact_lookup_value(
        ProductIdentifierType.GTIN,
        "[8806-0951-23456]",
    )
    assert accepted == "8806095123456"

    rejected = normalize_exact_lookup_value(
        ProductIdentifierType.GTIN,
        "[12345678 87654321]",
    )
    assert rejected == ""


def test_real_wdc_parseable_spec_table_content_yields_structured_attributes() -> None:
    record = {
        "id": 4001,
        "title": "Model aircraft kit",
        "specTableContent": _REAL_WDC_PARSEABLE_SPEC,
    }
    derived, _ = _derive_from_record(record)
    attributes = derived.structured.attributes

    assert len(attributes) == 3
    by_key = {attribute.source_key: attribute for attribute in attributes}
    assert by_key["Wingspan"].source_value == '39.3"'
    assert by_key["Overall Length"].source_value == '33.1"'
    assert by_key["Wing Area"].source_value == "272.8 sq inches"
    assert all(attribute.source_field == "specTableContent" for attribute in attributes)


def test_real_wdc_unparseable_spec_remains_lexical_semantic_without_structured_attrs() -> None:
    record = {
        "id": 4002,
        "title": "Eyewear accessory",
        "specTableContent": _REAL_WDC_UNPARSEABLE_SPEC,
    }
    derived, _ = _derive_from_record(record)

    assert derived.structured.attributes == ()
    assert any(
        fragment.source_field == "specTableContent"
        for fragment in derived.lexical.structured_text_fragments
    )
    assert any(
        field.source_field == "specTableContent"
        for field in derived.semantic.contributing_fields
    )


def test_key_value_pairs_behavior_remains_intact() -> None:
    record = {
        "id": 4003,
        "title": "Relay module",
        "keyValuePairs": {"Voltage": "24V", "Current": "10A"},
        "specTableContent": None,
    }
    derived, _ = _derive_from_record(record)

    assert len(derived.structured.attributes) == 2
    assert {attribute.source_key for attribute in derived.structured.attributes} == {
        "Current",
        "Voltage",
    }


def test_duplicate_structured_pair_from_spec_and_kvp_is_handled_deterministically() -> None:
    record = {
        "id": 4004,
        "title": "Storage device",
        "keyValuePairs": {"Capacity": "2TB"},
        "specTableContent": "Capacity: 2TB\nInterface: NVMe",
    }
    derived, _ = _derive_from_record(record)
    capacity_attributes = [
        attribute
        for attribute in derived.structured.attributes
        if attribute.source_key == "Capacity"
    ]
    assert len(capacity_attributes) == 1
    assert capacity_attributes[0].source_field == "keyValuePairs"
    interface_attributes = [
        attribute
        for attribute in derived.structured.attributes
        if attribute.source_key == "Interface"
    ]
    assert len(interface_attributes) == 1
    assert interface_attributes[0].source_field == "specTableContent"


def test_derivation_version_is_v2() -> None:
    derived, _ = _derive_from_record(_electronics_offer_record())
    assert SEARCH_REPRESENTATION_DERIVATION_VERSION == "v2"
    assert derived.derivation_version == "v2"


def test_gtin_classification_uses_explicit_wdc_keys_only() -> None:
    assert classify_wdc_identifier_type("/gtin13") is ProductIdentifierType.GTIN
    assert classify_wdc_identifier_type("/identifier") is None
    assert classify_wdc_identifier_type("/legacyGtinAlias") is None


def test_exact_identifier_term_rejects_empty_lookup_value() -> None:
    with pytest.raises(ValueError, match="lookup_value"):
        ExactIdentifierTerm(
            identifier_type=ProductIdentifierType.MPN,
            source_value="ABC",
            lookup_value="   ",
            source_field="/mpn",
        )


def test_construct_derived_envelope_with_mismatched_exact_source_ref_rejected() -> None:
    derived, source_ref = _derive_from_record(_electronics_offer_record())
    mismatched_ref = _source_ref_for_offer_id("mismatch")
    exact = ExactSearchRepresentation(
        source_ref=mismatched_ref,
        terms=derived.exact.terms,
    )
    with pytest.raises(ValueError, match="source_ref mismatch"):
        DerivedOfferSearchRepresentation(
            source_ref=source_ref,
            exact=exact,
            lexical=derived.lexical,
            structured=derived.structured,
            semantic=derived.semantic,
            derivation_version=derived.derivation_version,
        )
