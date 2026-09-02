"""Unit tests for provider-neutral derived search representation."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.application.catalog import (
    build_source_record_ref,
    derive_search_representation,
    flatten_lexical_text,
    resolve_source_record,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    SourceRecordFetchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ProductIdentifierType,
    ProductOfferId,
    ProductSourceProvenance,
    ProductSourceRecord,
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
    SourceRecordRef,
    parse_wdc_source_offer_json,
)

pytestmark = pytest.mark.unit

CATALOG_ID = "catalog-neutral"
SOURCE_REVISION = "rev-neutral"


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


def _derive_from_record(record: dict[str, object]) -> tuple[object, SourceRecordRef]:
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


def test_structured_kvp_and_spec_produce_bounded_attributes() -> None:
    derived, _ = _derive_from_record(_electronics_offer_record())
    attributes = derived.structured.attributes

    assert len(attributes) == 5
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
    spec = by_key["specTableContent"]
    assert spec.source_field == "specTableContent"
    assert "Interface NVMe" in spec.source_value


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
