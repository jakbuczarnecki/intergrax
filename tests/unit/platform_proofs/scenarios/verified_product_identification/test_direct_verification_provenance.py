"""Direct verification provenance integrity (5C10-R2)."""

from __future__ import annotations

import json

import pytest

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    NegativeAttributeConstraint,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.source_identity_fact import (
    SourceIdentityFactKind,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ProductIdentifier,
    ProductIdentifierType,
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.profile import (
    build_identity_profile,
    identifier_normalization_rule as profile_identifier_rule,
    structured_normalization_rule,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.source_identity_facts import (
    project_source_identity_facts,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.constraint_evaluation import (
    evaluate_negative_constraint,
    evaluate_required_constraint,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationOutcome,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.direct_source_evidence import (
    evaluate_requested_identifier,
)

from .test_verification_and_abstention import (
    CATALOG_ID,
    OFFER_A,
    _constraint,
    _fact,
    _hypothesis,
    _mpn_hypothesis,
    _ranked,
    _singleton_gtin_hypothesis,
    _verify,
)

pytestmark = pytest.mark.unit


def _source_ref(offer_id: ProductOfferId = OFFER_A) -> SourceRecordRef:
    return SourceRecordRef(offer_id=offer_id, catalog_id=CATALOG_ID)


def test_key_value_pairs_raw_source_value_preserved() -> None:
    ref = _source_ref()
    payload = json.dumps(
        {
            "id": OFFER_A.value,
            "keyValuePairs": {"Capacity": "2 TB"},
        }
    )
    profile = build_identity_profile(parse_wdc_source_offer_json(payload), source_ref=ref)
    capacity_attr = next(
        item
        for item in profile.structured_attributes
        if item.canonical_key.casefold() == "capacity"
    )
    assert capacity_attr.source_key == "Capacity"
    assert capacity_attr.source_value == "2 TB"
    facts = project_source_identity_facts((ref,), {ref: profile})
    capacity_fact = next(
        item for item in facts if item.attribute_key.casefold() == "capacity"
    )
    assert capacity_fact.provenance.source_field == "keyValuePairs"
    assert capacity_fact.provenance.source_value == "2 TB"
    assert capacity_fact.provenance.source_value != capacity_attr.source_key


def test_spec_table_content_raw_source_value_preserved() -> None:
    ref = _source_ref()
    payload = json.dumps(
        {
            "id": OFFER_A.value,
            "specTableContent": "Capacity: 2 TB\nInterface: NVMe",
        }
    )
    profile = build_identity_profile(parse_wdc_source_offer_json(payload), source_ref=ref)
    capacity_attr = next(
        item
        for item in profile.structured_attributes
        if item.canonical_key.casefold() == "capacity"
    )
    assert capacity_attr.source_field == "specTableContent"
    assert capacity_attr.source_value == "2 TB"
    facts = project_source_identity_facts((ref,), {ref: profile})
    capacity_fact = next(
        item for item in facts if item.attribute_key.casefold() == "capacity"
    )
    assert capacity_fact.provenance.source_field == "specTableContent"
    assert capacity_fact.provenance.source_value == "2 TB"


def test_brand_raw_source_value_preserved() -> None:
    ref = _source_ref()
    payload = json.dumps({"id": OFFER_A.value, "brand": "SaMsUnG"})
    profile = build_identity_profile(parse_wdc_source_offer_json(payload), source_ref=ref)
    assert profile.brand == "samsung"
    assert profile.brand_source_value == "SaMsUnG"
    facts = project_source_identity_facts((ref,), {ref: profile})
    brand_fact = next(item for item in facts if item.attribute_key == "brand")
    assert brand_fact.normalized_value == "samsung"
    assert brand_fact.provenance.source_value == "SaMsUnG"
    assert brand_fact.provenance.source_value != brand_fact.normalized_value


def test_identifier_fact_provenance_complete() -> None:
    ref = _source_ref()
    payload = json.dumps(
        {
            "id": OFFER_A.value,
            "identifiers": [{"/gtin13": "8806096660507"}],
        }
    )
    profile = build_identity_profile(parse_wdc_source_offer_json(payload), source_ref=ref)
    facts = project_source_identity_facts((ref,), {ref: profile})
    gtin_fact = next(item for item in facts if item.fact_kind is SourceIdentityFactKind.IDENTIFIER)
    assert gtin_fact.source_ref == ref
    assert gtin_fact.identifier_type is ProductIdentifierType.GTIN
    assert gtin_fact.normalized_value == "8806096660507"
    assert gtin_fact.provenance.source_field == "/gtin13"
    assert gtin_fact.provenance.source_value == "8806096660507"
    assert gtin_fact.provenance.normalization_rule == profile_identifier_rule()


def test_direct_identifier_support_retains_source_fact() -> None:
    ref = _source_ref()
    fact = _fact(
        ref,
        attribute_key="gtin",
        normalized_value="8806096660507",
        identifier_type=ProductIdentifierType.GTIN,
    )
    requested = ProductIdentifier(
        identifier_type=ProductIdentifierType.GTIN,
        value="8806096660507",
    )
    status, ok_row, _, _ = evaluate_requested_identifier(requested=requested, facts=(fact,))
    assert status == "supported"
    assert ok_row is not None
    assert ok_row.supporting_source_facts == (fact,)


def test_direct_identifier_contradiction_retains_source_fact() -> None:
    ref = _source_ref()
    fact = _fact(
        ref,
        attribute_key="gtin",
        normalized_value="8806096660507",
        identifier_type=ProductIdentifierType.GTIN,
    )
    requested = ProductIdentifier(
        identifier_type=ProductIdentifierType.GTIN,
        value="8806096660506",
    )
    status, _, bad_row, _ = evaluate_requested_identifier(requested=requested, facts=(fact,))
    assert status == "contradicted"
    assert bad_row is not None
    assert bad_row.contradicting_source_facts == (fact,)


def test_direct_structured_support_retains_source_facts() -> None:
    ref = _source_ref()
    fact = _fact(ref, attribute_key="capacity", normalized_value="2TB")
    hypothesis = _hypothesis((ref,), source_identity_facts=(fact,))
    constraint = StructuredAttributeConstraint(
        attribute_name="capacity",
        operator=StructuredConstraintOperator.EQUALS,
        value="2TB",
    )
    status, ok_row, _, _ = evaluate_required_constraint(hypothesis, constraint)
    assert status.value == "supported"
    assert ok_row is not None
    assert ok_row.supporting_source_facts == (fact,)


def test_direct_structured_contradiction_retains_source_facts() -> None:
    ref = _source_ref()
    fact = _fact(ref, attribute_key="capacity", normalized_value="1TB")
    hypothesis = _hypothesis((ref,), source_identity_facts=(fact,))
    constraint = StructuredAttributeConstraint(
        attribute_name="capacity",
        operator=StructuredConstraintOperator.EQUALS,
        value="2TB",
    )
    status, _, bad_row, _ = evaluate_required_constraint(hypothesis, constraint)
    assert status.value == "contradicted"
    assert bad_row is not None
    assert bad_row.contradicting_source_facts == (fact,)


def test_negative_constraint_retains_contradicting_source_fact() -> None:
    ref = _source_ref()
    fact = _fact(ref, attribute_key="interface", normalized_value="sata")
    hypothesis = _hypothesis((ref,), source_identity_facts=(fact,))
    negative = NegativeAttributeConstraint(
        attribute_name="interface",
        operator=StructuredConstraintOperator.EQUALS,
        excluded_value="SATA",
    )
    status, bad_row = evaluate_negative_constraint(hypothesis, negative)
    assert status.value == "contradicted"
    assert bad_row is not None
    assert bad_row.contradicting_source_facts == (fact,)


def test_verified_decision_retains_source_facts_without_hypothesis_search() -> None:
    query = ProductIdentificationQueryContext(
        requested_identifiers=(
            ProductIdentifier(
                identifier_type=ProductIdentifierType.GTIN,
                value="8806096660507",
            ),
        ),
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        ),
    )
    hypothesis = _singleton_gtin_hypothesis(
        "8806096660507",
        capacity="2TB",
        interface="NVMe",
    )
    decision = _verify(_ranked(hypothesis), query)
    assert decision.outcome is ProductIdentificationOutcome.VERIFIED
    by_attr = {row.attribute_name: row for row in decision.decision_evidence}
    assert by_attr["gtin"].supporting_source_facts
    assert by_attr["capacity"].supporting_source_facts
    assert by_attr["interface"].supporting_source_facts
    for row in decision.decision_evidence:
        for fact in row.supporting_source_facts:
            assert fact.source_ref.catalog_id == CATALOG_ID
            assert fact.provenance.normalization_rule.strip()


def test_no_match_decision_retains_contradicting_source_facts() -> None:
    ref = _source_ref()
    hypothesis = _hypothesis(
        (ref,),
        source_identity_facts=(_fact(ref, attribute_key="capacity", normalized_value="1TB"),),
    )
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(
            required_constraints=(_constraint("capacity", "2TB"),),
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.NO_MATCH
    assert decision.decision_contradicted_requirements
    row = decision.decision_contradicted_requirements[0]
    assert row.contradicting_source_facts
    fact = row.contradicting_source_facts[0]
    assert fact.provenance.source_value
    assert fact.normalized_value == "1TB"


def test_pair_evidence_fallback_still_populates_identity_evidence() -> None:
    hypothesis = _mpn_hypothesis("MZ-V9P2T0", capacity="2TB", interface="NVMe")
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(
            required_constraints=(
                _constraint("capacity", "2TB"),
                _constraint("interface", "NVMe"),
            )
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.VERIFIED
    assert decision.decision_evidence
    assert any(row.supporting_identity_evidence for row in decision.decision_evidence)
