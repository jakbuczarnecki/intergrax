"""Unit tests for verification and abstention decisions (5C10)."""

from __future__ import annotations

import pytest

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    HypothesisRejectionEvidence,
    MissingDistinguishingRequirement,
    MissingRequirementOrigin,
    NegativeAttributeConstraint,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.source_identity_fact import (
    SourceIdentityFact,
    SourceIdentityFactKind,
    SourceIdentityFactProvenance,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ProductIdentifier,
    ProductIdentifierType,
    ProductOfferId,
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source_identity import (
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion import (
    OfferChannelEvidence,
    reciprocal_rank_contribution,
)
from platform_proofs.scenarios.verified_product_identification.application.identity import (
    IdentityContradiction,
    IdentityContradictionType,
    IdentityEvidence,
    IdentityEvidenceProvenance,
    IdentityEvidenceStrengthClass,
    IdentityEvidenceType,
    IdentityHypothesisMember,
    ProductIdentityHypothesis,
    ProductIdentityHypothesisCollection,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation import (
    IdentityHypothesisEvaluationRequest,
    RankedIdentityHypothesisCollection,
    build_identity_hypothesis_evaluation_service,
)
from platform_proofs.scenarios.verified_product_identification.application.verification import (
    ProductIdentificationOutcome,
    ProductIdentificationVerificationRequest,
    build_product_identification_verification_service,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationDecision,
    ProductIdentificationDecisionReasonCode,
)

pytestmark = pytest.mark.unit

CATALOG_ID = "catalog-alpha"
OFFER_A = ProductOfferId("offer-a")
OFFER_B = ProductOfferId("offer-b")
OFFER_C = ProductOfferId("offer-c")
OFFER_D = ProductOfferId("offer-d")


def _source_ref(offer_id: ProductOfferId) -> SourceRecordRef:
    return SourceRecordRef(offer_id=offer_id, catalog_id=CATALOG_ID)


def _ordered_refs(
    left_ref: SourceRecordRef,
    right_ref: SourceRecordRef,
) -> tuple[SourceRecordRef, SourceRecordRef]:
    return tuple(sorted((left_ref, right_ref), key=source_ref_sort_key))


def _provenance(
    left_ref: SourceRecordRef,
    right_ref: SourceRecordRef,
) -> IdentityEvidenceProvenance:
    return IdentityEvidenceProvenance(
        left_source_ref=left_ref,
        right_source_ref=right_ref,
        source_field="test|test",
        normalization_rule="test/v1",
    )


def _channel_evidence(channel: RetrievalChannel, *, rank: int = 0) -> OfferChannelEvidence:
    return OfferChannelEvidence(
        channel=channel,
        channel_rank=rank,
        channel_score=None,
        reciprocal_rank_contribution=reciprocal_rank_contribution(rank=rank, rrf_k=60),
    )


def _evidence(
    left_ref: SourceRecordRef,
    right_ref: SourceRecordRef,
    *,
    evidence_type: IdentityEvidenceType,
    attribute_key: str,
    normalized_value: str,
    strength_class: IdentityEvidenceStrengthClass = IdentityEvidenceStrengthClass.STRONG,
    identifier_type: ProductIdentifierType | None = None,
) -> IdentityEvidence:
    ordered = _ordered_refs(left_ref, right_ref)
    return IdentityEvidence(
        evidence_type=evidence_type,
        source_refs=ordered,
        attribute_key=attribute_key,
        normalized_value=normalized_value,
        strength_class=strength_class,
        identifier_type=identifier_type,
        provenance=_provenance(ordered[0], ordered[1]),
    )


def _contradiction(
    left_ref: SourceRecordRef,
    right_ref: SourceRecordRef,
    *,
    attribute_key: str,
    left_value: str,
    right_value: str,
) -> IdentityContradiction:
    ordered = _ordered_refs(left_ref, right_ref)
    return IdentityContradiction(
        contradiction_type=IdentityContradictionType.STRUCTURED_ATTRIBUTE_CONFLICT,
        source_refs=ordered,
        attribute_key=attribute_key,
        left_normalized_value=left_value,
        right_normalized_value=right_value,
        provenance=_provenance(ordered[0], ordered[1]),
    )


def _fact(
    source_ref: SourceRecordRef,
    *,
    attribute_key: str,
    normalized_value: str,
    identifier_type: ProductIdentifierType | None = None,
) -> SourceIdentityFact:
    kind = (
        SourceIdentityFactKind.IDENTIFIER
        if identifier_type is not None
        else SourceIdentityFactKind.STRUCTURED_ATTRIBUTE
    )
    return SourceIdentityFact(
        source_ref=source_ref,
        fact_kind=kind,
        attribute_key=attribute_key,
        normalized_value=normalized_value,
        identifier_type=identifier_type,
        provenance=SourceIdentityFactProvenance(
            source_field=f"test|{attribute_key}",
            normalization_rule="test/v1",
            source_value=normalized_value,
        ),
    )


def _hypothesis(
    member_refs: tuple[SourceRecordRef, ...],
    *,
    evidence: tuple[IdentityEvidence, ...] = (),
    contradictions: tuple[IdentityContradiction, ...] = (),
    source_identity_facts: tuple[SourceIdentityFact, ...] | None = None,
) -> ProductIdentityHypothesis:
    ordered_refs = tuple(sorted(member_refs, key=source_ref_sort_key))
    members = tuple(
        IdentityHypothesisMember(
            source_ref=member_ref,
            fused_rank=index,
            fusion_evidence=(_channel_evidence(RetrievalChannel.EXACT),),
        )
        for index, member_ref in enumerate(ordered_refs)
    )
    return ProductIdentityHypothesis(
        hypothesis_id=source_ref_set_sha256(ordered_refs),
        members=members,
        evidence=evidence,
        contradictions=contradictions,
        source_identity_facts=source_identity_facts or (),
    )


def _ranked(*hypotheses: ProductIdentityHypothesis) -> RankedIdentityHypothesisCollection:
    service = build_identity_hypothesis_evaluation_service()
    return service.evaluate(
        IdentityHypothesisEvaluationRequest(
            hypotheses=ProductIdentityHypothesisCollection(hypotheses=hypotheses),
        )
    )


def _constraint(name: str, value: str) -> StructuredAttributeConstraint:
    return StructuredAttributeConstraint(
        attribute_name=name,
        operator=StructuredConstraintOperator.EQUALS,
        value=value,
    )


def _verify(
    ranked: RankedIdentityHypothesisCollection,
    query: ProductIdentificationQueryContext,
    *,
    rejection: tuple[HypothesisRejectionEvidence, ...] = (),
) -> ProductIdentificationDecision:
    service = build_product_identification_verification_service()
    outcome = service.run(
        ProductIdentificationVerificationRequest(
            ranked_hypotheses=ranked,
            query_context=query,
            empty_input_rejection_evidence=rejection,
        )
    )
    assert outcome.decision is not None
    return outcome.decision


def _mpn_hypothesis(
    mpn: str,
    *,
    capacity: str | None = None,
    interface: str | None = None,
    refs: tuple[SourceRecordRef, SourceRecordRef] | None = None,
) -> ProductIdentityHypothesis:
    ref_a, ref_b = refs or (_source_ref(OFFER_A), _source_ref(OFFER_B))
    evidence: list[IdentityEvidence] = [
        _evidence(
            ref_a,
            ref_b,
            evidence_type=IdentityEvidenceType.MODEL_NUMBER_MATCH,
            attribute_key="mpn",
            normalized_value=mpn,
            identifier_type=ProductIdentifierType.MPN,
        )
    ]
    if capacity is not None:
        evidence.append(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value=capacity,
            )
        )
    if interface is not None:
        evidence.append(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="interface",
                normalized_value=interface,
            )
        )
    return _hypothesis((ref_a, ref_b), evidence=tuple(evidence))


def _singleton_gtin_hypothesis(
    gtin: str,
    *,
    capacity: str | None = None,
    interface: str | None = None,
    offer: ProductOfferId = OFFER_A,
) -> ProductIdentityHypothesis:
    ref = _source_ref(offer)
    facts: list[SourceIdentityFact] = [
        _fact(ref, attribute_key="gtin", normalized_value=gtin, identifier_type=ProductIdentifierType.GTIN),
    ]
    if capacity is not None:
        facts.append(_fact(ref, attribute_key="capacity", normalized_value=capacity))
    if interface is not None:
        facts.append(_fact(ref, attribute_key="interface", normalized_value=interface))
    return _hypothesis((ref,), source_identity_facts=tuple(facts))


def test_golden_singleton_gtin_direct_verified() -> None:
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


def test_golden_false_uniqueness_supported_plus_incomplete() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    ref_c, ref_d = _source_ref(OFFER_C), _source_ref(OFFER_D)
    h1 = _mpn_hypothesis("MZ-V9P2T0", capacity="2TB", interface="NVMe", refs=(ref_a, ref_b))
    h2 = _hypothesis(
        (ref_c, ref_d),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.MODEL_NUMBER_MATCH,
                attribute_key="mpn",
                normalized_value="MZ-V9P2T0",
                identifier_type=ProductIdentifierType.MPN,
            ),
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2TB",
            ),
        ),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        )
    )
    decision = _verify(_ranked(h1, h2), query)
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION
    assert decision.decision_reason_code is ProductIdentificationDecisionReasonCode.UNRESOLVED_COMPETING_IDENTITY


def test_ambiguous_does_not_fabricate_variant_attribute() -> None:
    h1 = _mpn_hypothesis("MZ-V9P2T0", capacity="2TB", interface="NVMe")
    h2 = _mpn_hypothesis(
        "MZ-V9P2T0B",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    decision = _verify(
        _ranked(h1, h2),
        ProductIdentificationQueryContext(
            required_constraints=(
                _constraint("capacity", "2TB"),
                _constraint("interface", "NVMe"),
            )
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.AMBIGUOUS
    assert all(item.attribute_name != "variant" for item in decision.missing_requirements)


def test_requested_gtin_contradiction() -> None:
    ref = _source_ref(OFFER_A)
    hypothesis = _hypothesis(
        (ref,),
        source_identity_facts=(
            _fact(
                ref,
                attribute_key="gtin",
                normalized_value="8806096660507",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
    )
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(
            requested_identifiers=(
                ProductIdentifier(
                    identifier_type=ProductIdentifierType.GTIN,
                    value="8806096660506",
                ),
            ),
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.NO_MATCH


def test_requested_gtin_missing_incomplete() -> None:
    ref = _source_ref(OFFER_A)
    hypothesis = _hypothesis((ref,), source_identity_facts=())
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(
            requested_identifiers=(
                ProductIdentifier(
                    identifier_type=ProductIdentifierType.GTIN,
                    value="8806096660507",
                ),
            ),
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_singleton_direct_constraint_support() -> None:
    ref = _source_ref(OFFER_A)
    hypothesis = _hypothesis(
        (ref,),
        source_identity_facts=(
            _fact(ref, attribute_key="capacity", normalized_value="2TB"),
            _fact(
                ref,
                attribute_key="gtin",
                normalized_value="8806096660507",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
    )
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(
            requested_identifiers=(
                ProductIdentifier(
                    identifier_type=ProductIdentifierType.GTIN,
                    value="8806096660507",
                ),
            ),
            required_constraints=(_constraint("capacity", "2TB"),),
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.VERIFIED


def test_top_rank_incomplete_blocks_second_supported() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    ref_c, ref_d = _source_ref(OFFER_C), _source_ref(OFFER_D)
    incomplete_top = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.MODEL_NUMBER_MATCH,
                attribute_key="mpn",
                normalized_value="MZ-V9P2T0",
                identifier_type=ProductIdentifierType.MPN,
            ),
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2TB",
            ),
        ),
    )
    supported_second = _mpn_hypothesis(
        "MZ-V9P2T0",
        capacity="2TB",
        interface="NVMe",
        refs=(ref_c, ref_d),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        )
    )
    decision = _verify(_ranked(incomplete_top, supported_second), query)
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_mixed_contradicted_incomplete_not_no_match() -> None:
    h1 = _mpn_hypothesis("A", capacity="1TB", interface="SATA")
    h2 = _hypothesis(
        (_source_ref(OFFER_C), _source_ref(OFFER_D)),
        evidence=(
            _evidence(
                _source_ref(OFFER_C),
                _source_ref(OFFER_D),
                evidence_type=IdentityEvidenceType.MODEL_NUMBER_MATCH,
                attribute_key="mpn",
                normalized_value="B",
                identifier_type=ProductIdentifierType.MPN,
            ),
        ),
    )
    decision = _verify(
        _ranked(h1, h2),
        ProductIdentificationQueryContext(
            required_constraints=(
                _constraint("capacity", "2TB"),
                _constraint("interface", "NVMe"),
            )
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION
    assert decision.outcome is not ProductIdentificationOutcome.NO_MATCH


def test_golden_verified_samsung_990_pro() -> None:
    query = ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        )
    )
    h1 = _mpn_hypothesis("MZ-V9P2T0", capacity="2TB", interface="NVMe")
    h2 = _mpn_hypothesis(
        "MZ-V9P1T0",
        capacity="1TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    decision = _verify(_ranked(h2, h1), query)
    assert decision.outcome is ProductIdentificationOutcome.VERIFIED
    assert decision.verified_hypothesis_id == h1.hypothesis_id
    assert h2.hypothesis_id not in (decision.verified_hypothesis_id,)


def test_top_rank_does_not_auto_verify() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    ref_c, ref_d = _source_ref(OFFER_C), _source_ref(OFFER_D)
    weak_top = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.SEMANTIC_SUPPORT,
                attribute_key="title",
                normalized_value="samsung 990",
                strength_class=IdentityEvidenceStrengthClass.WEAK,
            ),
        ),
    )
    strong_second = _mpn_hypothesis(
        "MZ-V9P2T0",
        capacity="2TB",
        interface="NVMe",
        refs=(ref_c, ref_d),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        )
    )
    ranked = _ranked(weak_top, strong_second)
    decision = _verify(ranked, query)
    assert decision.outcome is ProductIdentificationOutcome.VERIFIED
    assert decision.verified_hypothesis_id == strong_second.hypothesis_id


def test_gtin_unique_verified() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    hypothesis = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="gtin",
                normalized_value="8806095123456",
                identifier_type=ProductIdentifierType.GTIN,
            ),
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2TB",
            ),
        ),
    )
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(required_constraints=(_constraint("capacity", "2TB"),)),
    )
    assert decision.outcome is ProductIdentificationOutcome.VERIFIED


def test_lexical_only_not_verified() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    hypothesis = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.TITLE_TOKEN_SUPPORT,
                attribute_key="title",
                normalized_value="samsung",
                strength_class=IdentityEvidenceStrengthClass.WEAK,
            ),
        ),
    )
    decision = _verify(_ranked(hypothesis), ProductIdentificationQueryContext())
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_brand_only_not_verified() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    hypothesis = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                attribute_key="brand",
                normalized_value="samsung",
            ),
        ),
    )
    decision = _verify(_ranked(hypothesis), ProductIdentificationQueryContext())
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_blocking_contradiction_not_verified() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    hypothesis = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="gtin",
                normalized_value="8806095123456",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
        contradictions=(
            _contradiction(ref_a, ref_b, attribute_key="gtin", left_value="1", right_value="2"),
        ),
    )
    decision = _verify(_ranked(hypothesis), ProductIdentificationQueryContext())
    assert decision.outcome is ProductIdentificationOutcome.NO_MATCH


def test_required_constraint_contradiction_no_match() -> None:
    hypothesis = _mpn_hypothesis("MZ-V9P1T0", capacity="1TB", interface="NVMe")
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(required_constraints=(_constraint("capacity", "2TB"),)),
    )
    assert decision.outcome is ProductIdentificationOutcome.NO_MATCH


def test_required_constraint_missing_insufficient() -> None:
    hypothesis = _mpn_hypothesis("MZ-V9P2T0", capacity="2TB", interface=None)
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(
            required_constraints=(
                _constraint("capacity", "2TB"),
                _constraint("interface", "NVMe"),
            )
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_golden_ambiguous_two_variants() -> None:
    h1 = _mpn_hypothesis("MZ-V9P2T0", capacity="2TB", interface="NVMe")
    h2 = _mpn_hypothesis(
        "MZ-V9P2T0B",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    decision = _verify(
        _ranked(h1, h2),
        ProductIdentificationQueryContext(
            required_constraints=(
                _constraint("capacity", "2TB"),
                _constraint("interface", "NVMe"),
            )
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.AMBIGUOUS
    assert len(decision.ambiguity_candidates) >= 2


def test_golden_no_match_all_contradicted() -> None:
    h1 = _mpn_hypothesis("A", capacity="1TB", interface="SATA")
    h2 = _mpn_hypothesis(
        "B",
        capacity="1TB",
        interface="SATA",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    decision = _verify(
        _ranked(h1, h2),
        ProductIdentificationQueryContext(
            required_constraints=(
                _constraint("capacity", "2TB"),
                _constraint("interface", "NVMe"),
            )
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.NO_MATCH


def test_empty_without_rejection_not_no_match() -> None:
    decision = _verify(
        RankedIdentityHypothesisCollection(hypotheses=()),
        ProductIdentificationQueryContext(),
    )
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_empty_with_rejection_is_no_match() -> None:
    decision = _verify(
        RankedIdentityHypothesisCollection(hypotheses=()),
        ProductIdentificationQueryContext(),
        rejection=(
            HypothesisRejectionEvidence(
                rejection_reason_code="capacity_mismatch",
                attribute_key="capacity",
                catalog_value="1TB",
                source_field="spec|capacity",
            ),
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.NO_MATCH


def test_negative_constraint_violation() -> None:
    hypothesis = _mpn_hypothesis("X", capacity="2TB", interface="SATA")
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(
            required_constraints=(_constraint("capacity", "2TB"),),
            negative_constraints=(
                NegativeAttributeConstraint(
                    attribute_name="interface",
                    operator=StructuredConstraintOperator.EQUALS,
                    excluded_value="SATA",
                ),
            ),
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.NO_MATCH


def test_soft_preference_does_not_verify() -> None:
    hypothesis = _mpn_hypothesis("MZ", capacity="2TB", interface="NVMe")
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(
            required_constraints=(
                _constraint("capacity", "2TB"),
                _constraint("interface", "NVMe"),
            ),
            soft_preferences=(_constraint("brand", "Samsung"),),
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.VERIFIED


def test_sku_only_cross_catalog_not_verified() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    hypothesis = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="sku",
                normalized_value="SKU-1",
                identifier_type=ProductIdentifierType.SKU,
            ),
        ),
    )
    decision = _verify(_ranked(hypothesis), ProductIdentificationQueryContext())
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_missing_user_fact_insufficient() -> None:
    hypothesis = _mpn_hypothesis("MZ", capacity="2TB", interface="NVMe")
    decision = _verify(
        _ranked(hypothesis),
        ProductIdentificationQueryContext(
            required_constraints=(_constraint("capacity", "2TB"),),
            missing_user_distinguishing_requirements=(
                MissingDistinguishingRequirement(
                    attribute_name="interface",
                    origin=MissingRequirementOrigin.USER,
                    requirement_id="user:interface",
                ),
            ),
        ),
    )
    assert decision.outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION


def test_verified_invariants_reject_invalid() -> None:
    with pytest.raises(ValueError):
        ProductIdentificationDecision(
            outcome=ProductIdentificationOutcome.VERIFIED,
            verified_hypothesis_id=None,
            verified_member_refs=(),
            evaluated_hypotheses=(),
            decision_evidence=(),
            decision_contradicted_requirements=(),
            decision_contradictions=(),
            missing_requirements=(),
            ambiguity_candidates=(),
            decision_reason_code=ProductIdentificationDecisionReasonCode.UNIQUE_IDENTITY_SUPPORTED,
        )


def test_deterministic_shuffled_hypotheses() -> None:
    h1 = _mpn_hypothesis("MZ-V9P2T0", capacity="2TB", interface="NVMe")
    h2 = _mpn_hypothesis(
        "MZ-V9P1T0",
        capacity="1TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        )
    )
    orderings = [(h1, h2), (h2, h1)]
    results = [_verify(_ranked(*ordering), query).verified_hypothesis_id for ordering in orderings]
    assert results[0] == results[1] == h1.hypothesis_id


def test_deterministic_shuffled_evidence_rows() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    row_a = _evidence(
        ref_a,
        ref_b,
        evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
        attribute_key="capacity",
        normalized_value="2TB",
    )
    row_b = _evidence(
        ref_a,
        ref_b,
        evidence_type=IdentityEvidenceType.MODEL_NUMBER_MATCH,
        attribute_key="mpn",
        normalized_value="MZ-V9P2T0",
        identifier_type=ProductIdentifierType.MPN,
    )
    for evidence in ((row_a, row_b), (row_b, row_a)):
        hypothesis = _hypothesis((ref_a, ref_b), evidence=evidence)
        decision = _verify(
            _ranked(hypothesis),
            ProductIdentificationQueryContext(required_constraints=(_constraint("capacity", "2TB"),)),
        )
        assert decision.outcome is ProductIdentificationOutcome.VERIFIED
