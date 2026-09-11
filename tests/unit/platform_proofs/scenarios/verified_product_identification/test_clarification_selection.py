"""Unit tests for targeted clarification selection (5C11)."""

from __future__ import annotations

import random

import pytest

from platform_proofs.scenarios.verified_product_identification.application.clarification import (
    ClarificationRequirementKind,
    ClarificationSelectionRequest,
    NoClarificationReason,
    build_clarification_requirement_selection_service,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.contracts import (
    ClarificationRequirement,
    ClarificationSelectionResult,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.selection_strategy import (
    ClarificationRequirementSelectionStrategy,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    MissingDistinguishingRequirement,
    MissingRequirementOrigin,
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
from platform_proofs.scenarios.verified_product_identification.application.clarification.answerability_policy import (
    ClarificationAnswerabilityClass,
    ClarificationAnswerabilityPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.clarification.materiality_policy import (
    ClarificationMaterialityPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    NegativeAttributeConstraint,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    HypothesisVerificationState,
    IdentityHypothesisVerification,
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


def _channel_evidence() -> OfferChannelEvidence:
    return OfferChannelEvidence(
        channel=RetrievalChannel.EXACT,
        channel_rank=0,
        channel_score=None,
        reciprocal_rank_contribution=reciprocal_rank_contribution(rank=0, rrf_k=60),
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
    source_identity_facts: tuple[SourceIdentityFact, ...],
    evidence: tuple[IdentityEvidence, ...] = (),
    contradictions: tuple[IdentityContradiction, ...] = (),
) -> ProductIdentityHypothesis:
    from platform_proofs.scenarios.verified_product_identification.application.domain.source_identity import (
        source_ref_set_sha256,
    )

    ordered_refs = tuple(sorted(member_refs, key=source_ref_sort_key))
    members = tuple(
        IdentityHypothesisMember(
            source_ref=member_ref,
            fused_rank=index,
            fusion_evidence=(_channel_evidence(),),
        )
        for index, member_ref in enumerate(ordered_refs)
    )
    return ProductIdentityHypothesis(
        hypothesis_id=source_ref_set_sha256(ordered_refs),
        members=members,
        evidence=evidence,
        contradictions=contradictions,
        source_identity_facts=source_identity_facts,
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


def _ordered_refs(
    left_ref: SourceRecordRef,
    right_ref: SourceRecordRef,
) -> tuple[SourceRecordRef, SourceRecordRef]:
    return tuple(sorted((left_ref, right_ref), key=source_ref_sort_key))


def _evidence(
    left_ref: SourceRecordRef,
    right_ref: SourceRecordRef,
    *,
    evidence_type: IdentityEvidenceType,
    attribute_key: str,
    normalized_value: str,
    identifier_type: ProductIdentifierType | None = None,
    strength_class: IdentityEvidenceStrengthClass = IdentityEvidenceStrengthClass.STRONG,
) -> IdentityEvidence:
    ordered = _ordered_refs(left_ref, right_ref)
    return IdentityEvidence(
        evidence_type=evidence_type,
        source_refs=ordered,
        attribute_key=attribute_key,
        normalized_value=normalized_value,
        strength_class=strength_class,
        identifier_type=identifier_type,
        provenance=IdentityEvidenceProvenance(
            left_source_ref=ordered[0],
            right_source_ref=ordered[1],
            source_field=f"test|{attribute_key}",
            normalization_rule="test/v1",
        ),
    )


def _verify(
    ranked: RankedIdentityHypothesisCollection,
    query: ProductIdentificationQueryContext,
) -> ProductIdentificationDecision:
    service = build_product_identification_verification_service()
    outcome = service.run(
        ProductIdentificationVerificationRequest(
            ranked_hypotheses=ranked,
            query_context=query,
        )
    )
    assert outcome.decision is not None
    return outcome.decision


def _supported_verification_row(hypothesis_id: str) -> IdentityHypothesisVerification:
    return IdentityHypothesisVerification(
        hypothesis_id=hypothesis_id,
        eligible_for_verification=True,
        verification_state=HypothesisVerificationState.SUPPORTED,
        supported_requirements=(),
        contradicted_requirements=(),
        missing_requirements=(),
        blocking_contradictions=(),
        identity_evidence_sufficient=True,
    )


def _verification_rows_for_hypotheses(
    *hypotheses: ProductIdentityHypothesis,
) -> tuple[IdentityHypothesisVerification, ...]:
    return tuple(
        sorted(
            (_supported_verification_row(item.hypothesis_id) for item in hypotheses),
            key=lambda row: row.hypothesis_id,
        )
    )


def _ambiguous_decision(
    *hypotheses: ProductIdentityHypothesis,
    reason: ProductIdentificationDecisionReasonCode = ProductIdentificationDecisionReasonCode.MULTIPLE_VIABLE_IDENTITIES,
    ambiguity_candidates: tuple[str, ...] | None = None,
) -> ProductIdentificationDecision:
    ranked = _ranked(*hypotheses)
    candidate_ids = ambiguity_candidates or tuple(
        sorted(item.hypothesis.hypothesis_id for item in ranked.hypotheses)
    )
    return ProductIdentificationDecision(
        outcome=ProductIdentificationOutcome.AMBIGUOUS,
        verified_hypothesis_id=None,
        verified_member_refs=(),
        evaluated_hypotheses=ranked.hypotheses,
        decision_evidence=(),
        decision_contradicted_requirements=(),
        decision_contradictions=(),
        missing_requirements=(),
        ambiguity_candidates=candidate_ids,
        decision_reason_code=reason,
        hypothesis_verifications=_verification_rows_for_hypotheses(
            *(item.hypothesis for item in ranked.hypotheses)
        ),
    )


def _select(
    decision: ProductIdentificationDecision,
    query: ProductIdentificationQueryContext,
) -> ClarificationSelectionResult:
    service = build_clarification_requirement_selection_service()
    return service.select(
        ClarificationSelectionRequest(decision=decision, query_context=query),
    )


def _pair_hypothesis_with_facts(
    *,
    mpn: str,
    capacity: str,
    interface: str,
    refs: tuple[SourceRecordRef, SourceRecordRef],
) -> ProductIdentityHypothesis:
    ref_a, ref_b = refs
    facts = (
        _fact(ref_a, attribute_key="capacity", normalized_value=capacity),
        _fact(ref_b, attribute_key="capacity", normalized_value=capacity),
        _fact(ref_a, attribute_key="interface", normalized_value=interface),
        _fact(ref_b, attribute_key="interface", normalized_value=interface),
        _fact(
            ref_a,
            attribute_key="mpn",
            normalized_value=mpn,
            identifier_type=ProductIdentifierType.MPN,
        ),
    )
    return _hypothesis((ref_a, ref_b), source_identity_facts=facts)


def test_verified_no_clarification() -> None:
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
    query = ProductIdentificationQueryContext(
        requested_identifiers=(
            ProductIdentifier(identifier_type=ProductIdentifierType.GTIN, value="8806096660507"),
        ),
        required_constraints=(_constraint("capacity", "2TB"),),
    )
    decision = _verify(_ranked(hypothesis), query)
    result = _select(decision, query)
    assert result.clarification_required is False
    assert result.no_clarification_reason is NoClarificationReason.DECISION_ALREADY_TERMINAL


def test_no_match_no_clarification() -> None:
    hypothesis = _pair_hypothesis_with_facts(
        mpn="MZ",
        capacity="1TB",
        interface="SATA",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        )
    )
    decision = _verify(_ranked(hypothesis), query)
    result = _select(decision, query)
    assert result.clarification_required is False


def test_ambiguous_capacity_primary() -> None:
    h1 = _pair_hypothesis_with_facts(
        mpn="MZ-V9P1T0",
        capacity="1TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    h2 = _pair_hypothesis_with_facts(
        mpn="MZ-V9P2T0",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    query = ProductIdentificationQueryContext(required_constraints=(_constraint("interface", "NVMe"),))
    decision = _verify(_ranked(h1, h2), query)
    assert decision.outcome is ProductIdentificationOutcome.AMBIGUOUS
    result = _select(decision, query)
    assert result.clarification_required is True
    assert result.primary_requirement is not None
    assert result.primary_requirement.attribute_name.casefold() == "capacity"
    assert result.primary_requirement.candidate_values == ("1TB", "2TB")


def test_ambiguous_interface_primary() -> None:
    h1 = _pair_hypothesis_with_facts(
        mpn="A",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    h2 = _pair_hypothesis_with_facts(
        mpn="B",
        capacity="2TB",
        interface="SATA",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    query = ProductIdentificationQueryContext(required_constraints=(_constraint("capacity", "2TB"),))
    decision = _verify(_ranked(h1, h2), query)
    result = _select(decision, query)
    assert result.primary_requirement is not None
    assert result.primary_requirement.attribute_name.casefold() == "interface"


def test_price_difference_no_clarification() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    ref_c, ref_d = _source_ref(OFFER_C), _source_ref(OFFER_D)
    h1 = _hypothesis(
        (ref_a, ref_b),
        source_identity_facts=(
            _fact(ref_a, attribute_key="price", normalized_value="10"),
            _fact(ref_b, attribute_key="price", normalized_value="10"),
            _fact(ref_a, attribute_key="capacity", normalized_value="2TB"),
        ),
    )
    h2 = _hypothesis(
        (ref_c, ref_d),
        source_identity_facts=(
            _fact(ref_c, attribute_key="price", normalized_value="20"),
            _fact(ref_d, attribute_key="price", normalized_value="20"),
            _fact(ref_c, attribute_key="capacity", normalized_value="2TB"),
        ),
    )
    query = ProductIdentificationQueryContext(required_constraints=(_constraint("capacity", "2TB"),))
    decision = _verify(_ranked(h1, h2), query)
    result = _select(decision, query)
    assert result.clarification_required is False


def test_user_known_capacity_not_selected() -> None:
    h1 = _pair_hypothesis_with_facts(
        mpn="A",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    ref_c, ref_d = _source_ref(OFFER_C), _source_ref(OFFER_D)
    h2 = _hypothesis(
        (ref_c, ref_d),
        source_identity_facts=(
            _fact(ref_c, attribute_key="capacity", normalized_value="2TB"),
            _fact(ref_c, attribute_key="interface", normalized_value="SATA"),
        ),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        )
    )
    decision = _verify(_ranked(h1, h2), query)
    result = _select(decision, query)
    if result.clarification_required and result.primary_requirement is not None:
        assert result.primary_requirement.attribute_name.casefold() != "capacity"


def test_missing_user_requirement_capacity() -> None:
    hypothesis = _pair_hypothesis_with_facts(
        mpn="MZ",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(_constraint("interface", "NVMe"),),
        missing_user_distinguishing_requirements=(
            MissingDistinguishingRequirement(
                attribute_name="capacity",
                origin=MissingRequirementOrigin.USER,
                requirement_id="user:capacity",
            ),
        ),
    )
    decision = _verify(_ranked(hypothesis), query)
    result = _select(decision, query)
    assert result.clarification_required is True
    assert result.primary_requirement is not None
    assert result.primary_requirement.kind is ClarificationRequirementKind.USER_MISSING_FACT
    assert result.primary_requirement.attribute_name.casefold() == "capacity"


def test_catalog_gap_user_already_supplied() -> None:
    ref = _source_ref(OFFER_A)
    hypothesis = _hypothesis(
        (ref,),
        source_identity_facts=(_fact(ref, attribute_key="capacity", normalized_value="2TB"),),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        ),
    )
    decision = _verify(_ranked(hypothesis), query)
    result = _select(decision, query)
    assert result.clarification_required is False
    assert result.no_clarification_reason is NoClarificationReason.CATALOG_EVIDENCE_ONLY_GAP


def test_gtin_identifier_candidate_exists() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    h1 = _hypothesis(
        (ref_a,),
        source_identity_facts=(
            _fact(ref_a, attribute_key="gtin", normalized_value="111", identifier_type=ProductIdentifierType.GTIN),
        ),
    )
    h2 = _hypothesis(
        (ref_b,),
        source_identity_facts=(
            _fact(ref_b, attribute_key="gtin", normalized_value="222", identifier_type=ProductIdentifierType.GTIN),
        ),
    )
    query = ProductIdentificationQueryContext()
    decision = _ambiguous_decision(h1, h2)
    result = _select(decision, query)
    assert result.clarification_required is True
    ids = {result.primary_requirement.requirement_id} if result.primary_requirement else set()
    ids.update(item.requirement_id for item in result.alternate_requirements)
    assert any("gtin" in item for item in ids)


def test_sku_not_global_identifier_clarification() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    h1 = _hypothesis(
        (ref_a,),
        source_identity_facts=(
            _fact(ref_a, attribute_key="sku", normalized_value="S1", identifier_type=ProductIdentifierType.SKU),
            _fact(ref_a, attribute_key="capacity", normalized_value="1TB"),
        ),
    )
    h2 = _hypothesis(
        (ref_b,),
        source_identity_facts=(
            _fact(ref_b, attribute_key="sku", normalized_value="S2", identifier_type=ProductIdentifierType.SKU),
            _fact(ref_b, attribute_key="capacity", normalized_value="2TB"),
        ),
    )
    decision = _ambiguous_decision(h1, h2)
    result = _select(decision, ProductIdentificationQueryContext())
    assert result.primary_requirement is not None
    assert result.primary_requirement.attribute_name.casefold() == "capacity"


def test_mpn_discriminator_across_hypotheses() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    h1 = _hypothesis(
        (ref_a,),
        source_identity_facts=(
            _fact(ref_a, attribute_key="mpn", normalized_value="MPN-A", identifier_type=ProductIdentifierType.MPN),
        ),
    )
    h2 = _hypothesis(
        (ref_b,),
        source_identity_facts=(
            _fact(ref_b, attribute_key="mpn", normalized_value="MPN-B", identifier_type=ProductIdentifierType.MPN),
        ),
    )
    decision = _ambiguous_decision(h1, h2)
    result = _select(decision, ProductIdentificationQueryContext())
    assert result.clarification_required is True
    all_reqs = (result.primary_requirement,) + result.alternate_requirements
    assert any(
        item.identifier_type is ProductIdentifierType.MPN or item.attribute_name.casefold() == "mpn"
        for item in all_reqs
        if item is not None
    )


def test_three_way_capacity() -> None:
    refs = [_source_ref(OFFER_A), _source_ref(OFFER_B), _source_ref(OFFER_C)]
    hypotheses = []
    for index, capacity in enumerate(("1TB", "2TB", "4TB")):
        ref = refs[index]
        hypotheses.append(
            _hypothesis(
                (ref,),
                source_identity_facts=(
                    _fact(ref, attribute_key="capacity", normalized_value=capacity),
                    _fact(ref, attribute_key="interface", normalized_value="NVMe"),
                ),
            )
        )
    decision = _ambiguous_decision(*hypotheses)
    result = _select(decision, ProductIdentificationQueryContext())
    assert result.primary_requirement is not None
    assert result.primary_requirement.attribute_name.casefold() == "capacity"
    assert result.primary_requirement.candidate_values == ("1TB", "2TB", "4TB")


def test_partial_coverage_metadata() -> None:
    ref_a, ref_b, ref_c = _source_ref(OFFER_A), _source_ref(OFFER_B), _source_ref(OFFER_C)
    h1 = _hypothesis((ref_a,), source_identity_facts=(_fact(ref_a, attribute_key="capacity", normalized_value="1TB"),))
    h2 = _hypothesis((ref_b,), source_identity_facts=(_fact(ref_b, attribute_key="capacity", normalized_value="2TB"),))
    h3 = _hypothesis((ref_c,), source_identity_facts=())
    decision = _ambiguous_decision(h1, h2, h3)
    result = _select(decision, ProductIdentificationQueryContext())
    assert result.primary_requirement is not None
    assert result.primary_requirement.discrimination.has_complete_coverage is False


def test_complete_discriminator_outranks_partial() -> None:
    ref_a, ref_b, ref_c = _source_ref(OFFER_A), _source_ref(OFFER_B), _source_ref(OFFER_C)
    h1 = _hypothesis(
        (ref_a,),
        source_identity_facts=(
            _fact(ref_a, attribute_key="capacity", normalized_value="1TB"),
            _fact(ref_a, attribute_key="interface", normalized_value="NVMe"),
        ),
    )
    h2 = _hypothesis(
        (ref_b,),
        source_identity_facts=(
            _fact(ref_b, attribute_key="capacity", normalized_value="2TB"),
            _fact(ref_b, attribute_key="interface", normalized_value="SATA"),
        ),
    )
    h3 = _hypothesis((ref_c,), source_identity_facts=(_fact(ref_c, attribute_key="interface", normalized_value="NVMe"),))
    decision = _ambiguous_decision(h1, h2, h3)
    result = _select(decision, ProductIdentificationQueryContext())
    assert result.primary_requirement is not None
    assert result.primary_requirement.attribute_name.casefold() == "interface"
    assert result.primary_requirement.discrimination.has_complete_coverage is True


def test_weak_semantic_evidence_does_not_create_clarification() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    ref_c, ref_d = _source_ref(OFFER_C), _source_ref(OFFER_D)

    def _semantic_pair(left: SourceRecordRef, right: SourceRecordRef) -> IdentityEvidence:
        ordered = tuple(sorted((left, right), key=source_ref_sort_key))
        return IdentityEvidence(
            evidence_type=IdentityEvidenceType.SEMANTIC_SUPPORT,
            source_refs=ordered,
            attribute_key="title",
            normalized_value="samsung",
            strength_class=IdentityEvidenceStrengthClass.WEAK,
            provenance=IdentityEvidenceProvenance(
                left_source_ref=ordered[0],
                right_source_ref=ordered[1],
                source_field="title",
                normalization_rule="test/v1",
            ),
        )

    h1 = _hypothesis((ref_a, ref_b), source_identity_facts=(), evidence=(_semantic_pair(ref_a, ref_b),))
    h2 = _hypothesis((ref_c, ref_d), source_identity_facts=(), evidence=(_semantic_pair(ref_c, ref_d),))
    decision = _ambiguous_decision(h1, h2)
    result = _select(decision, ProductIdentificationQueryContext())
    assert result.clarification_required is False


def test_golden_samsung_990_ambiguous_capacity() -> None:
    h1 = _pair_hypothesis_with_facts(
        mpn="MZ-V9P1T0",
        capacity="1TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    h2 = _pair_hypothesis_with_facts(
        mpn="MZ-V9P2T0",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    query = ProductIdentificationQueryContext(required_constraints=(_constraint("interface", "NVMe"),))
    decision = _verify(_ranked(h1, h2), query)
    assert decision.outcome is ProductIdentificationOutcome.AMBIGUOUS
    result = _select(decision, query)
    assert result.primary_requirement is not None
    assert result.primary_requirement.attribute_name.casefold() == "capacity"
    assert set(result.primary_requirement.candidate_values) == {"1TB", "2TB"}
    assert len(result.primary_requirement.provenance.affected_hypothesis_ids) >= 2


def test_no_competing_identity_attribute_selected() -> None:
    h1 = _pair_hypothesis_with_facts(
        mpn="MZ-V9P2T0",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    h2 = _pair_hypothesis_with_facts(
        mpn="MZ-V9P2T0B",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        )
    )
    decision = _verify(_ranked(h1, h2), query)
    result = _select(decision, query)
    if result.primary_requirement is not None:
        assert result.primary_requirement.attribute_name.casefold() != "competing_identity"


def test_strategy_injection() -> None:
    class _LastWinsStrategy:
        def select(
            self,
            candidates: tuple[ClarificationRequirement, ...],
        ) -> tuple[ClarificationRequirement | None, tuple[ClarificationRequirement, ...]]:
            if not candidates:
                return None, ()
            ordered = tuple(sorted(candidates, key=lambda item: item.requirement_id))
            return ordered[-1], ordered[:-1]

    h1 = _pair_hypothesis_with_facts(
        mpn="A",
        capacity="1TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    h2 = _pair_hypothesis_with_facts(
        mpn="B",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    decision = _verify(_ranked(h1, h2), ProductIdentificationQueryContext())
    service = build_clarification_requirement_selection_service(
        selection_strategy=_LastWinsStrategy(),
    )
    result = service.select(
        ClarificationSelectionRequest(decision=decision, query_context=ProductIdentificationQueryContext())
    )
    assert result.primary_requirement is not None
    assert result.primary_requirement.requirement_id == max(
        item.requirement_id for item in (result.primary_requirement,) + result.alternate_requirements
    )


def test_deterministic_shuffled_hypotheses() -> None:
    h1 = _pair_hypothesis_with_facts(
        mpn="A",
        capacity="1TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    h2 = _pair_hypothesis_with_facts(
        mpn="B",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    query = ProductIdentificationQueryContext()
    results = [
        _select(_verify(_ranked(*ordering), query), query).primary_requirement
        for ordering in ((h1, h2), (h2, h1))
    ]
    assert results[0] == results[1]


def test_deterministic_shuffled_source_facts() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    facts_a = (
        _fact(ref_a, attribute_key="interface", normalized_value="NVMe"),
        _fact(ref_a, attribute_key="capacity", normalized_value="1TB"),
    )
    facts_b = (
        _fact(ref_b, attribute_key="capacity", normalized_value="2TB"),
        _fact(ref_b, attribute_key="interface", normalized_value="NVMe"),
    )
    h1 = _hypothesis((ref_a,), source_identity_facts=tuple(random.sample(list(facts_a), len(facts_a))))
    h2 = _hypothesis((ref_b,), source_identity_facts=tuple(random.sample(list(facts_b), len(facts_b))))
    first = _select(_verify(_ranked(h1, h2), ProductIdentificationQueryContext()), ProductIdentificationQueryContext())
    h1b = _hypothesis((ref_a,), source_identity_facts=tuple(reversed(facts_a)))
    h2b = _hypothesis((ref_b,), source_identity_facts=tuple(reversed(facts_b)))
    second = _select(
        _verify(_ranked(h1b, h2b), ProductIdentificationQueryContext()),
        ProductIdentificationQueryContext(),
    )
    assert first.primary_requirement == second.primary_requirement


def test_identifier_vs_attribute_prefers_capacity() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    h1 = _hypothesis(
        (ref_a,),
        source_identity_facts=(
            _fact(ref_a, attribute_key="gtin", normalized_value="111", identifier_type=ProductIdentifierType.GTIN),
            _fact(ref_a, attribute_key="capacity", normalized_value="1TB"),
        ),
    )
    h2 = _hypothesis(
        (ref_b,),
        source_identity_facts=(
            _fact(ref_b, attribute_key="gtin", normalized_value="222", identifier_type=ProductIdentifierType.GTIN),
            _fact(ref_b, attribute_key="capacity", normalized_value="2TB"),
        ),
    )
    decision = _ambiguous_decision(h1, h2)
    result = _select(decision, ProductIdentificationQueryContext())
    assert result.primary_requirement is not None
    assert result.primary_requirement.attribute_name.casefold() == "capacity"


def _three_way_unresolved_competing_query() -> ProductIdentificationQueryContext:
    return ProductIdentificationQueryContext(
        required_constraints=(
            _constraint("capacity", "2TB"),
            _constraint("interface", "NVMe"),
        )
    )


def _three_way_unresolved_competing_hypotheses() -> tuple[ProductIdentityHypothesis, ...]:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    ref_c, ref_d = _source_ref(OFFER_C), _source_ref(OFFER_D)
    ref_e, ref_f = _source_ref(ProductOfferId("offer-e")), _source_ref(ProductOfferId("offer-f"))
    h1 = _pair_hypothesis_with_facts(
        mpn="MZ-V9P2T0",
        capacity="2TB",
        interface="NVMe",
        refs=(ref_a, ref_b),
    )
    h2 = _hypothesis(
        (ref_c, ref_d),
        source_identity_facts=(),
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
    h3 = _pair_hypothesis_with_facts(
        mpn="MZ-V9P1T0",
        capacity="1TB",
        interface="SATA",
        refs=(ref_e, ref_f),
    )
    return h1, h2, h3


def _assert_h3_excluded_from_clarification(
    result: ClarificationSelectionResult,
    h3_id: str,
) -> None:
    if result.primary_requirement is not None:
        assert h3_id not in result.primary_requirement.provenance.affected_hypothesis_ids
        assert "SATA" not in result.primary_requirement.candidate_values
    for alternate in result.alternate_requirements:
        assert h3_id not in alternate.provenance.affected_hypothesis_ids
        assert "SATA" not in alternate.candidate_values


def test_rejected_competitor_excluded_from_unresolved_scope() -> None:
    h1, h2, h3 = _three_way_unresolved_competing_hypotheses()
    query = _three_way_unresolved_competing_query()
    decision = _verify(_ranked(h1, h2, h3), query)
    assert decision.decision_reason_code is ProductIdentificationDecisionReasonCode.UNRESOLVED_COMPETING_IDENTITY
    contradicted = {
        row.hypothesis_id
        for row in decision.hypothesis_verifications
        if row.verification_state is HypothesisVerificationState.CONTRADICTED
    }
    assert h3.hypothesis_id in contradicted
    result = _select(decision, query)
    _assert_h3_excluded_from_clarification(result, h3.hypothesis_id)
    if result.clarification_required and result.primary_requirement is not None:
        affected = set(result.primary_requirement.provenance.affected_hypothesis_ids)
        assert h1.hypothesis_id in affected or h2.hypothesis_id in affected
        assert h3.hypothesis_id not in affected


def test_contradicted_competitor_cannot_fabricate_interface_discriminator() -> None:
    h1, h2, h3 = _three_way_unresolved_competing_hypotheses()
    query = _three_way_unresolved_competing_query()
    decision = _verify(_ranked(h1, h2, h3), query)
    result = _select(decision, query)
    if result.primary_requirement is not None:
        assert (
            result.primary_requirement.attribute_name.casefold() != "interface"
            or "SATA" not in result.primary_requirement.candidate_values
        )
    _assert_h3_excluded_from_clarification(result, h3.hypothesis_id)
    assert result.no_clarification_reason in (
        None,
        NoClarificationReason.NO_DISCRIMINATOR_AVAILABLE,
        NoClarificationReason.NO_USER_ANSWERABLE_REQUIREMENT,
        NoClarificationReason.NO_DISCRIMINATING_FACT,
    )


def test_negative_constraint_contradicted_excluded_from_scope() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    ref_c, ref_d = _source_ref(OFFER_C), _source_ref(OFFER_D)
    ref_e, ref_f = _source_ref(ProductOfferId("offer-e")), _source_ref(ProductOfferId("offer-f"))
    h1 = _pair_hypothesis_with_facts(
        mpn="A",
        capacity="2TB",
        interface="NVMe",
        refs=(ref_a, ref_b),
    )
    h2 = _hypothesis(
        (ref_c, ref_d),
        source_identity_facts=(),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.MODEL_NUMBER_MATCH,
                attribute_key="mpn",
                normalized_value="B",
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
    h3 = _pair_hypothesis_with_facts(
        mpn="C",
        capacity="2TB",
        interface="SATA",
        refs=(ref_e, ref_f),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(_constraint("capacity", "2TB"),),
        negative_constraints=(
            NegativeAttributeConstraint(
                attribute_name="interface",
                operator=StructuredConstraintOperator.EQUALS,
                excluded_value="SATA",
            ),
        ),
    )
    decision = _verify(_ranked(h1, h2, h3), query)
    assert any(
        row.hypothesis_id == h3.hypothesis_id
        and row.verification_state is HypothesisVerificationState.CONTRADICTED
        for row in decision.hypothesis_verifications
    )
    result = _select(decision, query)
    _assert_h3_excluded_from_clarification(result, h3.hypothesis_id)


def test_ambiguous_scope_uses_only_ambiguity_candidates() -> None:
    h1 = _pair_hypothesis_with_facts(
        mpn="A",
        capacity="1TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    h2 = _pair_hypothesis_with_facts(
        mpn="B",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    h3 = _pair_hypothesis_with_facts(
        mpn="C",
        capacity="4TB",
        interface="NVMe",
        refs=(_source_ref(ProductOfferId("offer-e")), _source_ref(ProductOfferId("offer-f"))),
    )
    h4 = _pair_hypothesis_with_facts(
        mpn="D",
        capacity="8TB",
        interface="NVMe",
        refs=(_source_ref(ProductOfferId("offer-g")), _source_ref(ProductOfferId("offer-h"))),
    )
    decision = _ambiguous_decision(
        h1,
        h2,
        h3,
        h4,
        ambiguity_candidates=(h1.hypothesis_id, h2.hypothesis_id),
    )
    result = _select(decision, ProductIdentificationQueryContext())
    assert result.clarification_required is True
    assert result.primary_requirement is not None
    affected = set(result.primary_requirement.provenance.affected_hypothesis_ids)
    assert h3.hypothesis_id not in affected
    assert h4.hypothesis_id not in affected
    assert result.primary_requirement.candidate_values == ("1TB", "2TB")


class _CapacityNotAnswerablePolicy:
    def classify_attribute(self, attribute_name: str) -> ClarificationAnswerabilityClass:
        if attribute_name.casefold() == "capacity":
            return ClarificationAnswerabilityClass.NOT_USER_ANSWERABLE
        return ClarificationAnswerabilityClass.USER_NATIVE

    def classify_identifier(
        self,
        identifier_type: ProductIdentifierType,
        *,
        query_context: ProductIdentificationQueryContext,
    ) -> ClarificationAnswerabilityClass:
        return ClarificationAnswerabilityClass.NOT_USER_ANSWERABLE

    def is_selectable(self, answerability: ClarificationAnswerabilityClass) -> bool:
        return answerability in (
            ClarificationAnswerabilityClass.USER_NATIVE,
            ClarificationAnswerabilityClass.TECHNICAL_BUT_REASONABLE,
        )


class _CapacityNonMaterialPolicy:
    def is_material_attribute(self, attribute_name: str) -> bool:
        return attribute_name.casefold() != "capacity"


class _FormFactorMaterialUserNativePolicy:
    def is_material_attribute(self, attribute_name: str) -> bool:
        return attribute_name.casefold() in {"form_factor", "interface"}

    def classify_attribute(self, attribute_name: str) -> ClarificationAnswerabilityClass:
        if attribute_name.casefold() == "form_factor":
            return ClarificationAnswerabilityClass.USER_NATIVE
        return ClarificationAnswerabilityClass.NOT_USER_ANSWERABLE

    def classify_identifier(
        self,
        identifier_type: ProductIdentifierType,
        *,
        query_context: ProductIdentificationQueryContext,
    ) -> ClarificationAnswerabilityClass:
        return ClarificationAnswerabilityClass.NOT_USER_ANSWERABLE

    def is_selectable(self, answerability: ClarificationAnswerabilityClass) -> bool:
        return answerability is ClarificationAnswerabilityClass.USER_NATIVE


def test_custom_answerability_policy_controls_user_missing_path() -> None:
    hypothesis = _pair_hypothesis_with_facts(
        mpn="MZ",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    query = ProductIdentificationQueryContext(
        required_constraints=(_constraint("interface", "NVMe"),),
        missing_user_distinguishing_requirements=(
            MissingDistinguishingRequirement(
                attribute_name="capacity",
                origin=MissingRequirementOrigin.USER,
                requirement_id="user:capacity",
            ),
        ),
    )
    decision = _verify(_ranked(hypothesis), query)
    service = build_clarification_requirement_selection_service(
        answerability_policy=_CapacityNotAnswerablePolicy(),
    )
    result = service.select(ClarificationSelectionRequest(decision=decision, query_context=query))
    if result.primary_requirement is not None:
        assert result.primary_requirement.attribute_name.casefold() != "capacity"


def test_custom_materiality_policy_controls_user_and_fact_paths() -> None:
    h1 = _pair_hypothesis_with_facts(
        mpn="A",
        capacity="1TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_A), _source_ref(OFFER_B)),
    )
    h2 = _pair_hypothesis_with_facts(
        mpn="B",
        capacity="2TB",
        interface="NVMe",
        refs=(_source_ref(OFFER_C), _source_ref(OFFER_D)),
    )
    query = ProductIdentificationQueryContext(
        missing_user_distinguishing_requirements=(
            MissingDistinguishingRequirement(
                attribute_name="capacity",
                origin=MissingRequirementOrigin.USER,
                requirement_id="user:capacity",
            ),
        ),
    )
    decision = _verify(_ranked(h1, h2), query)
    service = build_clarification_requirement_selection_service(
        materiality_policy=_CapacityNonMaterialPolicy(),
    )
    result = service.select(ClarificationSelectionRequest(decision=decision, query_context=query))
    if result.primary_requirement is not None:
        assert result.primary_requirement.attribute_name.casefold() != "capacity"
    for alternate in result.alternate_requirements:
        assert alternate.attribute_name.casefold() != "capacity"


def test_custom_positive_material_answerability_policy() -> None:
    ref_a, ref_b = _source_ref(OFFER_A), _source_ref(OFFER_B)
    ref_c, ref_d = _source_ref(OFFER_C), _source_ref(OFFER_D)
    h1 = _hypothesis(
        (ref_a,),
        source_identity_facts=(
            _fact(ref_a, attribute_key="form_factor", normalized_value="M.2"),
            _fact(ref_a, attribute_key="interface", normalized_value="NVMe"),
        ),
    )
    h2 = _hypothesis(
        (ref_c,),
        source_identity_facts=(
            _fact(ref_c, attribute_key="form_factor", normalized_value="2.5"),
            _fact(ref_c, attribute_key="interface", normalized_value="NVMe"),
        ),
    )
    decision = _ambiguous_decision(h1, h2)
    service = build_clarification_requirement_selection_service(
        answerability_policy=_FormFactorMaterialUserNativePolicy(),
        materiality_policy=_FormFactorMaterialUserNativePolicy(),
    )
    result = service.select(
        ClarificationSelectionRequest(decision=decision, query_context=ProductIdentificationQueryContext())
    )
    assert result.clarification_required is True
    assert result.primary_requirement is not None
    assert result.primary_requirement.attribute_name.casefold() == "form_factor"


def test_verification_row_order_does_not_change_clarification() -> None:
    h1, h2, h3 = _three_way_unresolved_competing_hypotheses()
    query = _three_way_unresolved_competing_query()
    base = _verify(_ranked(h1, h2, h3), query)
    shuffled_rows = tuple(reversed(base.hypothesis_verifications))
    with pytest.raises(ValueError, match="hypothesis_verifications must be sorted"):
        ProductIdentificationDecision(
            outcome=base.outcome,
            verified_hypothesis_id=base.verified_hypothesis_id,
            verified_member_refs=base.verified_member_refs,
            evaluated_hypotheses=tuple(reversed(base.evaluated_hypotheses)),
            decision_evidence=base.decision_evidence,
            decision_contradicted_requirements=base.decision_contradicted_requirements,
            decision_contradictions=base.decision_contradictions,
            missing_requirements=base.missing_requirements,
            ambiguity_candidates=base.ambiguity_candidates,
            decision_reason_code=base.decision_reason_code,
            hypothesis_verifications=shuffled_rows,
        )
    first = _select(base, query)
    second = _select(_verify(_ranked(h3, h1, h2), query), query)
    assert first.primary_requirement == second.primary_requirement
    assert first.alternate_requirements == second.alternate_requirements
