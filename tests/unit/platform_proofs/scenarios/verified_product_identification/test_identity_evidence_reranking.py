"""Unit tests for identity evidence reranking and contradiction evaluation (5C9)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import cast

import pytest

from platform_proofs.scenarios.verified_product_identification.application.domain import (
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
    ContradictionRelationScope,
    EvaluatedIdentityHypothesis,
    EvidenceRelationScope,
    IdentityHypothesisEvaluationBundle,
    IdentityHypothesisEvaluationError,
    IdentityHypothesisEvaluationRequest,
    IdentityHypothesisRankingStrategy,
    InternalPairCoverage,
    build_identity_hypothesis_evaluation_service,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    compare_internal_pair_coverage,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.scope import (
    classify_contradiction_scope,
    classify_evidence_scope,
)

pytestmark = pytest.mark.unit

CATALOG_ID = "catalog-alpha"
OFFER_A = ProductOfferId("offer-a")
OFFER_B = ProductOfferId("offer-b")
OFFER_C = ProductOfferId("offer-c")
OFFER_D = ProductOfferId("offer-d")
OFFER_E = ProductOfferId("offer-e")
OFFER_F = ProductOfferId("offer-f")
OFFER_G = ProductOfferId("offer-g")
OFFER_H = ProductOfferId("offer-h")


def _source_ref(
    offer_id: ProductOfferId,
    *,
    catalog_id: str = CATALOG_ID,
) -> SourceRecordRef:
    return SourceRecordRef(offer_id=offer_id, catalog_id=catalog_id)


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
    contradiction_type: IdentityContradictionType,
    attribute_key: str,
    left_value: str,
    right_value: str,
    identifier_type: ProductIdentifierType | None = None,
) -> IdentityContradiction:
    ordered = _ordered_refs(left_ref, right_ref)
    return IdentityContradiction(
        contradiction_type=contradiction_type,
        source_refs=ordered,
        attribute_key=attribute_key,
        left_normalized_value=left_value,
        right_normalized_value=right_value,
        identifier_type=identifier_type,
        provenance=_provenance(ordered[0], ordered[1]),
    )


def _hypothesis(
    member_refs: tuple[SourceRecordRef, ...],
    *,
    evidence: tuple[IdentityEvidence, ...] = (),
    contradictions: tuple[IdentityContradiction, ...] = (),
    fused_ranks: dict[SourceRecordRef, int] | None = None,
) -> ProductIdentityHypothesis:
    ordered_refs = tuple(sorted(member_refs, key=source_ref_sort_key))
    members = tuple(
        IdentityHypothesisMember(
            source_ref=member_ref,
            fused_rank=(
                fused_ranks[member_ref]
                if fused_ranks is not None
                else index
            ),
            fusion_evidence=(_channel_evidence(RetrievalChannel.EXACT),),
        )
        for index, member_ref in enumerate(ordered_refs)
    )
    return ProductIdentityHypothesis(
        hypothesis_id=source_ref_set_sha256(ordered_refs),
        members=members,
        evidence=evidence,
        contradictions=contradictions,
    )


def _evaluate(
    *hypotheses: ProductIdentityHypothesis,
    service: object | None = None,
) -> tuple[EvaluatedIdentityHypothesis, ...]:
    evaluator = service or build_identity_hypothesis_evaluation_service()
    result = evaluator.evaluate(
        IdentityHypothesisEvaluationRequest(
            hypotheses=ProductIdentityHypothesisCollection(hypotheses=hypotheses),
        )
    )
    return result.hypotheses


def test_empty_hypothesis_collection() -> None:
    service = build_identity_hypothesis_evaluation_service()
    result = service.evaluate(
        IdentityHypothesisEvaluationRequest(
            hypotheses=ProductIdentityHypothesisCollection(hypotheses=()),
        )
    )
    assert result.hypotheses == ()


def test_singleton_hypothesis_is_valid() -> None:
    ref = _source_ref(OFFER_A)
    evaluated = _evaluate(_hypothesis((ref,)))
    assert len(evaluated) == 1
    assert evaluated[0].reranked_position == 0
    assert evaluated[0].evidence_profile.global_gtin_pair_coverage.possible_pair_count == 0


def test_gtin_support_outranks_lexical_only_hypothesis() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    gtin_hypothesis = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="gtin",
                normalized_value="8806096660507",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
    )
    weak_hypothesis = _hypothesis(
        (ref_c, ref_d),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.TITLE_TOKEN_SUPPORT,
                attribute_key="retrieval_channel",
                normalized_value="lexical",
                strength_class=IdentityEvidenceStrengthClass.WEAK,
            ),
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.SEMANTIC_SUPPORT,
                attribute_key="retrieval_channel",
                normalized_value="vector",
                strength_class=IdentityEvidenceStrengthClass.WEAK,
            ),
        ),
    )
    ranked = _evaluate(gtin_hypothesis, weak_hypothesis)
    assert ranked[0].hypothesis.hypothesis_id == gtin_hypothesis.hypothesis_id


def test_mpn_support_outranks_structured_only_when_no_gtin() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    mpn_hypothesis = _hypothesis(
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
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                attribute_key="brand",
                normalized_value="samsung",
            ),
        ),
    )
    structured_hypothesis = _hypothesis(
        (ref_c, ref_d),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="interface",
                normalized_value="nvme",
            ),
        ),
    )
    ranked = _evaluate(mpn_hypothesis, structured_hypothesis)
    assert ranked[0].hypothesis.hypothesis_id == mpn_hypothesis.hypothesis_id


def test_structured_support_outranks_weak_retrieval_only() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    structured = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
        ),
    )
    weak = _hypothesis(
        (ref_c, ref_d),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.SEMANTIC_SUPPORT,
                attribute_key="retrieval_channel",
                normalized_value="vector",
                strength_class=IdentityEvidenceStrengthClass.WEAK,
            ),
        ),
    )
    ranked = _evaluate(structured, weak)
    assert ranked[0].hypothesis.hypothesis_id == structured.hypothesis_id


def test_brand_only_does_not_outrank_structured_evidence() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    brand_only = _hypothesis(
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
    structured = _hypothesis(
        (ref_c, ref_d),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
        ),
    )
    ranked = _evaluate(brand_only, structured)
    assert ranked[0].hypothesis.hypothesis_id == structured.hypothesis_id


def test_internal_blocking_gtin_contradiction_ranks_behind_viable() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    viable = _hypothesis(
        (ref_c, ref_d),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                attribute_key="brand",
                normalized_value="samsung",
            ),
        ),
    )
    blocked = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="gtin",
                normalized_value="8806096660507",
                identifier_type=ProductIdentifierType.GTIN,
            ),
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                attribute_key="brand",
                normalized_value="samsung",
            ),
        ),
        contradictions=(
            _contradiction(
                ref_a,
                ref_b,
                contradiction_type=IdentityContradictionType.IDENTIFIER_CONFLICT,
                attribute_key="gtin",
                left_value="8806096660507",
                right_value="8806096660508",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
    )
    ranked = _evaluate(blocked, viable)
    assert ranked[0].hypothesis.hypothesis_id == viable.hypothesis_id
    assert ranked[1].ranking_key.has_internal_blocking_contradiction is True


def test_internal_blocking_brand_contradiction_ranks_behind_viable() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    viable = _hypothesis((ref_c,))
    blocked = _hypothesis(
        (ref_a, ref_b),
        contradictions=(
            _contradiction(
                ref_a,
                ref_b,
                contradiction_type=IdentityContradictionType.BRAND_CONFLICT,
                attribute_key="brand",
                left_value="samsung",
                right_value="lg",
            ),
        ),
    )
    ranked = _evaluate(blocked, viable)
    assert ranked[0].hypothesis.hypothesis_id == viable.hypothesis_id


def test_internal_nonblocking_sku_does_not_veto_stronger_identity_evidence() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    gtin_with_sku_conflict = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="gtin",
                normalized_value="8806096660507",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
        contradictions=(
            _contradiction(
                ref_a,
                ref_b,
                contradiction_type=IdentityContradictionType.IDENTIFIER_CONFLICT,
                attribute_key="sku",
                left_value="sku-a",
                right_value="sku-b",
                identifier_type=ProductIdentifierType.SKU,
            ),
        ),
    )
    brand_only = _hypothesis(
        (ref_c, ref_d),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                attribute_key="brand",
                normalized_value="samsung",
            ),
        ),
    )
    ranked = _evaluate(gtin_with_sku_conflict, brand_only)
    assert ranked[0].hypothesis.hypothesis_id == gtin_with_sku_conflict.hypothesis_id
    assert len(ranked[0].contradiction_evaluation.internal_nonblocking) == 1


def test_external_gtin_contradiction_is_not_internal_veto() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    hypothesis = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="gtin",
                normalized_value="8806096660507",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
        contradictions=(
            _contradiction(
                ref_a,
                ref_c,
                contradiction_type=IdentityContradictionType.IDENTIFIER_CONFLICT,
                attribute_key="gtin",
                left_value="8806096660507",
                right_value="8806096660508",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
    )
    evaluated = _evaluate(hypothesis)[0]
    assert evaluated.ranking_key.has_internal_blocking_contradiction is False
    assert len(evaluated.contradiction_evaluation.external_blocking) == 1
    assert evaluated.contradiction_evaluation.external_separation_count == 1


def test_external_structured_contradiction_is_not_internal_veto() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    hypothesis = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
        ),
        contradictions=(
            _contradiction(
                ref_a,
                ref_c,
                contradiction_type=IdentityContradictionType.STRUCTURED_ATTRIBUTE_CONFLICT,
                attribute_key="capacity",
                left_value="2tb",
                right_value="1tb",
            ),
        ),
    )
    evaluated = _evaluate(hypothesis)[0]
    assert evaluated.ranking_key.has_internal_blocking_contradiction is False
    assert len(evaluated.contradiction_evaluation.external_blocking) == 1


def test_invalid_contradiction_relation_fails_closed() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    hypothesis = _hypothesis(
        (ref_a, ref_b),
        contradictions=(
            _contradiction(
                ref_c,
                ref_d,
                contradiction_type=IdentityContradictionType.BRAND_CONFLICT,
                attribute_key="brand",
                left_value="samsung",
                right_value="lg",
            ),
        ),
    )
    with pytest.raises(IdentityHypothesisEvaluationError, match="contradiction row unrelated"):
        _evaluate(hypothesis)


def test_invalid_evidence_relation_fails_closed() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    hypothesis = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                attribute_key="brand",
                normalized_value="samsung",
            ),
        ),
    )
    with pytest.raises(IdentityHypothesisEvaluationError, match="evidence row unrelated"):
        _evaluate(hypothesis)


def test_raw_evidence_count_does_not_bias_ranking_by_member_count() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    ref_e = _source_ref(OFFER_E)
    ref_f = _source_ref(OFFER_F)
    small_gtin = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="gtin",
                normalized_value="8806096660507",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
    )
    large_weak_rows: list[IdentityEvidence] = []
    large_members = (ref_c, ref_d, ref_e, ref_f)
    for left_index, left_ref in enumerate(large_members):
        for right_ref in large_members[left_index + 1 :]:
            large_weak_rows.append(
                _evidence(
                    left_ref,
                    right_ref,
                    evidence_type=IdentityEvidenceType.BRAND_MATCH,
                    attribute_key="brand",
                    normalized_value="samsung",
                )
            )
            large_weak_rows.append(
                _evidence(
                    left_ref,
                    right_ref,
                    evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                    attribute_key="capacity",
                    normalized_value="2tb",
                )
            )
            large_weak_rows.append(
                _evidence(
                    left_ref,
                    right_ref,
                    evidence_type=IdentityEvidenceType.TITLE_TOKEN_SUPPORT,
                    attribute_key="retrieval_channel",
                    normalized_value="lexical",
                    strength_class=IdentityEvidenceStrengthClass.WEAK,
                )
            )
    large_hypothesis = _hypothesis(tuple(large_members), evidence=tuple(large_weak_rows))
    ranked = _evaluate(small_gtin, large_hypothesis)
    assert ranked[0].hypothesis.hypothesis_id == small_gtin.hypothesis_id


def test_duplicate_equivalent_evidence_is_not_double_counted() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    duplicate = _evidence(
        ref_a,
        ref_b,
        evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
        attribute_key="gtin",
        normalized_value="8806096660507",
        identifier_type=ProductIdentifierType.GTIN,
    )
    hypothesis = _hypothesis((ref_a, ref_b), evidence=(duplicate, duplicate))
    evaluated = _evaluate(hypothesis)[0]
    assert (
        evaluated.evidence_profile.global_gtin_pair_coverage.supported_pair_count == 1
    )


def test_weak_support_never_outranks_global_gtin() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    gtin = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="gtin",
                normalized_value="8806096660507",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
    )
    noisy_weak = _hypothesis(
        (ref_c, ref_d),
        evidence=tuple(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.TITLE_TOKEN_SUPPORT,
                attribute_key="retrieval_channel",
                normalized_value="lexical",
                strength_class=IdentityEvidenceStrengthClass.WEAK,
            )
            for _ in range(10)
        ),
    )
    ranked = _evaluate(noisy_weak, gtin)
    assert ranked[0].hypothesis.hypothesis_id == gtin.hypothesis_id


def test_fusion_score_is_not_primary_ranking_evidence() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    strong_identity = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
        ),
        fused_ranks={ref_a: 5, ref_b: 6},
    )
    better_fusion_rank = _hypothesis(
        (ref_c, ref_d),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.TITLE_TOKEN_SUPPORT,
                attribute_key="retrieval_channel",
                normalized_value="lexical",
                strength_class=IdentityEvidenceStrengthClass.WEAK,
            ),
        ),
        fused_ranks={ref_c: 0, ref_d: 1},
    )
    ranked = _evaluate(better_fusion_rank, strong_identity)
    assert ranked[0].hypothesis.hypothesis_id == strong_identity.hypothesis_id


def test_best_fused_rank_is_late_tie_break_only() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    better_fusion = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
        ),
        fused_ranks={ref_a: 0, ref_b: 1},
    )
    worse_fusion = _hypothesis(
        (ref_c, ref_d),
        evidence=(
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
        ),
        fused_ranks={ref_c: 4, ref_d: 5},
    )
    ranked = _evaluate(worse_fusion, better_fusion)
    assert ranked[0].hypothesis.hypothesis_id == better_fusion.hypothesis_id


def test_deterministic_tie_break_by_hypothesis_id() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    left = _hypothesis((ref_a,))
    right = _hypothesis((ref_c,))
    ranked = _evaluate(right, left)
    assert ranked[0].hypothesis.hypothesis_id < ranked[1].hypothesis.hypothesis_id


def test_shuffled_hypothesis_input_is_deterministic() -> None:
    hypotheses = tuple(
        _hypothesis((_source_ref(ProductOfferId(f"offer-{index}")),))
        for index in range(5)
    )
    baseline = _evaluate(*hypotheses)
    shuffled = list(hypotheses)
    random.Random(17).shuffle(shuffled)
    reranked = _evaluate(*shuffled)
    assert [item.hypothesis.hypothesis_id for item in baseline] == [
        item.hypothesis.hypothesis_id for item in reranked
    ]


def test_shuffled_evidence_order_is_deterministic() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    rows = [
        _evidence(
            ref_a,
            ref_b,
            evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
            attribute_key=f"key-{index}",
            normalized_value=f"value-{index}",
        )
        for index in range(5)
    ]
    baseline = _evaluate(_hypothesis((ref_a, ref_b), evidence=tuple(rows)))
    shuffled = list(rows)
    random.Random(23).shuffle(shuffled)
    reranked = _evaluate(_hypothesis((ref_a, ref_b), evidence=tuple(shuffled)))
    assert baseline[0].ranking_key == reranked[0].ranking_key


def test_shuffled_contradiction_order_is_deterministic() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    rows = [
        _contradiction(
            ref_a,
            ref_c,
            contradiction_type=IdentityContradictionType.STRUCTURED_ATTRIBUTE_CONFLICT,
            attribute_key=f"key-{index}",
            left_value=f"left-{index}",
            right_value=f"right-{index}",
        )
        for index in range(3)
    ]
    baseline = _evaluate(_hypothesis((ref_a, ref_b), contradictions=tuple(rows)))
    shuffled = list(rows)
    random.Random(29).shuffle(shuffled)
    reranked = _evaluate(_hypothesis((ref_a, ref_b), contradictions=tuple(shuffled)))
    assert (
        baseline[0].contradiction_evaluation.external_separation_count
        == reranked[0].contradiction_evaluation.external_separation_count
    )


def test_reranked_positions_are_contiguous_from_zero() -> None:
    hypotheses = tuple(
        _hypothesis((_source_ref(ProductOfferId(f"offer-{index}")),))
        for index in range(4)
    )
    ranked = _evaluate(*hypotheses)
    assert [item.reranked_position for item in ranked] == [0, 1, 2, 3]


def test_original_hypotheses_remain_unmodified() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
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
    evidence_snapshot = hypothesis.evidence
    members_snapshot = hypothesis.members
    _evaluate(hypothesis)
    assert hypothesis.evidence == evidence_snapshot
    assert hypothesis.members == members_snapshot


def test_no_confidence_score_fields() -> None:
    evaluated = _evaluate(_hypothesis((_source_ref(OFFER_A),)))[0]
    assert not hasattr(evaluated, "confidence")
    assert not hasattr(evaluated.ranking_key, "identity_score")


def test_golden_ranking_fixture_order() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    ref_e = _source_ref(OFFER_E)
    ref_f = _source_ref(OFFER_F)
    h1 = _hypothesis(
        (ref_a, ref_b),
        evidence=(
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
                attribute_key="gtin",
                normalized_value="8806096660507",
                identifier_type=ProductIdentifierType.GTIN,
            ),
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
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                attribute_key="brand",
                normalized_value="samsung",
            ),
            _evidence(
                ref_a,
                ref_b,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
        ),
    )
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
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                attribute_key="brand",
                normalized_value="samsung",
            ),
            _evidence(
                ref_c,
                ref_d,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
        ),
    )
    ref_g = _source_ref(OFFER_G)
    ref_h = _source_ref(OFFER_H)
    h3 = _hypothesis(
        (ref_e, ref_f),
        evidence=(
            _evidence(
                ref_e,
                ref_f,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="capacity",
                normalized_value="2tb",
            ),
            _evidence(
                ref_e,
                ref_f,
                evidence_type=IdentityEvidenceType.STRUCTURED_ATTRIBUTE_MATCH,
                attribute_key="interface",
                normalized_value="nvme",
            ),
        ),
    )
    h4 = _hypothesis(
        (ref_g, ref_h),
        evidence=(
            _evidence(
                ref_g,
                ref_h,
                evidence_type=IdentityEvidenceType.TITLE_TOKEN_SUPPORT,
                attribute_key="retrieval_channel",
                normalized_value="lexical",
                strength_class=IdentityEvidenceStrengthClass.WEAK,
            ),
            _evidence(
                ref_g,
                ref_h,
                evidence_type=IdentityEvidenceType.SEMANTIC_SUPPORT,
                attribute_key="retrieval_channel",
                normalized_value="vector",
                strength_class=IdentityEvidenceStrengthClass.WEAK,
            ),
        ),
    )
    h5 = _hypothesis(
        (_source_ref(ProductOfferId("offer-i")), _source_ref(ProductOfferId("offer-j"))),
        evidence=(
            _evidence(
                _source_ref(ProductOfferId("offer-i")),
                _source_ref(ProductOfferId("offer-j")),
                evidence_type=IdentityEvidenceType.BRAND_MATCH,
                attribute_key="brand",
                normalized_value="samsung",
            ),
        ),
        contradictions=(
            _contradiction(
                _source_ref(ProductOfferId("offer-i")),
                _source_ref(ProductOfferId("offer-j")),
                contradiction_type=IdentityContradictionType.IDENTIFIER_CONFLICT,
                attribute_key="gtin",
                left_value="8806096660507",
                right_value="8806096660508",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
    )
    ranked = _evaluate(h5, h4, h3, h2, h1)
    ordered_ids = [item.hypothesis.hypothesis_id for item in ranked]
    assert ordered_ids == [
        h1.hypothesis_id,
        h2.hypothesis_id,
        h3.hypothesis_id,
        h4.hypothesis_id,
        h5.hypothesis_id,
    ]


def test_external_contradiction_fixture_mandatory() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    hypothesis = _hypothesis(
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
        ),
        contradictions=(
            _contradiction(
                ref_a,
                ref_c,
                contradiction_type=IdentityContradictionType.IDENTIFIER_CONFLICT,
                attribute_key="gtin",
                left_value="8806096660507",
                right_value="8806096660508",
                identifier_type=ProductIdentifierType.GTIN,
            ),
        ),
    )
    member_refs = frozenset(member.source_ref for member in hypothesis.members)
    scope = classify_contradiction_scope(
        hypothesis.contradictions[0],
        member_refs=member_refs,
    )
    assert scope is ContradictionRelationScope.EXTERNAL
    evaluated = _evaluate(hypothesis)[0]
    assert evaluated.ranking_key.has_internal_blocking_contradiction is False


@dataclass(frozen=True, slots=True)
class _ReverseRankingStrategy:
    """Test-only strategy proving service independence from canonical ranking."""

    def rank(
        self,
        bundles: tuple[IdentityHypothesisEvaluationBundle, ...],
    ) -> tuple[EvaluatedIdentityHypothesis, ...]:
        reversed_bundles = tuple(reversed(bundles))
        return tuple(
            EvaluatedIdentityHypothesis(
                hypothesis=bundle.hypothesis,
                reranked_position=position,
                evidence_profile=bundle.evidence_profile,
                contradiction_evaluation=bundle.contradiction_evaluation,
                ranking_key=bundle.ranking_key,
            )
            for position, bundle in enumerate(reversed_bundles)
        )


def test_strategy_injection_is_supported() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    first = _hypothesis((ref_a,))
    second = _hypothesis((ref_b,))
    third = _hypothesis((ref_c,))
    input_order = [first.hypothesis_id, second.hypothesis_id, third.hypothesis_id]
    injected = _evaluate(
        first,
        second,
        third,
        service=build_identity_hypothesis_evaluation_service(
            strategy=cast(IdentityHypothesisRankingStrategy, _ReverseRankingStrategy()),
        ),
    )
    assert [item.hypothesis.hypothesis_id for item in injected] == list(
        reversed(input_order)
    )


def test_scope_classification_helpers() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    member_refs = frozenset({ref_a, ref_b})
    internal_evidence = _evidence(
        ref_a,
        ref_b,
        evidence_type=IdentityEvidenceType.BRAND_MATCH,
        attribute_key="brand",
        normalized_value="samsung",
    )
    external_evidence = _evidence(
        ref_a,
        ref_c,
        evidence_type=IdentityEvidenceType.BRAND_MATCH,
        attribute_key="brand",
        normalized_value="samsung",
    )
    assert (
        classify_evidence_scope(internal_evidence, member_refs=member_refs)
        is EvidenceRelationScope.INTERNAL
    )
    assert (
        classify_evidence_scope(external_evidence, member_refs=member_refs)
        is EvidenceRelationScope.EXTERNAL
    )


def test_compare_internal_pair_coverage_gtin_one_over_one_outranks_one_over_six() -> None:
    assert (
        compare_internal_pair_coverage(
            InternalPairCoverage(supported_pair_count=1, possible_pair_count=1),
            InternalPairCoverage(supported_pair_count=1, possible_pair_count=6),
        )
        == -1
    )


def test_compare_internal_pair_coverage_gtin_two_over_two_outranks_two_over_six() -> None:
    assert (
        compare_internal_pair_coverage(
            InternalPairCoverage(supported_pair_count=2, possible_pair_count=2),
            InternalPairCoverage(supported_pair_count=2, possible_pair_count=6),
        )
        == -1
    )


def test_compare_internal_pair_coverage_gtin_two_over_three_outranks_one_over_two() -> None:
    assert (
        compare_internal_pair_coverage(
            InternalPairCoverage(supported_pair_count=2, possible_pair_count=3),
            InternalPairCoverage(supported_pair_count=1, possible_pair_count=2),
        )
        == -1
    )


def test_compare_internal_pair_coverage_equal_rational_uses_supported_tie_break() -> None:
    assert (
        compare_internal_pair_coverage(
            InternalPairCoverage(supported_pair_count=1, possible_pair_count=2),
            InternalPairCoverage(supported_pair_count=2, possible_pair_count=4),
        )
        == 1
    )


def test_compare_internal_pair_coverage_complete_equal_rational_prefers_more_supported() -> None:
    assert (
        compare_internal_pair_coverage(
            InternalPairCoverage(supported_pair_count=2, possible_pair_count=2),
            InternalPairCoverage(supported_pair_count=1, possible_pair_count=1),
        )
        == -1
    )


def test_compare_internal_pair_coverage_zero_support_does_not_differentiate() -> None:
    assert (
        compare_internal_pair_coverage(
            InternalPairCoverage(supported_pair_count=0, possible_pair_count=6),
            InternalPairCoverage(supported_pair_count=0, possible_pair_count=1),
        )
        == 0
    )


def test_compare_internal_pair_coverage_singleton_zero_over_zero_has_no_positive_support() -> None:
    assert (
        compare_internal_pair_coverage(
            InternalPairCoverage(supported_pair_count=0, possible_pair_count=0),
            InternalPairCoverage(supported_pair_count=1, possible_pair_count=1),
        )
        == 1
    )


def test_compare_internal_pair_coverage_mpn_one_over_one_outranks_one_over_six() -> None:
    assert (
        compare_internal_pair_coverage(
            InternalPairCoverage(supported_pair_count=1, possible_pair_count=1),
            InternalPairCoverage(supported_pair_count=1, possible_pair_count=6),
        )
        == -1
    )


def test_compare_internal_pair_coverage_is_transitive() -> None:
    samples = (
        InternalPairCoverage(supported_pair_count=1, possible_pair_count=1),
        InternalPairCoverage(supported_pair_count=1, possible_pair_count=6),
        InternalPairCoverage(supported_pair_count=2, possible_pair_count=3),
        InternalPairCoverage(supported_pair_count=1, possible_pair_count=2),
        InternalPairCoverage(supported_pair_count=0, possible_pair_count=0),
        InternalPairCoverage(supported_pair_count=0, possible_pair_count=3),
    )
    for left in samples:
        for middle in samples:
            for right in samples:
                left_middle = compare_internal_pair_coverage(left, middle)
                middle_right = compare_internal_pair_coverage(middle, right)
                left_right = compare_internal_pair_coverage(left, right)
                if left_middle <= 0 and middle_right <= 0:
                    assert left_right <= 0
                if left_middle >= 0 and middle_right >= 0:
                    assert left_right >= 0


def _gtin_hypothesis_with_pair_coverage(
    member_refs: tuple[SourceRecordRef, ...],
    *,
    supported_pairs: tuple[tuple[SourceRecordRef, SourceRecordRef], ...],
    gtin_value: str = "8806096660507",
) -> ProductIdentityHypothesis:
    evidence = tuple(
        _evidence(
            left_ref,
            right_ref,
            evidence_type=IdentityEvidenceType.EXACT_IDENTIFIER_MATCH,
            attribute_key="gtin",
            normalized_value=gtin_value,
            identifier_type=ProductIdentifierType.GTIN,
        )
        for left_ref, right_ref in supported_pairs
    )
    return _hypothesis(member_refs, evidence=evidence)


def _mpn_hypothesis_with_pair_coverage(
    member_refs: tuple[SourceRecordRef, ...],
    *,
    supported_pairs: tuple[tuple[SourceRecordRef, SourceRecordRef], ...],
    mpn_value: str = "MZ-V9P2T0",
) -> ProductIdentityHypothesis:
    evidence = tuple(
        _evidence(
            left_ref,
            right_ref,
            evidence_type=IdentityEvidenceType.MODEL_NUMBER_MATCH,
            attribute_key="mpn",
            normalized_value=mpn_value,
            identifier_type=ProductIdentifierType.MPN,
        )
        for left_ref, right_ref in supported_pairs
    )
    return _hypothesis(member_refs, evidence=evidence)


def test_gtin_one_over_one_outranks_one_over_six_in_ranking() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    ref_e = _source_ref(OFFER_E)
    ref_f = _source_ref(OFFER_F)
    complete = _gtin_hypothesis_with_pair_coverage(
        (ref_a, ref_b),
        supported_pairs=((ref_a, ref_b),),
    )
    partial = _gtin_hypothesis_with_pair_coverage(
        (ref_c, ref_d, ref_e, ref_f),
        supported_pairs=((ref_c, ref_d),),
    )
    ranked = _evaluate(partial, complete)
    assert ranked[0].hypothesis.hypothesis_id == complete.hypothesis_id


def test_gtin_two_over_three_outranks_one_over_two_in_ranking() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    ref_e = _source_ref(OFFER_E)
    ref_f = _source_ref(OFFER_F)
    stronger = _gtin_hypothesis_with_pair_coverage(
        (ref_a, ref_b, ref_c),
        supported_pairs=((ref_a, ref_b), (ref_a, ref_c)),
    )
    weaker = _gtin_hypothesis_with_pair_coverage(
        (ref_d, ref_e, ref_f, _source_ref(ProductOfferId("offer-k"))),
        supported_pairs=((ref_d, ref_e), (ref_d, ref_f), (ref_e, ref_f)),
    )
    ranked = _evaluate(weaker, stronger)
    assert ranked[0].hypothesis.hypothesis_id == stronger.hypothesis_id


def test_gtin_complete_coverage_tie_breaks_on_supported_pairs_without_denominator_bias() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    ref_e = _source_ref(OFFER_E)
    ref_f = _source_ref(OFFER_F)
    two_member_complete = _gtin_hypothesis_with_pair_coverage(
        (ref_a, ref_b),
        supported_pairs=((ref_a, ref_b),),
    )
    three_member_complete = _gtin_hypothesis_with_pair_coverage(
        (ref_c, ref_d, ref_e),
        supported_pairs=((ref_c, ref_d), (ref_c, ref_e), (ref_d, ref_e)),
    )
    ranked = _evaluate(two_member_complete, three_member_complete)
    assert ranked[0].hypothesis.hypothesis_id == three_member_complete.hypothesis_id


def test_singleton_zero_over_zero_does_not_outrank_actual_gtin_support() -> None:
    singleton = _hypothesis((_source_ref(OFFER_A),))
    supported = _gtin_hypothesis_with_pair_coverage(
        (_source_ref(OFFER_B), _source_ref(OFFER_C)),
        supported_pairs=((_source_ref(OFFER_B), _source_ref(OFFER_C)),),
    )
    ranked = _evaluate(singleton, supported)
    assert ranked[0].hypothesis.hypothesis_id == supported.hypothesis_id


def test_mpn_one_over_one_outranks_one_over_six_in_ranking() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    ref_e = _source_ref(OFFER_E)
    ref_f = _source_ref(OFFER_F)
    complete = _mpn_hypothesis_with_pair_coverage(
        (ref_a, ref_b),
        supported_pairs=((ref_a, ref_b),),
    )
    partial = _mpn_hypothesis_with_pair_coverage(
        (ref_c, ref_d, ref_e, ref_f),
        supported_pairs=((ref_c, ref_d),),
    )
    ranked = _evaluate(partial, complete)
    assert ranked[0].hypothesis.hypothesis_id == complete.hypothesis_id


def test_anti_size_bias_fixture_h_small_outranks_h_large() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    ref_e = _source_ref(OFFER_E)
    ref_f = _source_ref(OFFER_F)
    h_small = _gtin_hypothesis_with_pair_coverage(
        (ref_a, ref_b),
        supported_pairs=((ref_a, ref_b),),
    )
    h_large = _gtin_hypothesis_with_pair_coverage(
        (ref_c, ref_d, ref_e, ref_f),
        supported_pairs=((ref_c, ref_d),),
    )
    ranked = _evaluate(h_large, h_small)
    assert ranked[0].hypothesis.hypothesis_id == h_small.hypothesis_id
    assert ranked[0].evidence_profile.global_gtin_pair_coverage.supported_pair_count == 1
    assert ranked[0].evidence_profile.global_gtin_pair_coverage.possible_pair_count == 1
    assert ranked[1].evidence_profile.global_gtin_pair_coverage.supported_pair_count == 1
    assert ranked[1].evidence_profile.global_gtin_pair_coverage.possible_pair_count == 6


def test_large_denominator_cannot_win_with_equal_supported_pairs_only() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    ref_e = _source_ref(OFFER_E)
    ref_f = _source_ref(OFFER_F)
    h_small = _gtin_hypothesis_with_pair_coverage(
        (ref_a, ref_b),
        supported_pairs=((ref_a, ref_b),),
    )
    h_large = _gtin_hypothesis_with_pair_coverage(
        (ref_c, ref_d, ref_e, ref_f),
        supported_pairs=((ref_e, ref_f),),
    )
    ranked = _evaluate(h_large, h_small)
    assert ranked[0].hypothesis.hypothesis_id == h_small.hypothesis_id


def test_member_count_remains_late_tie_break_after_equal_gtin_coverage() -> None:
    ref_a = _source_ref(OFFER_A)
    ref_b = _source_ref(OFFER_B)
    ref_c = _source_ref(OFFER_C)
    ref_d = _source_ref(OFFER_D)
    smaller_member_count = _gtin_hypothesis_with_pair_coverage(
        (ref_a, ref_b),
        supported_pairs=((ref_a, ref_b),),
    )
    larger_member_count = _gtin_hypothesis_with_pair_coverage(
        (ref_a, ref_b, ref_c, ref_d),
        supported_pairs=((ref_a, ref_b),),
    )
    ranked = _evaluate(larger_member_count, smaller_member_count)
    assert ranked[0].hypothesis.hypothesis_id == smaller_member_count.hypothesis_id
