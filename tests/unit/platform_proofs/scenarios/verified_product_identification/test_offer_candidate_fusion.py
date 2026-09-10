"""Unit tests for offer-level candidate fusion (5C7)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ChannelCandidateBatch,
    ExactChannelScore,
    LexicalChannelScore,
    MultiChannelCandidateCollection,
    ProductCandidate,
    ProductIdentifier,
    ProductIdentifierType,
    ProductOfferId,
    RetrievalChannel,
    SourceRecordRef,
    StructuredChannelScore,
    VectorChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion import (
    FusedOfferCandidateCollection,
    OfferCandidateFusionError,
    OfferCandidateFusionRequest,
    OfferCandidateFusionService,
    OfferCandidateFusionStrategy,
    OfferFusionConfiguration,
    ReciprocalRankFusionStrategy,
    build_offer_candidate_fusion,
    reciprocal_rank_contribution,
)

pytestmark = pytest.mark.unit

CATALOG_ALPHA = "catalog-alpha"
CATALOG_BETA = "catalog-beta"
OFFER_A = ProductOfferId("offer-a")
OFFER_B = ProductOfferId("offer-b")
OFFER_C = ProductOfferId("offer-c")
OFFER_Z = ProductOfferId("offer-z")
OFFER_Y = ProductOfferId("offer-y")
DEFAULT_K = 60


def _source_ref(
    offer_id: ProductOfferId,
    *,
    catalog_id: str = CATALOG_ALPHA,
    source_revision: str | None = None,
) -> SourceRecordRef:
    return SourceRecordRef(
        offer_id=offer_id,
        catalog_id=catalog_id,
        source_revision=source_revision,
    )


def _candidate(
    *,
    offer_id: ProductOfferId,
    channel: RetrievalChannel,
    rank: int,
    catalog_id: str = CATALOG_ALPHA,
    source_revision: str | None = None,
    channel_score: ExactChannelScore
    | LexicalChannelScore
    | StructuredChannelScore
    | VectorChannelScore
    | None = None,
) -> ProductCandidate:
    return ProductCandidate(
        offer_id=offer_id,
        channel=channel,
        rank=rank,
        source_ref=_source_ref(
            offer_id,
            catalog_id=catalog_id,
            source_revision=source_revision,
        ),
        channel_score=channel_score,
    )


def _collection(*candidates: ProductCandidate) -> MultiChannelCandidateCollection:
    return MultiChannelCandidateCollection(candidates=candidates)


def _fuse(
    collection: MultiChannelCandidateCollection,
    *,
    limit: int = 20,
    rrf_k: int = DEFAULT_K,
) -> FusedOfferCandidateCollection:
    service = build_offer_candidate_fusion(
        configuration=OfferFusionConfiguration(rrf_k=rrf_k),
    )
    return service.fuse(OfferCandidateFusionRequest(candidates=collection, limit=limit))


def test_empty_collection_returns_empty_result() -> None:
    result = _fuse(_collection(), limit=5)
    assert result.candidates == ()


def test_single_channel_preserves_order_through_rrf() -> None:
    collection = _collection(
        _candidate(
            offer_id=OFFER_B,
            channel=RetrievalChannel.LEXICAL,
            rank=0,
            channel_score=LexicalChannelScore(bm25_score=12.5),
        ),
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.LEXICAL,
            rank=1,
            channel_score=LexicalChannelScore(bm25_score=99.0),
        ),
    )
    result = _fuse(collection)
    assert [candidate.offer_id for candidate in result.candidates] == [OFFER_B, OFFER_A]


def test_two_channels_same_offer_merge_to_one_fused_offer() -> None:
    collection = _collection(
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.LEXICAL,
            rank=1,
            channel_score=LexicalChannelScore(bm25_score=4.0),
        ),
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.STRUCTURED,
            rank=0,
            channel_score=StructuredChannelScore(
                matched_constraint_count=2,
                total_constraint_count=3,
            ),
        ),
    )
    result = _fuse(collection)
    assert len(result.candidates) == 1
    fused = result.candidates[0]
    assert fused.supporting_channel_count == 2
    assert {item.channel for item in fused.evidence} == {
        RetrievalChannel.LEXICAL,
        RetrievalChannel.STRUCTURED,
    }


def test_same_offer_across_all_four_channels() -> None:
    identifier = ProductIdentifier(
        identifier_type=ProductIdentifierType.MPN,
        value="MZ-V9P2T0BW",
    )
    collection = _collection(
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.EXACT,
            rank=0,
            channel_score=ExactChannelScore(matched_identifier=identifier),
        ),
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.LEXICAL,
            rank=2,
            channel_score=LexicalChannelScore(bm25_score=1.0),
        ),
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.STRUCTURED,
            rank=1,
            channel_score=StructuredChannelScore(
                matched_constraint_count=1,
                total_constraint_count=2,
            ),
        ),
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.VECTOR,
            rank=3,
            channel_score=VectorChannelScore(cosine_similarity=0.42),
        ),
    )
    result = _fuse(collection)
    assert len(result.candidates) == 1
    assert result.candidates[0].supporting_channel_count == 4
    assert len(result.candidates[0].evidence) == 4


def test_distinct_offers_remain_distinct() -> None:
    collection = _collection(
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.VECTOR, rank=0),
        _candidate(offer_id=OFFER_B, channel=RetrievalChannel.VECTOR, rank=1),
        _candidate(offer_id=OFFER_C, channel=RetrievalChannel.VECTOR, rank=2),
    )
    result = _fuse(collection)
    assert [candidate.offer_id for candidate in result.candidates] == [OFFER_A, OFFER_B, OFFER_C]


def test_same_offer_id_different_catalog_id_remain_distinct() -> None:
    collection = _collection(
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.LEXICAL,
            rank=0,
            catalog_id=CATALOG_ALPHA,
        ),
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.LEXICAL,
            rank=0,
            catalog_id=CATALOG_BETA,
        ),
    )
    result = _fuse(collection)
    assert len(result.candidates) == 2


def test_same_catalog_offer_different_source_revision_remain_distinct() -> None:
    collection = _collection(
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.LEXICAL,
            rank=0,
            source_revision=None,
        ),
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.LEXICAL,
            rank=0,
            source_revision="rev-2026-01",
        ),
    )
    result = _fuse(collection)
    assert len(result.candidates) == 2


def test_duplicate_same_source_within_same_channel_fails_closed() -> None:
    duplicate = _candidate(offer_id=OFFER_A, channel=RetrievalChannel.LEXICAL, rank=0)
    collection = _collection(duplicate, duplicate)
    with pytest.raises(OfferCandidateFusionError, match="duplicate source offer"):
        _fuse(collection)


def test_rrf_rank_zero_contribution() -> None:
    assert reciprocal_rank_contribution(rank=0, rrf_k=DEFAULT_K) == pytest.approx(1.0 / 61.0)


def test_rrf_rank_one_contribution() -> None:
    assert reciprocal_rank_contribution(rank=1, rrf_k=DEFAULT_K) == pytest.approx(1.0 / 62.0)


def test_rrf_contributions_sum_by_channel() -> None:
    collection = _collection(
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.LEXICAL, rank=0),
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.VECTOR, rank=2),
    )
    fused = _fuse(collection).candidates[0]
    expected = reciprocal_rank_contribution(rank=0, rrf_k=DEFAULT_K) + reciprocal_rank_contribution(
        rank=2,
        rrf_k=DEFAULT_K,
    )
    assert fused.fusion_score == pytest.approx(expected)
    assert sum(item.reciprocal_rank_contribution for item in fused.evidence) == pytest.approx(expected)


def test_raw_bm25_does_not_alter_rrf_contribution() -> None:
    low = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.LEXICAL,
                rank=0,
                channel_score=LexicalChannelScore(bm25_score=0.1),
            )
        )
    ).candidates[0].fusion_score
    high = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.LEXICAL,
                rank=0,
                channel_score=LexicalChannelScore(bm25_score=999.0),
            )
        )
    ).candidates[0].fusion_score
    assert low == pytest.approx(high)


def test_raw_cosine_does_not_alter_rrf_contribution() -> None:
    low = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.VECTOR,
                rank=1,
                channel_score=VectorChannelScore(cosine_similarity=-0.9),
            )
        )
    ).candidates[0].fusion_score
    high = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.VECTOR,
                rank=1,
                channel_score=VectorChannelScore(cosine_similarity=0.99),
            )
        )
    ).candidates[0].fusion_score
    assert low == pytest.approx(high)


def test_structured_ratio_does_not_alter_rrf_contribution() -> None:
    weak = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.STRUCTURED,
                rank=2,
                channel_score=StructuredChannelScore(
                    matched_constraint_count=1,
                    total_constraint_count=5,
                ),
            )
        )
    ).candidates[0].fusion_score
    strong = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.STRUCTURED,
                rank=2,
                channel_score=StructuredChannelScore(
                    matched_constraint_count=5,
                    total_constraint_count=5,
                ),
            )
        )
    ).candidates[0].fusion_score
    assert weak == pytest.approx(strong)


def test_exact_evidence_does_not_bypass_fusion_semantics() -> None:
    exact_only = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.EXACT,
                rank=0,
                channel_score=ExactChannelScore(
                    matched_identifier=ProductIdentifier(
                        identifier_type=ProductIdentifierType.GTIN,
                        value="8806095123456",
                    )
                ),
            )
        )
    ).candidates[0].fusion_score
    lexical_only = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_B,
                channel=RetrievalChannel.LEXICAL,
                rank=0,
                channel_score=LexicalChannelScore(bm25_score=1.0),
            )
        )
    ).candidates[0].fusion_score
    assert exact_only == pytest.approx(lexical_only)


def test_output_ranks_are_contiguous_from_zero() -> None:
    collection = _collection(
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.VECTOR, rank=0),
        _candidate(offer_id=OFFER_B, channel=RetrievalChannel.VECTOR, rank=1),
        _candidate(offer_id=OFFER_C, channel=RetrievalChannel.VECTOR, rank=2),
    )
    result = _fuse(collection)
    assert [candidate.fused_rank for candidate in result.candidates] == [0, 1, 2]


def test_output_limit_enforced() -> None:
    collection = _collection(
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.VECTOR, rank=0),
        _candidate(offer_id=OFFER_B, channel=RetrievalChannel.VECTOR, rank=1),
        _candidate(offer_id=OFFER_C, channel=RetrievalChannel.VECTOR, rank=2),
    )
    result = _fuse(collection, limit=2)
    assert len(result.candidates) == 2
    assert [candidate.offer_id for candidate in result.candidates] == [OFFER_A, OFFER_B]


def test_deterministic_tie_break_by_source_identity() -> None:
    collection = _collection(
        _candidate(offer_id=OFFER_Y, channel=RetrievalChannel.LEXICAL, rank=0),
        _candidate(offer_id=OFFER_Z, channel=RetrievalChannel.VECTOR, rank=0),
    )
    result = _fuse(collection)
    assert result.candidates[0].fusion_score == pytest.approx(result.candidates[1].fusion_score)
    assert [candidate.offer_id for candidate in result.candidates] == [OFFER_Y, OFFER_Z]


def test_deterministic_evidence_order() -> None:
    collection = _collection(
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.VECTOR, rank=1),
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.STRUCTURED, rank=2),
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.LEXICAL, rank=3),
        _candidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.EXACT,
            rank=0,
            channel_score=ExactChannelScore(
                matched_identifier=ProductIdentifier(
                    identifier_type=ProductIdentifierType.SKU,
                    value="SKU-1",
                )
            ),
        ),
    )
    evidence_channels = [item.channel for item in _fuse(collection).candidates[0].evidence]
    assert evidence_channels == [
        RetrievalChannel.EXACT,
        RetrievalChannel.LEXICAL,
        RetrievalChannel.STRUCTURED,
        RetrievalChannel.VECTOR,
    ]


def test_supporting_channel_count_correct() -> None:
    collection = _collection(
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.LEXICAL, rank=0),
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.STRUCTURED, rank=1),
        _candidate(offer_id=OFFER_A, channel=RetrievalChannel.VECTOR, rank=2),
        _candidate(offer_id=OFFER_B, channel=RetrievalChannel.VECTOR, rank=0),
    )
    by_offer = {candidate.offer_id: candidate for candidate in _fuse(collection).candidates}
    assert by_offer[OFFER_A].supporting_channel_count == 3
    assert by_offer[OFFER_B].supporting_channel_count == 1


def test_original_product_candidates_unchanged() -> None:
    original = _candidate(
        offer_id=OFFER_A,
        channel=RetrievalChannel.LEXICAL,
        rank=0,
        channel_score=LexicalChannelScore(bm25_score=3.5),
    )
    collection = _collection(original)
    _fuse(collection)
    assert collection.candidates[0].rank == 0
    assert collection.candidates[0].channel_score == LexicalChannelScore(bm25_score=3.5)


def test_golden_fusion_fixture_ordering() -> None:
    identifier = ProductIdentifier(
        identifier_type=ProductIdentifierType.MPN,
        value="MPN-A",
    )
    batches = (
        ChannelCandidateBatch(
            channel=RetrievalChannel.EXACT,
            candidates=(
                _candidate(
                    offer_id=OFFER_A,
                    channel=RetrievalChannel.EXACT,
                    rank=0,
                    channel_score=ExactChannelScore(matched_identifier=identifier),
                ),
            ),
        ),
        ChannelCandidateBatch(
            channel=RetrievalChannel.LEXICAL,
            candidates=(
                _candidate(
                    offer_id=OFFER_B,
                    channel=RetrievalChannel.LEXICAL,
                    rank=0,
                    channel_score=LexicalChannelScore(bm25_score=10.0),
                ),
                _candidate(
                    offer_id=OFFER_A,
                    channel=RetrievalChannel.LEXICAL,
                    rank=1,
                    channel_score=LexicalChannelScore(bm25_score=9.0),
                ),
                _candidate(
                    offer_id=OFFER_C,
                    channel=RetrievalChannel.LEXICAL,
                    rank=2,
                    channel_score=LexicalChannelScore(bm25_score=8.0),
                ),
            ),
        ),
        ChannelCandidateBatch(
            channel=RetrievalChannel.STRUCTURED,
            candidates=(
                _candidate(
                    offer_id=OFFER_A,
                    channel=RetrievalChannel.STRUCTURED,
                    rank=0,
                    channel_score=StructuredChannelScore(
                        matched_constraint_count=2,
                        total_constraint_count=2,
                    ),
                ),
                _candidate(
                    offer_id=OFFER_B,
                    channel=RetrievalChannel.STRUCTURED,
                    rank=1,
                    channel_score=StructuredChannelScore(
                        matched_constraint_count=1,
                        total_constraint_count=2,
                    ),
                ),
            ),
        ),
        ChannelCandidateBatch(
            channel=RetrievalChannel.VECTOR,
            candidates=(
                _candidate(
                    offer_id=OFFER_C,
                    channel=RetrievalChannel.VECTOR,
                    rank=0,
                    channel_score=VectorChannelScore(cosine_similarity=0.95),
                ),
                _candidate(
                    offer_id=OFFER_A,
                    channel=RetrievalChannel.VECTOR,
                    rank=1,
                    channel_score=VectorChannelScore(cosine_similarity=0.90),
                ),
            ),
        ),
    )

    def score_for(offer_id: ProductOfferId, entries: tuple[tuple[RetrievalChannel, int], ...]) -> float:
        return sum(reciprocal_rank_contribution(rank=rank, rrf_k=DEFAULT_K) for _, rank in entries)

    expected_scores = {
        OFFER_A: score_for(
            OFFER_A,
            (
                (RetrievalChannel.EXACT, 0),
                (RetrievalChannel.LEXICAL, 1),
                (RetrievalChannel.STRUCTURED, 0),
                (RetrievalChannel.VECTOR, 1),
            ),
        ),
        OFFER_B: score_for(
            OFFER_B,
            (
                (RetrievalChannel.LEXICAL, 0),
                (RetrievalChannel.STRUCTURED, 1),
            ),
        ),
        OFFER_C: score_for(
            OFFER_C,
            (
                (RetrievalChannel.LEXICAL, 2),
                (RetrievalChannel.VECTOR, 0),
            ),
        ),
    }

    ordered_input = MultiChannelCandidateCollection.from_channel_batches(*batches)
    shuffled_input = MultiChannelCandidateCollection(
        candidates=tuple(reversed(ordered_input.candidates)),
    )

    ordered_result = _fuse(ordered_input)
    shuffled_result = _fuse(shuffled_input)

    assert ordered_result.candidates == shuffled_result.candidates
    assert [candidate.offer_id for candidate in ordered_result.candidates] == [
        OFFER_A,
        OFFER_B,
        OFFER_C,
    ]
    for candidate in ordered_result.candidates:
        assert candidate.fusion_score == pytest.approx(expected_scores[candidate.offer_id])


@dataclass(frozen=True, slots=True)
class _FakeFusionStrategy:
    marker: str = "fake"

    def fuse(
        self,
        candidates: MultiChannelCandidateCollection,
        *,
        limit: int,
    ) -> FusedOfferCandidateCollection:
        del candidates, limit
        return FusedOfferCandidateCollection(candidates=())


def test_strategy_swap_proves_service_independence_from_rrf() -> None:
    service = OfferCandidateFusionService(strategy=_FakeFusionStrategy())
    result = service.fuse(
        OfferCandidateFusionRequest(
            candidates=_collection(
                _candidate(offer_id=OFFER_A, channel=RetrievalChannel.VECTOR, rank=0),
            ),
            limit=1,
        )
    )
    assert result.candidates == ()


def test_invalid_limit_rejected() -> None:
    with pytest.raises(ValueError, match="limit must be a positive int"):
        OfferCandidateFusionRequest(candidates=_collection(), limit=0)


def test_invalid_rrf_k_rejected() -> None:
    with pytest.raises(ValueError, match="rrf_k must be a positive int"):
        OfferFusionConfiguration(rrf_k=0)


def test_exact_score_preserved_in_evidence() -> None:
    identifier = ProductIdentifier(
        identifier_type=ProductIdentifierType.GTIN,
        value="123",
    )
    score = ExactChannelScore(matched_identifier=identifier)
    fused = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.EXACT,
                rank=0,
                channel_score=score,
            )
        )
    ).candidates[0]
    assert fused.evidence[0].channel_score == score


def test_lexical_bm25_preserved_in_evidence() -> None:
    score = LexicalChannelScore(bm25_score=7.25)
    fused = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.LEXICAL,
                rank=0,
                channel_score=score,
            )
        )
    ).candidates[0]
    assert fused.evidence[0].channel_score == score


def test_structured_score_preserved_in_evidence() -> None:
    score = StructuredChannelScore(matched_constraint_count=2, total_constraint_count=4)
    fused = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.STRUCTURED,
                rank=0,
                channel_score=score,
            )
        )
    ).candidates[0]
    assert fused.evidence[0].channel_score == score


def test_vector_cosine_preserved_in_evidence() -> None:
    score = VectorChannelScore(cosine_similarity=0.77)
    fused = _fuse(
        _collection(
            _candidate(
                offer_id=OFFER_A,
                channel=RetrievalChannel.VECTOR,
                rank=0,
                channel_score=score,
            )
        )
    ).candidates[0]
    assert fused.evidence[0].channel_score == score
