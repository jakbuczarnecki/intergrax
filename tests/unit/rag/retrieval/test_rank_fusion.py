# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import copy

import pytest

from intergrax.rag.retrieval.fusion import (
    RankFusionConfiguration,
    RankFusionContractError,
    RankFusionResult,
    RankedRetrievalCandidate,
    RankedRetrievalChannel,
    ReciprocalRankFusionStrategy,
    reciprocal_rank_contribution,
    reciprocal_rank_fusion,
)

pytestmark = pytest.mark.unit

DEFAULT_K = 60


def test_reciprocal_rank_contribution_zero_based_ranks() -> None:
    assert reciprocal_rank_contribution(rank=0, rrf_k=DEFAULT_K) == pytest.approx(1.0 / 61.0)
    assert reciprocal_rank_contribution(rank=1, rrf_k=DEFAULT_K) == pytest.approx(1.0 / 62.0)


def test_canonical_math_vector_two_channels() -> None:
    channel_a = ("a", "b", "c")
    channel_b = ("c", "a", "d")
    fused = reciprocal_rank_fusion([channel_a, channel_b], k=DEFAULT_K)

    expected_scores = {
        "a": reciprocal_rank_contribution(rank=0, rrf_k=DEFAULT_K)
        + reciprocal_rank_contribution(rank=1, rrf_k=DEFAULT_K),
        "b": reciprocal_rank_contribution(rank=1, rrf_k=DEFAULT_K),
        "c": reciprocal_rank_contribution(rank=2, rrf_k=DEFAULT_K)
        + reciprocal_rank_contribution(rank=0, rrf_k=DEFAULT_K),
        "d": reciprocal_rank_contribution(rank=2, rrf_k=DEFAULT_K),
    }
    assert [item[0] for item in fused] == ["a", "c", "b", "d"]
    for candidate_id, score in fused:
        assert score == pytest.approx(expected_scores[candidate_id])


def test_strategy_matches_primitive_math() -> None:
    strategy = ReciprocalRankFusionStrategy(configuration=RankFusionConfiguration(rrf_k=DEFAULT_K))
    channels = (
        RankedRetrievalChannel(
            channel_key="lexical",
            candidates=(
                RankedRetrievalCandidate(candidate_id="a", rank=0),
                RankedRetrievalCandidate(candidate_id="b", rank=1),
                RankedRetrievalCandidate(candidate_id="c", rank=2),
            ),
        ),
        RankedRetrievalChannel(
            channel_key="vector",
            candidates=(
                RankedRetrievalCandidate(candidate_id="c", rank=0),
                RankedRetrievalCandidate(candidate_id="a", rank=1),
                RankedRetrievalCandidate(candidate_id="d", rank=2),
            ),
        ),
    )
    result = strategy.fuse(channels)
    assert [item.candidate_id for item in result.candidates] == ["a", "c", "b", "d"]
    assert result.candidates[0].fusion_score == pytest.approx(
        reciprocal_rank_contribution(rank=0, rrf_k=DEFAULT_K)
        + reciprocal_rank_contribution(rank=1, rrf_k=DEFAULT_K)
    )


def test_multi_channel_candidate_sums_contributions() -> None:
    strategy = ReciprocalRankFusionStrategy()
    result = strategy.fuse(
        (
            RankedRetrievalChannel(
                channel_key="one",
                candidates=(RankedRetrievalCandidate(candidate_id="x", rank=0),),
            ),
            RankedRetrievalChannel(
                channel_key="two",
                candidates=(RankedRetrievalCandidate(candidate_id="x", rank=3),),
            ),
        )
    )
    assert len(result.candidates) == 1
    expected = reciprocal_rank_contribution(rank=0, rrf_k=DEFAULT_K) + reciprocal_rank_contribution(
        rank=3,
        rrf_k=DEFAULT_K,
    )
    assert result.candidates[0].fusion_score == pytest.approx(expected)
    assert result.candidates[0].supporting_channel_count == 2
    assert {item.channel_key for item in result.candidates[0].evidence} == {"one", "two"}


def test_single_channel_single_candidate() -> None:
    strategy = ReciprocalRankFusionStrategy()
    result = strategy.fuse(
        (
            RankedRetrievalChannel(
                channel_key="only",
                candidates=(RankedRetrievalCandidate(candidate_id="solo", rank=0),),
            ),
        )
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].evidence[0].channel_rank == 0


def test_empty_channel_and_all_empty() -> None:
    strategy = ReciprocalRankFusionStrategy()
    assert strategy.fuse(()).candidates == ()
    assert (
        strategy.fuse(
            (
                RankedRetrievalChannel(channel_key="empty", candidates=()),
                RankedRetrievalChannel(
                    channel_key="filled",
                    candidates=(RankedRetrievalCandidate(candidate_id="z", rank=0),),
                ),
            )
        ).candidates[0].candidate_id
        == "z"
    )


def test_duplicate_in_channel_is_contract_violation() -> None:
    strategy = ReciprocalRankFusionStrategy()
    with pytest.raises(RankFusionContractError, match="duplicate candidate_id"):
        strategy.fuse(
            (
                RankedRetrievalChannel(
                    channel_key="dup",
                    candidates=(
                        RankedRetrievalCandidate(candidate_id="x", rank=0),
                        RankedRetrievalCandidate(candidate_id="x", rank=1),
                    ),
                ),
            )
        )


def test_primitive_dedupes_duplicate_id_in_one_channel_to_best_rank() -> None:
    fused = reciprocal_rank_fusion([("a", "a", "b")], k=DEFAULT_K)
    assert fused[0][0] == "a"
    assert fused[0][1] == pytest.approx(reciprocal_rank_contribution(rank=0, rrf_k=DEFAULT_K))


def test_tie_break_is_deterministic_by_candidate_id() -> None:
    strategy = ReciprocalRankFusionStrategy()
    result = strategy.fuse(
        (
            RankedRetrievalChannel(
                channel_key="c1",
                candidates=(
                    RankedRetrievalCandidate(candidate_id="beta", rank=0),
                    RankedRetrievalCandidate(candidate_id="alpha", rank=1),
                ),
            ),
            RankedRetrievalChannel(
                channel_key="c2",
                candidates=(
                    RankedRetrievalCandidate(candidate_id="alpha", rank=0),
                    RankedRetrievalCandidate(candidate_id="beta", rank=1),
                ),
            ),
        )
    )
    assert result.candidates[0].fusion_score == result.candidates[1].fusion_score
    assert [item.candidate_id for item in result.candidates] == ["alpha", "beta"]


def test_fusion_does_not_mutate_input_channels() -> None:
    channels = (
        RankedRetrievalChannel(
            channel_key="c1",
            candidates=(RankedRetrievalCandidate(candidate_id="a", rank=0),),
        ),
    )
    snapshot = copy.deepcopy(channels)
    ReciprocalRankFusionStrategy().fuse(channels)
    assert channels == snapshot


def test_custom_strategy_injection() -> None:
    from intergrax.rag.retrieval.fusion.contracts import FusedRankedCandidate, RankFusionChannelEvidence

    class _StubStrategy:
        @property
        def strategy_id(self) -> str:
            return "stub.v1"

        def fuse(self, channels, *, limit=None):
            return RankFusionResult(
                candidates=(
                    FusedRankedCandidate(
                        candidate_id="plugged",
                        fused_rank=0,
                        fusion_score=1.0,
                        supporting_channel_count=1,
                        evidence=(
                            RankFusionChannelEvidence(
                                channel_key="stub",
                                channel_rank=0,
                                reciprocal_rank_contribution=1.0,
                            ),
                        ),
                    ),
                )
            )

    from intergrax.rag.retrievers.providers.fusion_retriever import FusionRetriever
    from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry

    registry = RetrieverRegistry()
    retriever = FusionRetriever(
        registry=registry,
        retrievers=[],
        fusion_strategy=_StubStrategy(),
    )
    assert retriever._fusion_strategy.strategy_id == "stub.v1"


def test_invalid_rrf_k_rejected() -> None:
    with pytest.raises(RankFusionContractError, match="rrf_k"):
        RankFusionConfiguration(rrf_k=0)
