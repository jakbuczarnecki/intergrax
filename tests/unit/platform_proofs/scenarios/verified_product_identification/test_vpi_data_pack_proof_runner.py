"""Focused proof_runner retrieval metric integration tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    LexicalSearchQuery,
    StructuredAttributeConstraint,
    StructuredConstraintOperator,
    StructuredSearchQuery,
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    ExactIdentifierLookupResult,
    LexicalSearchResult,
    StructuredSearchResult,
    VectorSearchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ExactChannelScore,
    LexicalChannelScore,
    ProductCandidate,
    ProductIdentifier,
    ProductIdentifierType,
    ProductOfferId,
    RetrievalChannel,
    SourceRecordRef,
    StructuredChannelScore,
    VectorChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.proof_runner import (
    _evaluate_query_case,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.query_set import (
    ProofQueryCase,
)

pytestmark = pytest.mark.unit

CATALOG_ID = "wdc-v2-selected"
EXPECTED_ID = "offer-expected"
RANKED_AT_4 = ("offer-a", "offer-b", "offer-c", EXPECTED_ID, "offer-e")


def _source_ref(offer_id: str) -> SourceRecordRef:
    return SourceRecordRef(
        catalog_id=CATALOG_ID,
        offer_id=ProductOfferId(offer_id),
        source_revision="rev-1",
    )


def _candidate(channel: RetrievalChannel, offer_id: str, rank: int) -> ProductCandidate:
    if channel is RetrievalChannel.EXACT:
        score = ExactChannelScore(
            matched_identifier=ProductIdentifier(
                identifier_type=ProductIdentifierType.MPN,
                value="mpn-1",
            )
        )
    elif channel is RetrievalChannel.LEXICAL:
        score = LexicalChannelScore(bm25_score=12.0)
    elif channel is RetrievalChannel.STRUCTURED:
        score = StructuredChannelScore(
            matched_constraint_count=1,
            total_constraint_count=1,
        )
    else:
        score = VectorChannelScore(cosine_similarity=0.91)
    return ProductCandidate(
        offer_id=ProductOfferId(offer_id),
        channel=channel,
        rank=rank,
        source_ref=_source_ref(offer_id),
        channel_score=score,
    )


def _candidates_for_channel(channel: RetrievalChannel) -> tuple[ProductCandidate, ...]:
    return tuple(
        _candidate(channel, offer_id, rank)
        for rank, offer_id in enumerate(RANKED_AT_4)
    )


@dataclass(frozen=True, slots=True)
class Rank4ExactLookup:
    def lookup(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
        del query
        return ExactIdentifierLookupResult(
            candidates=_candidates_for_channel(RetrievalChannel.EXACT)
        )


@dataclass(frozen=True, slots=True)
class Rank4LexicalSearch:
    def search(self, query: LexicalSearchQuery) -> LexicalSearchResult:
        del query
        return LexicalSearchResult(
            candidates=_candidates_for_channel(RetrievalChannel.LEXICAL)
        )


@dataclass(frozen=True, slots=True)
class Rank4StructuredSearch:
    def search(self, query: StructuredSearchQuery) -> StructuredSearchResult:
        del query
        return StructuredSearchResult(
            candidates=_candidates_for_channel(RetrievalChannel.STRUCTURED)
        )


@dataclass(frozen=True, slots=True)
class Rank4VectorSearch:
    def search(self, query: VectorSearchQuery) -> VectorSearchResult:
        del query
        return VectorSearchResult(
            candidates=_candidates_for_channel(RetrievalChannel.VECTOR)
        )


@dataclass(frozen=True, slots=True)
class Rank1ExactLookup:
    def lookup(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
        del query
        return ExactIdentifierLookupResult(
            candidates=(
                _candidate(RetrievalChannel.EXACT, EXPECTED_ID, 0),
                _candidate(RetrievalChannel.EXACT, "offer-b", 1),
            )
        )


@dataclass(frozen=True, slots=True)
class NegativeLexicalSearch:
    def search(self, query: LexicalSearchQuery) -> LexicalSearchResult:
        del query
        return LexicalSearchResult(
            candidates=(
                _candidate(RetrievalChannel.LEXICAL, "offer-unrelated", 0),
            )
        )


def _rank4_case(channel: str) -> ProofQueryCase:
    if channel == "exact":
        return ProofQueryCase(
            query_id="exact-rank-4",
            channel="exact",
            expected_offer_id=EXPECTED_ID,
            negative=False,
            exact_query=ExactIdentifierQuery(
                identifier=ProductIdentifier(
                    identifier_type=ProductIdentifierType.MPN,
                    value="mpn-1",
                ),
                limit=5,
            ),
        )
    if channel == "lexical":
        return ProofQueryCase(
            query_id="lexical-rank-4",
            channel="lexical",
            expected_offer_id=EXPECTED_ID,
            negative=False,
            lexical_query=LexicalSearchQuery(query_text="widget", limit=10),
        )
    if channel == "structured":
        return ProofQueryCase(
            query_id="structured-rank-4",
            channel="structured",
            expected_offer_id=EXPECTED_ID,
            negative=False,
            structured_query=StructuredSearchQuery(
                constraints=(
                    StructuredAttributeConstraint(
                        attribute_name="Voltage",
                        operator=StructuredConstraintOperator.CONTAINS,
                        value="24V",
                    ),
                ),
                limit=10,
            ),
        )
    return ProofQueryCase(
        query_id="vector-rank-4",
        channel="vector",
        expected_offer_id=EXPECTED_ID,
        negative=False,
        vector_query=VectorSearchQuery(query_text="semantic query", limit=10),
    )


def _assert_rank4_metrics(row) -> None:
    assert row.recall_at_1 == 0.0
    assert row.recall_at_5 == 1.0
    assert row.recall_at_10 == 1.0
    assert row.mrr_at_10 == 0.25
    assert row.ndcg_at_10 == pytest.approx(1.0 / math.log2(5))


@pytest.mark.parametrize("channel", ("exact", "lexical", "structured", "vector"))
def test_all_channels_share_rank4_metric_semantics(channel: str) -> None:
    row = _evaluate_query_case(
        _rank4_case(channel),
        exact_lookup=Rank4ExactLookup(),
        lexical_search=Rank4LexicalSearch(),
        structured_search=Rank4StructuredSearch(),
        vector_search=Rank4VectorSearch(),
    )
    _assert_rank4_metrics(row)


def test_exact_rank1_regression_metrics() -> None:
    case = ProofQueryCase(
        query_id="exact-rank-1",
        channel="exact",
        expected_offer_id=EXPECTED_ID,
        negative=False,
        exact_query=ExactIdentifierQuery(
            identifier=ProductIdentifier(
                identifier_type=ProductIdentifierType.MPN,
                value="mpn-1",
            ),
            limit=5,
        ),
    )
    row = _evaluate_query_case(
        case,
        exact_lookup=Rank1ExactLookup(),
        lexical_search=Rank4LexicalSearch(),
        structured_search=Rank4StructuredSearch(),
        vector_search=Rank4VectorSearch(),
    )
    assert row.recall_at_1 == 1.0
    assert row.recall_at_5 == 1.0
    assert row.recall_at_10 == 1.0
    assert row.mrr_at_10 == 1.0
    assert row.ndcg_at_10 == 1.0
    assert row.passed is True


def test_negative_lexical_query_keeps_none_metrics() -> None:
    row = _evaluate_query_case(
        ProofQueryCase(
            query_id="negative-unrelated-token",
            channel="lexical",
            expected_offer_id=EXPECTED_ID,
            negative=True,
            lexical_query=LexicalSearchQuery(
                query_text="__vpi_proof_negative_token_5c4d1__",
                limit=5,
            ),
        ),
        exact_lookup=Rank4ExactLookup(),
        lexical_search=NegativeLexicalSearch(),
        structured_search=Rank4StructuredSearch(),
        vector_search=Rank4VectorSearch(),
    )
    assert row.recall_at_1 is None
    assert row.recall_at_5 is None
    assert row.recall_at_10 is None
    assert row.mrr_at_10 is None
    assert row.ndcg_at_10 is None
    assert row.passed is True
