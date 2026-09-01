"""Unit tests for provider-neutral catalog/search contract foundation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from platform_proofs.scenarios.verified_product_identification.application.catalog import (
    collect_channel_candidates,
    resolve_source_record,
)
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
    SourceRecordFetchResult,
    StructuredSearchResult,
    VectorSearchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ChannelCandidateBatch,
    ExactChannelScore,
    LexicalChannelScore,
    ProductCandidate,
    ProductIdentifier,
    ProductIdentifierType,
    ProductOfferId,
    ProductSourceProvenance,
    ProductSourceRecord,
    RetrievalChannel,
    SourceRecordRef,
    StructuredChannelScore,
    VectorChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.application.ports import (
    ExactIdentifierLookupPort,
    LexicalCandidateSearchPort,
    SourceRecordFetchPort,
    StructuredCandidateSearchPort,
    VectorCandidateSearchPort,
)

pytestmark = pytest.mark.unit

OFFER_A = ProductOfferId("offer-1001")
CATALOG_A = "catalog-alpha"
CATALOG_B = "catalog-beta"


def _source_ref(offer_id: ProductOfferId, *, catalog_id: str) -> SourceRecordRef:
    return SourceRecordRef(offer_id=offer_id, catalog_id=catalog_id)


def _exact_candidate(
    *,
    offer_id: ProductOfferId,
    rank: int,
    identifier: ProductIdentifier,
    catalog_id: str,
) -> ProductCandidate:
    return ProductCandidate(
        offer_id=offer_id,
        channel=RetrievalChannel.EXACT,
        rank=rank,
        source_ref=_source_ref(offer_id, catalog_id=catalog_id),
        channel_score=ExactChannelScore(matched_identifier=identifier),
    )


@dataclass(frozen=True, slots=True)
class FakeExactLookupA:
    catalog_id: str = CATALOG_A

    def lookup(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
        candidate = _exact_candidate(
            offer_id=OFFER_A,
            rank=0,
            identifier=query.identifier,
            catalog_id=self.catalog_id,
        )
        return ExactIdentifierLookupResult(candidates=(candidate,))


@dataclass(frozen=True, slots=True)
class FakeExactLookupB:
    catalog_id: str = CATALOG_B

    def lookup(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
        candidate = _exact_candidate(
            offer_id=OFFER_A,
            rank=0,
            identifier=query.identifier,
            catalog_id=self.catalog_id,
        )
        return ExactIdentifierLookupResult(candidates=(candidate,))


def _consume_exact_lookup(port: ExactIdentifierLookupPort) -> ProductCandidate:
    query = ExactIdentifierQuery(
        identifier=ProductIdentifier(
            identifier_type=ProductIdentifierType.GTIN,
            value="8806095123456",
        )
    )
    result = port.lookup(query)
    assert result.failure is None
    assert len(result.candidates) == 1
    return result.candidates[0]


def test_provider_neutrality_two_exact_lookup_implementations() -> None:
    candidate_a = _consume_exact_lookup(FakeExactLookupA())
    candidate_b = _consume_exact_lookup(FakeExactLookupB())

    assert candidate_a.offer_id == candidate_b.offer_id
    assert candidate_a.channel == RetrievalChannel.EXACT
    assert candidate_b.channel == RetrievalChannel.EXACT
    assert candidate_a.rank == candidate_b.rank == 0
    assert candidate_a.source_ref.catalog_id == CATALOG_A
    assert candidate_b.source_ref.catalog_id == CATALOG_B
    assert isinstance(candidate_a.channel_score, ExactChannelScore)
    assert isinstance(candidate_b.channel_score, ExactChannelScore)


@dataclass(frozen=True, slots=True)
class FakeSourceRecordStoreA:
    catalog_id: str = CATALOG_A

    def fetch(self, offer_id: ProductOfferId) -> SourceRecordFetchResult:
        return SourceRecordFetchResult(
            record=ProductSourceRecord(
                offer_id=offer_id,
                record_payload_ref=f"{self.catalog_id}:payload:{offer_id.value}",
                provenance=ProductSourceProvenance(catalog_id=self.catalog_id),
            )
        )


@dataclass(frozen=True, slots=True)
class FakeSourceRecordStoreB:
    catalog_id: str = CATALOG_B

    def fetch(self, offer_id: ProductOfferId) -> SourceRecordFetchResult:
        return SourceRecordFetchResult(
            record=ProductSourceRecord(
                offer_id=offer_id,
                record_payload_ref=f"{self.catalog_id}:payload:{offer_id.value}",
                provenance=ProductSourceProvenance(catalog_id=self.catalog_id),
            )
        )


def _resolve_with_store(
    port: SourceRecordFetchPort,
    candidate: ProductCandidate,
) -> ProductSourceRecord:
    return resolve_source_record(candidate, port)


def test_source_truth_boundary_candidate_to_source_fetch() -> None:
    candidate = _consume_exact_lookup(FakeExactLookupA())

    source_a = _resolve_with_store(FakeSourceRecordStoreA(), candidate)
    source_b = _resolve_with_store(FakeSourceRecordStoreB(), candidate)

    assert source_a.offer_id == candidate.offer_id
    assert source_b.offer_id == candidate.offer_id
    assert source_a.record_payload_ref != source_b.record_payload_ref
    assert source_a.provenance.catalog_id == CATALOG_A
    assert source_b.provenance.catalog_id == CATALOG_B


@dataclass(frozen=True, slots=True)
class FakeLexicalSearch:
    def search(self, query: LexicalSearchQuery) -> LexicalSearchResult:
        candidate = ProductCandidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.LEXICAL,
            rank=0,
            source_ref=_source_ref(OFFER_A, catalog_id=CATALOG_A),
            channel_score=LexicalChannelScore(bm25_score=14.2),
        )
        del query
        return LexicalSearchResult(candidates=(candidate,))


@dataclass(frozen=True, slots=True)
class FakeStructuredSearch:
    def search(self, query: StructuredSearchQuery) -> StructuredSearchResult:
        candidate = ProductCandidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.STRUCTURED,
            rank=0,
            source_ref=_source_ref(OFFER_A, catalog_id=CATALOG_A),
            channel_score=StructuredChannelScore(
                matched_constraint_count=len(query.constraints),
                total_constraint_count=len(query.constraints),
            ),
        )
        return StructuredSearchResult(candidates=(candidate,))


@dataclass(frozen=True, slots=True)
class FakeVectorSearch:
    def search(self, query: VectorSearchQuery) -> VectorSearchResult:
        candidate = ProductCandidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.VECTOR,
            rank=0,
            source_ref=_source_ref(OFFER_A, catalog_id=CATALOG_A),
            channel_score=VectorChannelScore(cosine_similarity=0.87),
        )
        del query
        return VectorSearchResult(candidates=(candidate,))


def test_multi_channel_candidate_handoff_preserves_semantics() -> None:
    exact = FakeExactLookupA().lookup(
        ExactIdentifierQuery(
            identifier=ProductIdentifier(
                identifier_type=ProductIdentifierType.MPN,
                value="MZ-V9P2T0BW",
            )
        )
    )
    lexical = FakeLexicalSearch().search(LexicalSearchQuery(query_text="990 PRO 2TB"))
    structured = FakeStructuredSearch().search(
        StructuredSearchQuery(
            constraints=(
                StructuredAttributeConstraint(
                    attribute_name="capacity",
                    operator=StructuredConstraintOperator.EQUALS,
                    value="2TB",
                ),
            )
        )
    )
    vector = FakeVectorSearch().search(VectorSearchQuery(query_text="Samsung NVMe SSD 2TB"))

    collection = collect_channel_candidates(
        ChannelCandidateBatch(channel=RetrievalChannel.EXACT, candidates=exact.candidates),
        ChannelCandidateBatch(channel=RetrievalChannel.LEXICAL, candidates=lexical.candidates),
        ChannelCandidateBatch(
            channel=RetrievalChannel.STRUCTURED,
            candidates=structured.candidates,
        ),
        ChannelCandidateBatch(channel=RetrievalChannel.VECTOR, candidates=vector.candidates),
    )

    assert len(collection.candidates) == 4
    channels = {candidate.channel for candidate in collection.candidates}
    assert channels == {
        RetrievalChannel.EXACT,
        RetrievalChannel.LEXICAL,
        RetrievalChannel.STRUCTURED,
        RetrievalChannel.VECTOR,
    }

    scores_by_channel = {candidate.channel: candidate.channel_score for candidate in collection.candidates}
    assert isinstance(scores_by_channel[RetrievalChannel.EXACT], ExactChannelScore)
    assert isinstance(scores_by_channel[RetrievalChannel.LEXICAL], LexicalChannelScore)
    assert isinstance(scores_by_channel[RetrievalChannel.STRUCTURED], StructuredChannelScore)
    assert isinstance(scores_by_channel[RetrievalChannel.VECTOR], VectorChannelScore)

    lexical_score = scores_by_channel[RetrievalChannel.LEXICAL]
    vector_score = scores_by_channel[RetrievalChannel.VECTOR]
    assert isinstance(lexical_score, LexicalChannelScore)
    assert isinstance(vector_score, VectorChannelScore)
    assert lexical_score.bm25_score != vector_score.cosine_similarity

    for candidate in collection.candidates:
        assert candidate.offer_id == OFFER_A
        assert candidate.source_ref.offer_id == OFFER_A
        assert candidate.rank == 0


def test_candidate_channel_score_type_mismatch_rejected() -> None:
    with pytest.raises(TypeError):
        ProductCandidate(
            offer_id=OFFER_A,
            channel=RetrievalChannel.LEXICAL,
            rank=0,
            source_ref=_source_ref(OFFER_A, catalog_id=CATALOG_A),
            channel_score=VectorChannelScore(cosine_similarity=0.5),
        )
