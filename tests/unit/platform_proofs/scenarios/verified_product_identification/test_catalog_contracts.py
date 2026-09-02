"""Unit tests for provider-neutral catalog/search contract foundation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from platform_proofs.scenarios.verified_product_identification.application.catalog import (
    SourceTruthResolutionError,
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
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
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


def _source_ref(
    offer_id: ProductOfferId,
    *,
    catalog_id: str,
    source_revision: str | None = None,
) -> SourceRecordRef:
    return SourceRecordRef(
        offer_id=offer_id,
        catalog_id=catalog_id,
        source_revision=source_revision,
    )


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
    source_revision: str | None = None

    def fetch(self, source_ref: SourceRecordRef) -> SourceRecordFetchResult:
        return SourceRecordFetchResult(
            record=ProductSourceRecord(
                offer_id=source_ref.offer_id,
                record_payload_ref=f"{self.catalog_id}:payload:{source_ref.offer_id.value}",
                provenance=ProductSourceProvenance(
                    catalog_id=self.catalog_id,
                    source_revision=self.source_revision,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class FakeSourceRecordStoreB:
    catalog_id: str = CATALOG_B
    source_revision: str | None = None

    def fetch(self, source_ref: SourceRecordRef) -> SourceRecordFetchResult:
        return SourceRecordFetchResult(
            record=ProductSourceRecord(
                offer_id=source_ref.offer_id,
                record_payload_ref=f"{self.catalog_id}:payload:{source_ref.offer_id.value}",
                provenance=ProductSourceProvenance(
                    catalog_id=self.catalog_id,
                    source_revision=self.source_revision,
                ),
            )
        )


def _resolve_with_store(
    port: SourceRecordFetchPort,
    candidate: ProductCandidate,
) -> ProductSourceRecord:
    return resolve_source_record(candidate, port)


def test_matching_source_reference_resolves() -> None:
    candidate = _consume_exact_lookup(FakeExactLookupA())

    source = _resolve_with_store(FakeSourceRecordStoreA(), candidate)

    assert source.offer_id == candidate.offer_id
    assert source.provenance.catalog_id == CATALOG_A
    assert source.provenance.catalog_id == candidate.source_ref.catalog_id


def test_catalog_mismatch_rejects() -> None:
    candidate = _consume_exact_lookup(FakeExactLookupA())

    with pytest.raises(SourceTruthResolutionError, match="catalog"):
        _resolve_with_store(FakeSourceRecordStoreB(), candidate)


def test_matching_source_revision_resolves() -> None:
    candidate = ProductCandidate(
        offer_id=OFFER_A,
        channel=RetrievalChannel.EXACT,
        rank=0,
        source_ref=_source_ref(OFFER_A, catalog_id=CATALOG_A, source_revision="rev-1"),
        channel_score=ExactChannelScore(
            matched_identifier=ProductIdentifier(
                identifier_type=ProductIdentifierType.GTIN,
                value="8806095123456",
            )
        ),
    )
    store = FakeSourceRecordStoreA(source_revision="rev-1")

    source = _resolve_with_store(store, candidate)

    assert source.provenance.source_revision == "rev-1"
    assert source.provenance.source_revision == candidate.source_ref.source_revision


def test_unspecified_source_revision_accepts_record_revision() -> None:
    candidate = ProductCandidate(
        offer_id=OFFER_A,
        channel=RetrievalChannel.EXACT,
        rank=0,
        source_ref=_source_ref(OFFER_A, catalog_id=CATALOG_A, source_revision=None),
        channel_score=ExactChannelScore(
            matched_identifier=ProductIdentifier(
                identifier_type=ProductIdentifierType.GTIN,
                value="8806095123456",
            )
        ),
    )
    store = FakeSourceRecordStoreA(source_revision="rev-populated")

    source = _resolve_with_store(store, candidate)

    assert candidate.source_ref.source_revision is None
    assert source.provenance.source_revision == "rev-populated"


def test_source_fetch_port_receives_full_source_ref() -> None:
    candidate = _consume_exact_lookup(FakeExactLookupA())
    captured: list[SourceRecordRef] = []

    @dataclass(frozen=True, slots=True)
    class CapturingSourceStore:
        def fetch(self, source_ref: SourceRecordRef) -> SourceRecordFetchResult:
            captured.append(source_ref)
            return FakeSourceRecordStoreA().fetch(source_ref)

    _resolve_with_store(CapturingSourceStore(), candidate)

    assert len(captured) == 1
    assert captured[0] == candidate.source_ref
    assert captured[0].offer_id == candidate.offer_id
    assert captured[0].catalog_id == candidate.source_ref.catalog_id


def test_source_revision_mismatch_rejects_when_revision_specified() -> None:
    candidate = ProductCandidate(
        offer_id=OFFER_A,
        channel=RetrievalChannel.EXACT,
        rank=0,
        source_ref=_source_ref(OFFER_A, catalog_id=CATALOG_A, source_revision="rev-expected"),
        channel_score=ExactChannelScore(
            matched_identifier=ProductIdentifier(
                identifier_type=ProductIdentifierType.GTIN,
                value="8806095123456",
            )
        ),
    )
    store = FakeSourceRecordStoreA(source_revision="rev-actual")

    with pytest.raises(SourceTruthResolutionError, match="revision"):
        _resolve_with_store(store, candidate)


def test_same_offer_id_across_catalogs_does_not_establish_identity() -> None:
    candidate_a = _consume_exact_lookup(FakeExactLookupA())
    candidate_b = _consume_exact_lookup(FakeExactLookupB())

    assert candidate_a.offer_id == candidate_b.offer_id
    assert candidate_a.source_ref.catalog_id != candidate_b.source_ref.catalog_id

    source_a = _resolve_with_store(FakeSourceRecordStoreA(), candidate_a)
    source_b = _resolve_with_store(FakeSourceRecordStoreB(), candidate_b)

    assert source_a.provenance.catalog_id == CATALOG_A
    assert source_b.provenance.catalog_id == CATALOG_B
    assert source_a.record_payload_ref != source_b.record_payload_ref

    with pytest.raises(SourceTruthResolutionError, match="catalog"):
        _resolve_with_store(FakeSourceRecordStoreB(), candidate_a)


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


def test_invalid_lexical_score_rejects() -> None:
    with pytest.raises(ValueError, match="bm25_score"):
        LexicalChannelScore(bm25_score=float("nan"))
    with pytest.raises(ValueError, match="bm25_score"):
        LexicalChannelScore(bm25_score=float("inf"))


def test_invalid_vector_score_rejects() -> None:
    with pytest.raises(ValueError, match="cosine_similarity"):
        VectorChannelScore(cosine_similarity=float("nan"))
    with pytest.raises(ValueError, match="cosine_similarity"):
        VectorChannelScore(cosine_similarity=float("inf"))
    with pytest.raises(ValueError, match="cosine_similarity"):
        VectorChannelScore(cosine_similarity=float("-inf"))
    with pytest.raises(ValueError, match="cosine_similarity"):
        VectorChannelScore(cosine_similarity=1.5)
    with pytest.raises(ValueError, match="cosine_similarity"):
        VectorChannelScore(cosine_similarity=-1.5)


def test_search_result_failure_rejects_non_empty_candidates() -> None:
    failure = CatalogSearchFailure(
        kind=CatalogSearchFailureKind.UNAVAILABLE,
        message="catalog backend unavailable",
    )
    candidate = _exact_candidate(
        offer_id=OFFER_A,
        rank=0,
        identifier=ProductIdentifier(
            identifier_type=ProductIdentifierType.GTIN,
            value="8806095123456",
        ),
        catalog_id=CATALOG_A,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        ExactIdentifierLookupResult(candidates=(candidate,), failure=failure)

    with pytest.raises(ValueError, match="mutually exclusive"):
        LexicalSearchResult(candidates=(candidate,), failure=failure)

    with pytest.raises(ValueError, match="mutually exclusive"):
        StructuredSearchResult(candidates=(candidate,), failure=failure)

    with pytest.raises(ValueError, match="mutually exclusive"):
        VectorSearchResult(candidates=(candidate,), failure=failure)


def test_source_fetch_result_failure_rejects_record() -> None:
    failure = CatalogSearchFailure(
        kind=CatalogSearchFailureKind.UNAVAILABLE,
        message="source store unavailable",
    )
    record = ProductSourceRecord(
        offer_id=OFFER_A,
        record_payload_ref="catalog-alpha:payload:offer-1001",
        provenance=ProductSourceProvenance(catalog_id=CATALOG_A),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        SourceRecordFetchResult(record=record, failure=failure)
