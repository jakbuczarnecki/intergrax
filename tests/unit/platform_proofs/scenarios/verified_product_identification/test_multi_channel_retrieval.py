"""Unit tests for provider-neutral multi-channel retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
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
from platform_proofs.scenarios.verified_product_identification.application.retrieval import (
    MultiChannelRetrievalRequest,
    MultiChannelRetrievalService,
    RetrievalChannelExecutionStatus,
    RetrievalExecutionPolicy,
)

pytestmark = pytest.mark.unit

OFFER_A = ProductOfferId("offer-1001")
OFFER_B = ProductOfferId("offer-1002")
CATALOG_ID = "catalog-alpha"


def _source_ref(offer_id: ProductOfferId) -> SourceRecordRef:
    return SourceRecordRef(offer_id=offer_id, catalog_id=CATALOG_ID)


def _exact_candidate(
    *,
    offer_id: ProductOfferId,
    rank: int,
    identifier: ProductIdentifier,
) -> ProductCandidate:
    return ProductCandidate(
        offer_id=offer_id,
        channel=RetrievalChannel.EXACT,
        rank=rank,
        source_ref=_source_ref(offer_id),
        channel_score=ExactChannelScore(matched_identifier=identifier),
    )


def _lexical_candidate(*, offer_id: ProductOfferId, rank: int, score: float) -> ProductCandidate:
    return ProductCandidate(
        offer_id=offer_id,
        channel=RetrievalChannel.LEXICAL,
        rank=rank,
        source_ref=_source_ref(offer_id),
        channel_score=LexicalChannelScore(bm25_score=score),
    )


def _structured_candidate(*, offer_id: ProductOfferId, rank: int) -> ProductCandidate:
    return ProductCandidate(
        offer_id=offer_id,
        channel=RetrievalChannel.STRUCTURED,
        rank=rank,
        source_ref=_source_ref(offer_id),
        channel_score=StructuredChannelScore(
            matched_constraint_count=1,
            total_constraint_count=1,
        ),
    )


def _vector_candidate(*, offer_id: ProductOfferId, rank: int, score: float) -> ProductCandidate:
    return ProductCandidate(
        offer_id=offer_id,
        channel=RetrievalChannel.VECTOR,
        rank=rank,
        source_ref=_source_ref(offer_id),
        channel_score=VectorChannelScore(cosine_similarity=score),
    )


GTIN_IDENTIFIER = ProductIdentifier(
    identifier_type=ProductIdentifierType.GTIN,
    value="8806095123456",
)
MPN_IDENTIFIER = ProductIdentifier(
    identifier_type=ProductIdentifierType.MPN,
    value="MZ-V9P2T0BW",
)


@dataclass
class RecordingExactLookup:
    calls: list[ExactIdentifierQuery] = field(default_factory=list)
    responses: tuple[ExactIdentifierLookupResult, ...] = ()

    def lookup(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
        self.calls.append(query)
        index = len(self.calls) - 1
        if index < len(self.responses):
            return self.responses[index]
        return ExactIdentifierLookupResult(candidates=())


@dataclass
class RecordingLexicalSearch:
    calls: list[LexicalSearchQuery] = field(default_factory=list)
    response: LexicalSearchResult = field(
        default_factory=lambda: LexicalSearchResult(candidates=())
    )

    def search(self, query: LexicalSearchQuery) -> LexicalSearchResult:
        self.calls.append(query)
        return self.response


@dataclass
class RecordingStructuredSearch:
    calls: list[StructuredSearchQuery] = field(default_factory=list)
    response: StructuredSearchResult = field(
        default_factory=lambda: StructuredSearchResult(candidates=())
    )

    def search(self, query: StructuredSearchQuery) -> StructuredSearchResult:
        self.calls.append(query)
        return self.response


@dataclass
class RecordingVectorSearch:
    calls: list[VectorSearchQuery] = field(default_factory=list)
    response: VectorSearchResult = field(
        default_factory=lambda: VectorSearchResult(candidates=())
    )

    def search(self, query: VectorSearchQuery) -> VectorSearchResult:
        self.calls.append(query)
        return self.response


def _service(
    *,
    exact: RecordingExactLookup | None = None,
    lexical: RecordingLexicalSearch | None = None,
    structured: RecordingStructuredSearch | None = None,
    vector: RecordingVectorSearch | None = None,
) -> tuple[
    MultiChannelRetrievalService,
    RecordingExactLookup,
    RecordingLexicalSearch,
    RecordingStructuredSearch,
    RecordingVectorSearch,
]:
    exact_port = exact or RecordingExactLookup()
    lexical_port = lexical or RecordingLexicalSearch()
    structured_port = structured or RecordingStructuredSearch()
    vector_port = vector or RecordingVectorSearch()
    service = MultiChannelRetrievalService(
        exact_lookup=exact_port,
        lexical_search=lexical_port,
        structured_search=structured_port,
        vector_search=vector_port,
    )
    return service, exact_port, lexical_port, structured_port, vector_port


def _structured_query() -> StructuredSearchQuery:
    return StructuredSearchQuery(
        constraints=(
            StructuredAttributeConstraint(
                attribute_name="capacity",
                operator=StructuredConstraintOperator.EQUALS,
                value="2TB",
            ),
        )
    )


def test_all_four_channels_execute_in_deterministic_order() -> None:
    service, exact_port, lexical_port, structured_port, vector_port = _service(
        exact=RecordingExactLookup(
            responses=(
                ExactIdentifierLookupResult(
                    candidates=(
                        _exact_candidate(
                            offer_id=OFFER_A,
                            rank=0,
                            identifier=GTIN_IDENTIFIER,
                        ),
                    )
                ),
            )
        ),
        lexical=RecordingLexicalSearch(
            response=LexicalSearchResult(
                candidates=(_lexical_candidate(offer_id=OFFER_A, rank=0, score=12.0),)
            )
        ),
        structured=RecordingStructuredSearch(
            response=StructuredSearchResult(
                candidates=(_structured_candidate(offer_id=OFFER_A, rank=0),)
            )
        ),
        vector=RecordingVectorSearch(
            response=VectorSearchResult(
                candidates=(_vector_candidate(offer_id=OFFER_A, rank=0, score=0.91),)
            )
        ),
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),),
            lexical_query=LexicalSearchQuery(query_text="990 PRO 2TB"),
            structured_query=_structured_query(),
            vector_query=VectorSearchQuery(query_text="Samsung NVMe SSD"),
        )
    )

    assert len(exact_port.calls) == 1
    assert len(lexical_port.calls) == 1
    assert len(structured_port.calls) == 1
    assert len(vector_port.calls) == 1
    assert exact_port.calls and lexical_port.calls and structured_port.calls and vector_port.calls
    assert result.exact.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.lexical.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.structured.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.vector.status is RetrievalChannelExecutionStatus.SUCCESS
    assert len(result.candidates.candidates) == 4
    assert result.execution_summary.channels_attempted == 4
    assert result.execution_summary.channels_succeeded == 4
    assert result.execution_summary.channels_failed == 0
    assert result.execution_summary.channels_skipped == 0


def test_exact_only_skips_other_channels() -> None:
    service, exact_port, lexical_port, structured_port, vector_port = _service(
        exact=RecordingExactLookup(
            responses=(
                ExactIdentifierLookupResult(
                    candidates=(
                        _exact_candidate(
                            offer_id=OFFER_A,
                            rank=0,
                            identifier=GTIN_IDENTIFIER,
                        ),
                    )
                ),
            )
        )
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),),
        )
    )

    assert len(exact_port.calls) == 1
    assert lexical_port.calls == []
    assert structured_port.calls == []
    assert vector_port.calls == []
    assert result.exact.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.lexical.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.structured.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.vector.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.execution_summary.channels_skipped == 3


def test_lexical_only_skips_other_channels() -> None:
    service, exact_port, lexical_port, structured_port, vector_port = _service(
        lexical=RecordingLexicalSearch(
            response=LexicalSearchResult(
                candidates=(_lexical_candidate(offer_id=OFFER_A, rank=0, score=8.5),)
            )
        )
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            lexical_query=LexicalSearchQuery(query_text="990 PRO"),
        )
    )

    assert exact_port.calls == []
    assert len(lexical_port.calls) == 1
    assert structured_port.calls == []
    assert vector_port.calls == []
    assert result.exact.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.lexical.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.structured.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.vector.status is RetrievalChannelExecutionStatus.SKIPPED


def test_structured_only_skips_other_channels() -> None:
    service, exact_port, lexical_port, structured_port, vector_port = _service(
        structured=RecordingStructuredSearch(
            response=StructuredSearchResult(
                candidates=(_structured_candidate(offer_id=OFFER_A, rank=0),)
            )
        )
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(structured_query=_structured_query())
    )

    assert exact_port.calls == []
    assert lexical_port.calls == []
    assert len(structured_port.calls) == 1
    assert vector_port.calls == []
    assert result.structured.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.exact.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.lexical.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.vector.status is RetrievalChannelExecutionStatus.SKIPPED


def test_vector_only_skips_other_channels() -> None:
    service, exact_port, lexical_port, structured_port, vector_port = _service(
        vector=RecordingVectorSearch(
            response=VectorSearchResult(
                candidates=(_vector_candidate(offer_id=OFFER_A, rank=0, score=0.77),)
            )
        )
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            vector_query=VectorSearchQuery(query_text="NVMe SSD"),
        )
    )

    assert exact_port.calls == []
    assert lexical_port.calls == []
    assert structured_port.calls == []
    assert len(vector_port.calls) == 1
    assert result.vector.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.exact.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.lexical.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.structured.status is RetrievalChannelExecutionStatus.SKIPPED


def test_empty_request_rejects_contract_error() -> None:
    with pytest.raises(ValueError, match="at least one channel input"):
        MultiChannelRetrievalRequest()


def test_zero_candidate_success_for_each_channel() -> None:
    service, _, _, _, _ = _service()

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),),
            lexical_query=LexicalSearchQuery(query_text="missing product"),
            structured_query=_structured_query(),
            vector_query=VectorSearchQuery(query_text="missing product"),
        )
    )

    assert result.exact.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.lexical.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.structured.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.vector.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.candidates.candidates == ()
    assert result.execution_summary.exact_candidate_count == 0


def test_partial_channel_failure_preserves_other_candidates() -> None:
    failure = CatalogSearchFailure(
        kind=CatalogSearchFailureKind.UNAVAILABLE,
        message="lexical backend unavailable",
    )
    service, _, _, _, _ = _service(
        exact=RecordingExactLookup(
            responses=(
                ExactIdentifierLookupResult(
                    candidates=(
                        _exact_candidate(
                            offer_id=OFFER_A,
                            rank=0,
                            identifier=GTIN_IDENTIFIER,
                        ),
                    )
                ),
            )
        ),
        lexical=RecordingLexicalSearch(
            response=LexicalSearchResult(candidates=(), failure=failure),
        ),
        vector=RecordingVectorSearch(
            response=VectorSearchResult(
                candidates=(_vector_candidate(offer_id=OFFER_A, rank=0, score=0.66),)
            )
        ),
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),),
            lexical_query=LexicalSearchQuery(query_text="990 PRO"),
            vector_query=VectorSearchQuery(query_text="990 PRO"),
        )
    )

    assert result.exact.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.lexical.status is RetrievalChannelExecutionStatus.FAILED
    assert result.lexical.failure == failure
    assert result.vector.status is RetrievalChannelExecutionStatus.SUCCESS
    assert len(result.candidates.candidates) == 2
    channels = {candidate.channel for candidate in result.candidates.candidates}
    assert channels == {RetrievalChannel.EXACT, RetrievalChannel.VECTOR}


def test_all_requested_channels_fail_without_fabricated_candidates() -> None:
    exact_failure = CatalogSearchFailure(
        kind=CatalogSearchFailureKind.TIMEOUT,
        message="exact lookup timeout",
    )
    lexical_failure = CatalogSearchFailure(
        kind=CatalogSearchFailureKind.UNAVAILABLE,
        message="lexical unavailable",
    )
    service, _, _, _, _ = _service(
        exact=RecordingExactLookup(
            responses=(ExactIdentifierLookupResult(candidates=(), failure=exact_failure),)
        ),
        lexical=RecordingLexicalSearch(
            response=LexicalSearchResult(candidates=(), failure=lexical_failure),
        ),
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),),
            lexical_query=LexicalSearchQuery(query_text="990 PRO"),
        )
    )

    assert result.exact.status is RetrievalChannelExecutionStatus.FAILED
    assert result.lexical.status is RetrievalChannelExecutionStatus.FAILED
    assert result.candidates.candidates == ()
    assert result.execution_summary.channels_failed == 2


def test_skipped_is_not_failed() -> None:
    service, _, _, _, _ = _service(
        lexical=RecordingLexicalSearch(
            response=LexicalSearchResult(
                candidates=(_lexical_candidate(offer_id=OFFER_A, rank=0, score=5.0),)
            )
        )
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            lexical_query=LexicalSearchQuery(query_text="990 PRO"),
        )
    )

    assert result.lexical.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.exact.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.exact.failure is None
    assert result.execution_summary.channels_failed == 0
    assert result.execution_summary.channels_skipped == 3


def test_same_offer_from_multiple_channels_remains_distinct_events() -> None:
    service, _, _, _, _ = _service(
        exact=RecordingExactLookup(
            responses=(
                ExactIdentifierLookupResult(
                    candidates=(
                        _exact_candidate(
                            offer_id=OFFER_A,
                            rank=0,
                            identifier=GTIN_IDENTIFIER,
                        ),
                    )
                ),
            )
        ),
        lexical=RecordingLexicalSearch(
            response=LexicalSearchResult(
                candidates=(_lexical_candidate(offer_id=OFFER_A, rank=0, score=9.0),)
            )
        ),
        vector=RecordingVectorSearch(
            response=VectorSearchResult(
                candidates=(_vector_candidate(offer_id=OFFER_A, rank=0, score=0.8),)
            )
        ),
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),),
            lexical_query=LexicalSearchQuery(query_text="990 PRO"),
            vector_query=VectorSearchQuery(query_text="990 PRO"),
        )
    )

    same_offer_events = [
        candidate
        for candidate in result.candidates.candidates
        if candidate.offer_id == OFFER_A
    ]
    assert len(same_offer_events) == 3
    assert {candidate.channel for candidate in same_offer_events} == {
        RetrievalChannel.EXACT,
        RetrievalChannel.LEXICAL,
        RetrievalChannel.VECTOR,
    }


def test_channel_local_rank_and_score_types_preserved() -> None:
    service, _, _, _, _ = _service(
        exact=RecordingExactLookup(
            responses=(
                ExactIdentifierLookupResult(
                    candidates=(
                        _exact_candidate(
                            offer_id=OFFER_A,
                            rank=0,
                            identifier=GTIN_IDENTIFIER,
                        ),
                        _exact_candidate(
                            offer_id=OFFER_B,
                            rank=1,
                            identifier=MPN_IDENTIFIER,
                        ),
                    )
                ),
            )
        ),
        lexical=RecordingLexicalSearch(
            response=LexicalSearchResult(
                candidates=(
                    _lexical_candidate(offer_id=OFFER_A, rank=0, score=11.0),
                    _lexical_candidate(offer_id=OFFER_B, rank=1, score=7.5),
                )
            )
        ),
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),),
            lexical_query=LexicalSearchQuery(query_text="990 PRO"),
        )
    )

    exact_candidates = [c for c in result.candidates.candidates if c.channel == RetrievalChannel.EXACT]
    lexical_candidates = [c for c in result.candidates.candidates if c.channel == RetrievalChannel.LEXICAL]
    assert [candidate.rank for candidate in exact_candidates] == [0, 1]
    assert [candidate.rank for candidate in lexical_candidates] == [0, 1]
    assert all(isinstance(candidate.channel_score, ExactChannelScore) for candidate in exact_candidates)
    assert all(isinstance(candidate.channel_score, LexicalChannelScore) for candidate in lexical_candidates)


def test_multiple_exact_queries_execute_in_input_order() -> None:
    service, exact_port, _, _, _ = _service(
        exact=RecordingExactLookup(
            responses=(
                ExactIdentifierLookupResult(
                    candidates=(
                        _exact_candidate(
                            offer_id=OFFER_A,
                            rank=0,
                            identifier=GTIN_IDENTIFIER,
                        ),
                    )
                ),
                ExactIdentifierLookupResult(
                    candidates=(
                        _exact_candidate(
                            offer_id=OFFER_B,
                            rank=0,
                            identifier=MPN_IDENTIFIER,
                        ),
                    )
                ),
            )
        )
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(
                ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),
                ExactIdentifierQuery(identifier=MPN_IDENTIFIER),
            ),
        )
    )

    assert len(exact_port.calls) == 2
    assert exact_port.calls[0].identifier == GTIN_IDENTIFIER
    assert exact_port.calls[1].identifier == MPN_IDENTIFIER
    assert result.exact.status is RetrievalChannelExecutionStatus.SUCCESS
    assert len(result.exact.lookup_results) == 2
    assert len(result.candidates.candidates) == 2


def test_exact_partial_lookup_failure_preserves_successful_candidates() -> None:
    failure = CatalogSearchFailure(
        kind=CatalogSearchFailureKind.INVALID_QUERY,
        message="invalid gtin format",
    )
    service, _, _, _, _ = _service(
        exact=RecordingExactLookup(
            responses=(
                ExactIdentifierLookupResult(candidates=(), failure=failure),
                ExactIdentifierLookupResult(
                    candidates=(
                        _exact_candidate(
                            offer_id=OFFER_B,
                            rank=0,
                            identifier=MPN_IDENTIFIER,
                        ),
                    )
                ),
            )
        )
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(
                ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),
                ExactIdentifierQuery(identifier=MPN_IDENTIFIER),
            ),
        )
    )

    assert result.exact.status is RetrievalChannelExecutionStatus.FAILED
    assert result.exact.failure == failure
    assert len(result.exact.candidates) == 1
    assert result.exact.candidates[0].offer_id == OFFER_B


def test_fail_fast_on_exact_failure_skips_downstream_channels() -> None:
    failure = CatalogSearchFailure(
        kind=CatalogSearchFailureKind.TIMEOUT,
        message="exact timeout",
    )
    service, exact_port, lexical_port, structured_port, vector_port = _service(
        exact=RecordingExactLookup(
            responses=(ExactIdentifierLookupResult(candidates=(), failure=failure),)
        ),
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),),
            lexical_query=LexicalSearchQuery(query_text="990 PRO"),
            structured_query=_structured_query(),
            vector_query=VectorSearchQuery(query_text="990 PRO"),
            execution_policy=RetrievalExecutionPolicy(fail_fast_on_exact_failure=True),
        )
    )

    assert len(exact_port.calls) == 1
    assert lexical_port.calls == []
    assert structured_port.calls == []
    assert vector_port.calls == []
    assert result.exact.status is RetrievalChannelExecutionStatus.FAILED
    assert result.lexical.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.structured.status is RetrievalChannelExecutionStatus.SKIPPED
    assert result.vector.status is RetrievalChannelExecutionStatus.SKIPPED


def test_provider_neutral_wiring_with_alternate_fake_backends() -> None:
    @dataclass(frozen=True, slots=True)
    class MysqlStyleExactLookup:
        def lookup(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
            return ExactIdentifierLookupResult(
                candidates=(
                    _exact_candidate(
                        offer_id=OFFER_A,
                        rank=0,
                        identifier=query.identifier,
                    ),
                )
            )

    @dataclass(frozen=True, slots=True)
    class QdrantStyleVectorSearch:
        def search(self, query: VectorSearchQuery) -> VectorSearchResult:
            del query
            return VectorSearchResult(
                candidates=(_vector_candidate(offer_id=OFFER_A, rank=0, score=0.55),)
            )

    service = MultiChannelRetrievalService(
        exact_lookup=MysqlStyleExactLookup(),
        lexical_search=RecordingLexicalSearch(),
        structured_search=RecordingStructuredSearch(),
        vector_search=QdrantStyleVectorSearch(),
    )

    result = service.retrieve(
        MultiChannelRetrievalRequest(
            exact_queries=(ExactIdentifierQuery(identifier=GTIN_IDENTIFIER),),
            vector_query=VectorSearchQuery(query_text="ssd"),
        )
    )

    assert result.exact.status is RetrievalChannelExecutionStatus.SUCCESS
    assert result.vector.status is RetrievalChannelExecutionStatus.SUCCESS
    assert len(result.candidates.candidates) == 2


def test_malformed_channel_outcome_rejects_channel_mismatch() -> None:
    from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
        LexicalChannelRetrievalOutcome,
    )

    with pytest.raises(ValueError, match="LEXICAL channel"):
        LexicalChannelRetrievalOutcome(
            status=RetrievalChannelExecutionStatus.SUCCESS,
            search_result=LexicalSearchResult(candidates=()),
            candidates=(
                _vector_candidate(offer_id=OFFER_A, rank=0, score=0.5),
            ),
        )
