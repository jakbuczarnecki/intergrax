"""Deterministic provider-neutral multi-channel retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.catalog.candidate_handoff import (
    collect_channel_candidates,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    LexicalSearchQuery,
    StructuredSearchQuery,
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    ExactIdentifierLookupResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ProductCandidate,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ChannelCandidateBatch,
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    ExactIdentifierLookupPort,
    LexicalCandidateSearchPort,
    StructuredCandidateSearchPort,
    VectorCandidateSearchPort,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
    ExactChannelRetrievalOutcome,
    LexicalChannelRetrievalOutcome,
    MultiChannelRetrievalRequest,
    MultiChannelRetrievalResult,
    RetrievalChannelExecutionStatus,
    RetrievalExecutionSummary,
    StructuredChannelRetrievalOutcome,
    VectorChannelRetrievalOutcome,
)


def _skipped_exact_outcome() -> ExactChannelRetrievalOutcome:
    return ExactChannelRetrievalOutcome(
        status=RetrievalChannelExecutionStatus.SKIPPED,
        lookup_results=(),
        candidates=(),
    )


def _skipped_lexical_outcome() -> LexicalChannelRetrievalOutcome:
    return LexicalChannelRetrievalOutcome(
        status=RetrievalChannelExecutionStatus.SKIPPED,
        search_result=None,
        candidates=(),
    )


def _skipped_structured_outcome() -> StructuredChannelRetrievalOutcome:
    return StructuredChannelRetrievalOutcome(
        status=RetrievalChannelExecutionStatus.SKIPPED,
        search_result=None,
        candidates=(),
    )


def _skipped_vector_outcome() -> VectorChannelRetrievalOutcome:
    return VectorChannelRetrievalOutcome(
        status=RetrievalChannelExecutionStatus.SKIPPED,
        search_result=None,
        candidates=(),
    )


@dataclass(frozen=True, slots=True)
class MultiChannelRetrievalService:
    """Coordinate typed retrieval channels without fusion or provider coupling."""

    exact_lookup: ExactIdentifierLookupPort
    lexical_search: LexicalCandidateSearchPort
    structured_search: StructuredCandidateSearchPort
    vector_search: VectorCandidateSearchPort

    def retrieve(self, request: MultiChannelRetrievalRequest) -> MultiChannelRetrievalResult:
        exact = self._execute_exact(request.exact_queries)
        if (
            request.execution_policy.fail_fast_on_exact_failure
            and exact.status is RetrievalChannelExecutionStatus.FAILED
        ):
            lexical = _skipped_lexical_outcome()
            structured = _skipped_structured_outcome()
            vector = _skipped_vector_outcome()
        else:
            lexical = self._execute_lexical(request.lexical_query)
            structured = self._execute_structured(request.structured_query)
            vector = self._execute_vector(request.vector_query)

        batches: list[ChannelCandidateBatch] = []
        if exact.candidates:
            batches.append(
                ChannelCandidateBatch(
                    channel=RetrievalChannel.EXACT,
                    candidates=exact.candidates,
                )
            )
        if lexical.candidates:
            batches.append(
                ChannelCandidateBatch(
                    channel=RetrievalChannel.LEXICAL,
                    candidates=lexical.candidates,
                )
            )
        if structured.candidates:
            batches.append(
                ChannelCandidateBatch(
                    channel=RetrievalChannel.STRUCTURED,
                    candidates=structured.candidates,
                )
            )
        if vector.candidates:
            batches.append(
                ChannelCandidateBatch(
                    channel=RetrievalChannel.VECTOR,
                    candidates=vector.candidates,
                )
            )

        candidates = collect_channel_candidates(*batches) if batches else collect_channel_candidates()
        execution_summary = _build_execution_summary(
            exact=exact,
            lexical=lexical,
            structured=structured,
            vector=vector,
        )
        return MultiChannelRetrievalResult(
            exact=exact,
            lexical=lexical,
            structured=structured,
            vector=vector,
            candidates=candidates,
            execution_summary=execution_summary,
        )

    def _execute_exact(
        self,
        queries: tuple[ExactIdentifierQuery, ...],
    ) -> ExactChannelRetrievalOutcome:
        if not queries:
            return _skipped_exact_outcome()

        lookup_results: list[ExactIdentifierLookupResult] = []
        merged_candidates: list[ProductCandidate] = []
        first_failure = None

        for query in queries:
            result = self.exact_lookup.lookup(query)
            lookup_results.append(result)
            if result.failure is not None:
                if first_failure is None:
                    first_failure = result.failure
                continue
            merged_candidates.extend(result.candidates)

        if first_failure is not None:
            return ExactChannelRetrievalOutcome(
                status=RetrievalChannelExecutionStatus.FAILED,
                lookup_results=tuple(lookup_results),
                candidates=tuple(merged_candidates),
                failure=first_failure,
            )

        return ExactChannelRetrievalOutcome(
            status=RetrievalChannelExecutionStatus.SUCCESS,
            lookup_results=tuple(lookup_results),
            candidates=tuple(merged_candidates),
        )

    def _execute_lexical(
        self,
        query: LexicalSearchQuery | None,
    ) -> LexicalChannelRetrievalOutcome:
        if query is None:
            return _skipped_lexical_outcome()

        result = self.lexical_search.search(query)
        if result.failure is not None:
            return LexicalChannelRetrievalOutcome(
                status=RetrievalChannelExecutionStatus.FAILED,
                search_result=result,
                candidates=(),
                failure=result.failure,
            )

        return LexicalChannelRetrievalOutcome(
            status=RetrievalChannelExecutionStatus.SUCCESS,
            search_result=result,
            candidates=result.candidates,
        )

    def _execute_structured(
        self,
        query: StructuredSearchQuery | None,
    ) -> StructuredChannelRetrievalOutcome:
        if query is None:
            return _skipped_structured_outcome()

        result = self.structured_search.search(query)
        if result.failure is not None:
            return StructuredChannelRetrievalOutcome(
                status=RetrievalChannelExecutionStatus.FAILED,
                search_result=result,
                candidates=(),
                failure=result.failure,
            )

        return StructuredChannelRetrievalOutcome(
            status=RetrievalChannelExecutionStatus.SUCCESS,
            search_result=result,
            candidates=result.candidates,
        )

    def _execute_vector(
        self,
        query: VectorSearchQuery | None,
    ) -> VectorChannelRetrievalOutcome:
        if query is None:
            return _skipped_vector_outcome()

        result = self.vector_search.search(query)
        if result.failure is not None:
            return VectorChannelRetrievalOutcome(
                status=RetrievalChannelExecutionStatus.FAILED,
                search_result=result,
                candidates=(),
                failure=result.failure,
            )

        return VectorChannelRetrievalOutcome(
            status=RetrievalChannelExecutionStatus.SUCCESS,
            search_result=result,
            candidates=result.candidates,
        )


def _build_execution_summary(
    *,
    exact: ExactChannelRetrievalOutcome,
    lexical: LexicalChannelRetrievalOutcome,
    structured: StructuredChannelRetrievalOutcome,
    vector: VectorChannelRetrievalOutcome,
) -> RetrievalExecutionSummary:
    outcomes = (exact, lexical, structured, vector)
    channels_attempted = sum(
        1
        for outcome in outcomes
        if outcome.status is not RetrievalChannelExecutionStatus.SKIPPED
    )
    channels_succeeded = sum(
        1
        for outcome in outcomes
        if outcome.status is RetrievalChannelExecutionStatus.SUCCESS
    )
    channels_failed = sum(
        1
        for outcome in outcomes
        if outcome.status is RetrievalChannelExecutionStatus.FAILED
    )
    channels_skipped = sum(
        1
        for outcome in outcomes
        if outcome.status is RetrievalChannelExecutionStatus.SKIPPED
    )
    return RetrievalExecutionSummary(
        channels_attempted=channels_attempted,
        channels_succeeded=channels_succeeded,
        channels_failed=channels_failed,
        channels_skipped=channels_skipped,
        exact_candidate_count=len(exact.candidates),
        lexical_candidate_count=len(lexical.candidates),
        structured_candidate_count=len(structured.candidates),
        vector_candidate_count=len(vector.candidates),
    )
