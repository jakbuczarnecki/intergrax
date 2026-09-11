"""Provider-neutral multi-channel retrieval orchestration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    LexicalSearchQuery,
    StructuredSearchQuery,
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    ExactIdentifierLookupResult,
    LexicalSearchResult,
    StructuredSearchResult,
    VectorSearchResult,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    MultiChannelCandidateCollection,
    ProductCandidate,
    RetrievalChannel,
)


class RetrievalChannelExecutionStatus(StrEnum):
    """Per-channel orchestration outcome — distinct from candidate verification."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class RetrievalExecutionPolicy:
    """Orchestration policy — limits remain on typed query contracts."""

    fail_fast_on_exact_failure: bool = False

    def __post_init__(self) -> None:
        if type(self.fail_fast_on_exact_failure) is not bool:
            raise TypeError("fail_fast_on_exact_failure must be a bool")


@dataclass(frozen=True, slots=True)
class MultiChannelRetrievalRequest:
    """Immutable orchestration request over already-typed retrieval inputs."""

    exact_queries: tuple[ExactIdentifierQuery, ...] = ()
    lexical_query: LexicalSearchQuery | None = None
    structured_query: StructuredSearchQuery | None = None
    vector_query: VectorSearchQuery | None = None
    execution_policy: RetrievalExecutionPolicy = RetrievalExecutionPolicy()

    def __post_init__(self) -> None:
        if type(self.exact_queries) is not tuple:
            raise TypeError("exact_queries must be a tuple")
        if not self.has_any_channel_input():
            raise ValueError(
                "MultiChannelRetrievalRequest requires at least one channel input"
            )

    def has_any_channel_input(self) -> bool:
        return bool(self.exact_queries) or any(
            query is not None
            for query in (
                self.lexical_query,
                self.structured_query,
                self.vector_query,
            )
        )


@dataclass(frozen=True, slots=True)
class ExactChannelRetrievalOutcome:
    status: RetrievalChannelExecutionStatus
    lookup_results: tuple[ExactIdentifierLookupResult, ...]
    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        if type(self.lookup_results) is not tuple:
            raise TypeError("lookup_results must be a tuple")
        if type(self.candidates) is not tuple:
            raise TypeError("candidates must be a tuple")
        for candidate in self.candidates:
            if candidate.channel != RetrievalChannel.EXACT:
                raise ValueError("exact channel outcome candidates must use EXACT channel")
        if self.status is RetrievalChannelExecutionStatus.SKIPPED:
            if self.lookup_results or self.candidates or self.failure is not None:
                raise ValueError("skipped exact channel must have empty results and no failure")
        if self.status is RetrievalChannelExecutionStatus.FAILED and self.failure is None:
            raise ValueError("failed exact channel must include failure evidence")
        if self.status is RetrievalChannelExecutionStatus.SUCCESS and self.failure is not None:
            raise ValueError("successful exact channel must not include failure evidence")
        if self.status is RetrievalChannelExecutionStatus.FAILED and self.candidates:
            if len(self.lookup_results) <= 1:
                raise ValueError(
                    "failed exact channel with candidates requires multi-query partial execution"
                )
            has_failed_lookup = any(
                lookup_result.failure is not None for lookup_result in self.lookup_results
            )
            has_successful_lookup = any(
                lookup_result.failure is None for lookup_result in self.lookup_results
            )
            if not has_failed_lookup or not has_successful_lookup:
                raise ValueError(
                    "failed exact channel with candidates requires both failed and successful lookups"
                )


@dataclass(frozen=True, slots=True)
class LexicalChannelRetrievalOutcome:
    status: RetrievalChannelExecutionStatus
    search_result: LexicalSearchResult | None
    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple:
            raise TypeError("candidates must be a tuple")
        for candidate in self.candidates:
            if candidate.channel != RetrievalChannel.LEXICAL:
                raise ValueError("lexical channel outcome candidates must use LEXICAL channel")
        if self.status is RetrievalChannelExecutionStatus.SKIPPED:
            if self.search_result is not None or self.candidates or self.failure is not None:
                raise ValueError("skipped lexical channel must have empty results and no failure")
        if self.status is RetrievalChannelExecutionStatus.FAILED and self.failure is None:
            raise ValueError("failed lexical channel must include failure evidence")
        if self.status is RetrievalChannelExecutionStatus.SUCCESS and self.failure is not None:
            raise ValueError("successful lexical channel must not include failure evidence")
        if self.status is RetrievalChannelExecutionStatus.FAILED and self.candidates:
            raise ValueError("failed lexical channel must have empty candidates")


@dataclass(frozen=True, slots=True)
class StructuredChannelRetrievalOutcome:
    status: RetrievalChannelExecutionStatus
    search_result: StructuredSearchResult | None
    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple:
            raise TypeError("candidates must be a tuple")
        for candidate in self.candidates:
            if candidate.channel != RetrievalChannel.STRUCTURED:
                raise ValueError(
                    "structured channel outcome candidates must use STRUCTURED channel"
                )
        if self.status is RetrievalChannelExecutionStatus.SKIPPED:
            if self.search_result is not None or self.candidates or self.failure is not None:
                raise ValueError(
                    "skipped structured channel must have empty results and no failure"
                )
        if self.status is RetrievalChannelExecutionStatus.FAILED and self.failure is None:
            raise ValueError("failed structured channel must include failure evidence")
        if self.status is RetrievalChannelExecutionStatus.SUCCESS and self.failure is not None:
            raise ValueError("successful structured channel must not include failure evidence")
        if self.status is RetrievalChannelExecutionStatus.FAILED and self.candidates:
            raise ValueError("failed structured channel must have empty candidates")


@dataclass(frozen=True, slots=True)
class VectorChannelRetrievalOutcome:
    status: RetrievalChannelExecutionStatus
    search_result: VectorSearchResult | None
    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple:
            raise TypeError("candidates must be a tuple")
        for candidate in self.candidates:
            if candidate.channel != RetrievalChannel.VECTOR:
                raise ValueError("vector channel outcome candidates must use VECTOR channel")
        if self.status is RetrievalChannelExecutionStatus.SKIPPED:
            if self.search_result is not None or self.candidates or self.failure is not None:
                raise ValueError("skipped vector channel must have empty results and no failure")
        if self.status is RetrievalChannelExecutionStatus.FAILED and self.failure is None:
            raise ValueError("failed vector channel must include failure evidence")
        if self.status is RetrievalChannelExecutionStatus.SUCCESS and self.failure is not None:
            raise ValueError("successful vector channel must not include failure evidence")
        if self.status is RetrievalChannelExecutionStatus.FAILED and self.candidates:
            raise ValueError("failed vector channel must have empty candidates")


@dataclass(frozen=True, slots=True)
class RetrievalExecutionSummary:
    """Deterministic execution metadata — no wall-clock timing."""

    channels_attempted: int
    channels_succeeded: int
    channels_failed: int
    channels_skipped: int
    exact_candidate_count: int
    lexical_candidate_count: int
    structured_candidate_count: int
    vector_candidate_count: int

    def __post_init__(self) -> None:
        for value in (
            self.channels_attempted,
            self.channels_succeeded,
            self.channels_failed,
            self.channels_skipped,
            self.exact_candidate_count,
            self.lexical_candidate_count,
            self.structured_candidate_count,
            self.vector_candidate_count,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("execution summary counts must be non-negative ints")
        if self.channels_attempted != self.channels_succeeded + self.channels_failed:
            raise ValueError("channels_attempted must equal channels_succeeded + channels_failed")
        if self.channels_attempted + self.channels_skipped != 4:
            raise ValueError("channels_attempted + channels_skipped must equal 4")


@dataclass(frozen=True, slots=True)
class MultiChannelRetrievalResult:
    """Aggregate multi-channel retrieval outcome before offer-level fusion."""

    exact: ExactChannelRetrievalOutcome
    lexical: LexicalChannelRetrievalOutcome
    structured: StructuredChannelRetrievalOutcome
    vector: VectorChannelRetrievalOutcome
    candidates: MultiChannelCandidateCollection
    execution_summary: RetrievalExecutionSummary

    def __post_init__(self) -> None:
        summary = self.execution_summary
        if summary.exact_candidate_count != len(self.exact.candidates):
            raise ValueError("exact_candidate_count must match exact channel candidates")
        if summary.lexical_candidate_count != len(self.lexical.candidates):
            raise ValueError("lexical_candidate_count must match lexical channel candidates")
        if summary.structured_candidate_count != len(self.structured.candidates):
            raise ValueError(
                "structured_candidate_count must match structured channel candidates"
            )
        if summary.vector_candidate_count != len(self.vector.candidates):
            raise ValueError("vector_candidate_count must match vector channel candidates")

        expected_candidates = (
            self.exact.candidates
            + self.lexical.candidates
            + self.structured.candidates
            + self.vector.candidates
        )
        if self.candidates.candidates != expected_candidates:
            raise ValueError(
                "candidate collection must concatenate channel outcomes in deterministic order"
            )
