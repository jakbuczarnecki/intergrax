"""Provider-neutral catalog search result envelopes."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ProductCandidate,
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    ProductSourceRecord,
)


def _validate_channel_candidates(
    *,
    channel: RetrievalChannel,
    candidates: tuple[ProductCandidate, ...],
) -> None:
    if not isinstance(candidates, tuple):
        raise TypeError("candidates must be a tuple")
    for candidate in candidates:
        if candidate.channel != channel:
            raise ValueError(f"all candidates must use channel {channel.value}")


def _validate_search_result_state(
    *,
    candidates: tuple[ProductCandidate, ...],
    failure: CatalogSearchFailure | None,
) -> None:
    if failure is not None and len(candidates) > 0:
        raise ValueError("failure and candidates are mutually exclusive")


@dataclass(frozen=True, slots=True)
class ExactIdentifierLookupResult:
    """Exact lookup outcome — success candidates or failure, never both."""

    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        _validate_search_result_state(candidates=self.candidates, failure=self.failure)
        _validate_channel_candidates(channel=RetrievalChannel.EXACT, candidates=self.candidates)


@dataclass(frozen=True, slots=True)
class LexicalSearchResult:
    """Lexical search outcome — success candidates or failure, never both."""

    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        _validate_search_result_state(candidates=self.candidates, failure=self.failure)
        _validate_channel_candidates(channel=RetrievalChannel.LEXICAL, candidates=self.candidates)


@dataclass(frozen=True, slots=True)
class StructuredSearchResult:
    """Structured search outcome — success candidates or failure, never both."""

    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        _validate_search_result_state(candidates=self.candidates, failure=self.failure)
        _validate_channel_candidates(
            channel=RetrievalChannel.STRUCTURED,
            candidates=self.candidates,
        )


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """Vector search outcome — success candidates or failure, never both."""

    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        _validate_search_result_state(candidates=self.candidates, failure=self.failure)
        _validate_channel_candidates(channel=RetrievalChannel.VECTOR, candidates=self.candidates)


@dataclass(frozen=True, slots=True)
class SourceRecordFetchResult:
    """Source fetch outcome — resolved record or failure, never both."""

    record: ProductSourceRecord | None
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        if self.record is not None and self.failure is not None:
            raise ValueError("record and failure are mutually exclusive")
