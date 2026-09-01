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
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    ProductSourceRecord,
)


def _validate_channel_candidates(
    *,
    channel: RetrievalChannel,
    candidates: tuple[ProductCandidate, ...],
) -> tuple[ProductCandidate, ...]:
    normalized = tuple(candidates)
    for candidate in normalized:
        if candidate.channel != channel:
            raise ValueError(f"all candidates must use channel {channel.value}")
    return normalized


@dataclass(frozen=True, slots=True)
class ExactIdentifierLookupResult:
    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidates",
            _validate_channel_candidates(
                channel=RetrievalChannel.EXACT,
                candidates=self.candidates,
            ),
        )


@dataclass(frozen=True, slots=True)
class LexicalSearchResult:
    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidates",
            _validate_channel_candidates(
                channel=RetrievalChannel.LEXICAL,
                candidates=self.candidates,
            ),
        )


@dataclass(frozen=True, slots=True)
class StructuredSearchResult:
    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidates",
            _validate_channel_candidates(
                channel=RetrievalChannel.STRUCTURED,
                candidates=self.candidates,
            ),
        )


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    candidates: tuple[ProductCandidate, ...]
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidates",
            _validate_channel_candidates(
                channel=RetrievalChannel.VECTOR,
                candidates=self.candidates,
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceRecordFetchResult:
    record: ProductSourceRecord | None
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        if self.record is not None and self.failure is not None:
            raise ValueError("record and failure are mutually exclusive")
