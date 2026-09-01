"""Provider-neutral catalog/search ports — no backend implementations."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    LexicalSearchQuery,
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
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)


class ExactIdentifierLookupPort(Protocol):
    """Exact identifier lookup against catalog source identities."""

    def lookup(self, query: ExactIdentifierQuery) -> ExactIdentifierLookupResult:
        """Return exact-match candidates for one typed identifier."""


class LexicalCandidateSearchPort(Protocol):
    """Lexical candidate retrieval over derived lexical representations."""

    def search(self, query: LexicalSearchQuery) -> LexicalSearchResult:
        """Return lexical channel candidates for one query."""


class StructuredCandidateSearchPort(Protocol):
    """Structured candidate retrieval over normalized attribute representations."""

    def search(self, query: StructuredSearchQuery) -> StructuredSearchResult:
        """Return structured channel candidates for one constraint set."""


class VectorCandidateSearchPort(Protocol):
    """Vector candidate retrieval over derived embedding representations."""

    def search(self, query: VectorSearchQuery) -> VectorSearchResult:
        """Return vector channel candidates for one semantic query."""


class SourceRecordFetchPort(Protocol):
    """Fetch immutable source-truth records by offer identity."""

    def fetch(self, offer_id: ProductOfferId) -> SourceRecordFetchResult:
        """Return the immutable source record for one offer identity."""
