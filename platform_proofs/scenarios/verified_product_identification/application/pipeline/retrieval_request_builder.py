"""Derive MultiChannelRetrievalRequest from authoritative ProductIdentificationQuery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.contracts.product_identification_query import (
    ProductIdentificationQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    ExactIdentifierQuery,
    LexicalSearchQuery,
    StructuredSearchQuery,
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.pipeline.contracts import (
    ProductIdentificationPipelineConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.retrieval.contracts import (
    MultiChannelRetrievalRequest,
)


class ProductIdentificationRetrievalRequestBuilder(Protocol):
    def build(self, query: ProductIdentificationQuery) -> MultiChannelRetrievalRequest:
        ...


@dataclass(frozen=True, slots=True)
class DeterministicProductIdentificationRetrievalRequestBuilder:
    """Maps typed query semantics into recall-oriented retrieval contracts only."""

    configuration: ProductIdentificationPipelineConfiguration = (
        ProductIdentificationPipelineConfiguration()
    )

    def build(self, query: ProductIdentificationQuery) -> MultiChannelRetrievalRequest:
        context = query.verification_context
        exact_queries = tuple(
            ExactIdentifierQuery(
                identifier=identifier,
                limit=self.configuration.exact_retrieval_limit,
            )
            for identifier in context.requested_identifiers
        )
        structured_query: StructuredSearchQuery | None = None
        if context.required_constraints:
            structured_query = StructuredSearchQuery(
                constraints=context.required_constraints,
                limit=self.configuration.structured_retrieval_limit,
            )
        lexical_query: LexicalSearchQuery | None = None
        vector_query: VectorSearchQuery | None = None
        if query.search_text is not None:
            lexical_query = LexicalSearchQuery(
                query_text=query.search_text,
                limit=self.configuration.lexical_retrieval_limit,
            )
            vector_query = VectorSearchQuery(
                query_text=query.search_text,
                limit=self.configuration.vector_retrieval_limit,
            )
        return MultiChannelRetrievalRequest(
            exact_queries=exact_queries,
            lexical_query=lexical_query,
            structured_query=structured_query,
            vector_query=vector_query,
        )
