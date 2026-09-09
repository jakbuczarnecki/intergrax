"""LEGACY / REFERENCE ONLY — not canonical production retrieval.

Canonical path: ``integrations/search_store/qdrant_vector_candidate_search_adapter.py``
Composition: ``retrieval/composition.py`` → ``build_vector_candidate_search``
"""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingProviderExecutionConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    VectorSearchQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.results import (
    VectorSearchResult,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.qdrant_vector_candidate_search_adapter import (
    QdrantVectorCandidateSearchAdapter,
)


@dataclass(slots=True)
class PlatformVectorSearchAdapter:
    _delegate: QdrantVectorCandidateSearchAdapter

    @classmethod
    def from_env(
        cls,
        *,
        collection_name: str,
        catalog_id: str,
        embedding_configuration: VpiEmbeddingConfiguration,
        execution_configuration: VpiEmbeddingProviderExecutionConfiguration,
    ) -> PlatformVectorSearchAdapter:
        return cls(
            _delegate=QdrantVectorCandidateSearchAdapter.from_env(
                collection_name=collection_name,
                catalog_scope_id=catalog_id,
                embedding_configuration=embedding_configuration,
                execution_configuration=execution_configuration,
            )
        )

    def search(self, query: VectorSearchQuery) -> VectorSearchResult:
        return self._delegate.search(query)

    def close(self) -> None:
        self._delegate.close()
