"""Platform vector search adapter for VPI vector retrieval port."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from intergrax.integrations.contracts.vector_index_administration import VectorIndexIdentity
from intergrax.integrations.contracts.vector_store import VectorStore, VectorStoreScope
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.opens import open_qdrant_vector_store

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
from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ProductCandidate,
    RetrievalChannel,
    VectorChannelScore,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)


@dataclass(slots=True)
class PlatformVectorSearchAdapter:
    _vector_store: VectorStore
    _scope: VectorStoreScope
    _embedding: IntergraxEmbeddingBootstrapAdapter
    _catalog_id: str

    @classmethod
    def from_env(
        cls,
        *,
        collection_name: str,
        catalog_id: str,
        embedding_configuration: VpiEmbeddingConfiguration,
        execution_configuration: VpiEmbeddingProviderExecutionConfiguration,
    ) -> PlatformVectorSearchAdapter:
        qdrant_config = QdrantIntegrationConfig.from_env(
            collection_name=collection_name,
            enable_sparse_vectors=True,
        )
        vector_store = open_qdrant_vector_store(qdrant_config)
        scope = VectorStoreScope(tenant_id=qdrant_config.tenant_id)
        embedding = IntergraxEmbeddingBootstrapAdapter(
            embedding_configuration,
            execution_configuration=execution_configuration,
        )
        return cls(
            _vector_store=vector_store,
            _scope=scope,
            _embedding=embedding,
            _catalog_id=catalog_id,
        )

    def search(self, query: VectorSearchQuery) -> VectorSearchResult:
        vectors = self._embedding.embed_batch((query.query_text,))
        query_vector = np.asarray(vectors[0], dtype=np.float32)
        hits = self._vector_store.query(
            query_vector,
            scope=self._scope,
            top_k=query.limit,
        )
        candidates: list[ProductCandidate] = []
        for rank, hit in enumerate(hits):
            metadata = hit.document.metadata
            offer_id_raw = metadata.get("offer_id")
            if offer_id_raw is None:
                continue
            source_ref = SourceRecordRef(
                offer_id=ProductOfferId(str(offer_id_raw)),
                catalog_id=str(metadata.get("catalog_id", self._catalog_id)),
                source_revision=metadata.get("source_revision"),
            )
            candidates.append(
                ProductCandidate(
                    offer_id=source_ref.offer_id,
                    channel=RetrievalChannel.VECTOR,
                    rank=rank,
                    source_ref=source_ref,
                    channel_score=VectorChannelScore(cosine_similarity=float(hit.similarity_score)),
                )
            )
        return VectorSearchResult(candidates=tuple(candidates))

    def close(self) -> None:
        self._embedding.close()
