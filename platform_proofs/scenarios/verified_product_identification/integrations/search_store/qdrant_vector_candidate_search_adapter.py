"""Canonical Qdrant vector candidate search adapter for VPI vector retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.contracts.vector_store import (
    MetadataFilter,
    VectorStore,
    VectorStoreHit,
    VectorStoreScope,
)
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.opens import open_qdrant_vector_store
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreContractError

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
    VpiEmbeddingDimensionMismatchError,
    load_vpi_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingProviderExecutionConfiguration,
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
    CatalogSearchFailureKind,
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
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.search_store.vector_hit_identity import (
    VectorHitIdentity,
    VectorHitIdentityDecodeError,
    decode_vector_hit_identity_from_metadata,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.qdrant.payload import (
    SOURCE_CATALOG_PAYLOAD_KEY,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapProviderError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    EmbeddingExecutionPort,
)


class VectorHitScoreValidationError(ValueError):
    """Raised when a provider hit score violates cosine similarity semantics."""


def _map_embedding_failure(exc: BaseException) -> CatalogSearchFailure:
    if isinstance(exc, VpiEmbeddingDimensionMismatchError):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="query embedding dimension mismatch",
        )
    if isinstance(exc, VpiBootstrapProviderError):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="embedding provider unavailable",
        )
    return CatalogSearchFailure(
        kind=CatalogSearchFailureKind.UNAVAILABLE,
        message="embedding provider unavailable",
    )


def _map_vector_store_failure(exc: BaseException) -> CatalogSearchFailure | None:
    if isinstance(exc, VectorStoreContractError):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.INVALID_QUERY,
            message="vector query rejected by provider contract",
        )
    if isinstance(exc, IntegrationDependencyError):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="vector store unavailable",
        )
    if isinstance(exc, (OSError, ConnectionError, TimeoutError)):
        return CatalogSearchFailure(
            kind=CatalogSearchFailureKind.UNAVAILABLE,
            message="vector store unavailable",
        )
    return None


def _validate_cosine_similarity(score: float) -> float:
    if not math.isfinite(score):
        raise VectorHitScoreValidationError("vector hit score must be finite")
    if score < -1.0 or score > 1.0:
        raise VectorHitScoreValidationError(
            "vector hit score must be within [-1.0, 1.0]"
        )
    return score


def _provider_data_failure(message: str) -> CatalogSearchFailure:
    return CatalogSearchFailure(
        kind=CatalogSearchFailureKind.INVALID_QUERY,
        message=message,
    )


def _rank_hits_deterministic(
    hits: tuple[tuple[VectorHitIdentity, float], ...],
) -> tuple[tuple[VectorHitIdentity, float], ...]:
    return tuple(
        sorted(
            hits,
            key=lambda item: (
                -item[1],
                item[0].catalog_id,
                item[0].offer_id.value,
                item[0].source_revision_norm,
            ),
        )
    )


def _deduplicate_hits(
    hits: tuple[tuple[VectorHitIdentity, float], ...],
) -> tuple[tuple[VectorHitIdentity, float], ...]:
    best_by_identity: dict[tuple[str, str, str], tuple[VectorHitIdentity, float]] = {}
    for identity, score in hits:
        key = (
            identity.catalog_id,
            identity.offer_id.value,
            identity.source_revision_norm,
        )
        existing = best_by_identity.get(key)
        if existing is None or score > existing[1]:
            best_by_identity[key] = (identity, score)
    return tuple(best_by_identity.values())


@dataclass(slots=True)
class QdrantVectorCandidateSearchAdapter:
    """``VectorCandidateSearchPort`` over platform ``VectorStore`` with typed identity decode."""

    _vector_store: VectorStore
    _scope: VectorStoreScope
    _embedding: EmbeddingExecutionPort
    _embedding_configuration: VpiEmbeddingConfiguration
    _catalog_scope_id: str | None
    _owns_embedding: bool

    @classmethod
    def from_env(
        cls,
        *,
        collection_name: str,
        catalog_scope_id: str | None = None,
        embedding_configuration: VpiEmbeddingConfiguration | None = None,
        execution_configuration: VpiEmbeddingProviderExecutionConfiguration | None = None,
    ) -> QdrantVectorCandidateSearchAdapter:
        resolved_embedding_configuration = (
            embedding_configuration or load_vpi_embedding_configuration()
        )
        resolved_execution_configuration = (
            execution_configuration or load_vpi_embedding_provider_execution_configuration()
        )
        qdrant_config = QdrantIntegrationConfig.from_env(
            collection_name=collection_name,
            enable_sparse_vectors=False,
            metric="cosine",
        )
        vector_store = open_qdrant_vector_store(qdrant_config)
        scope = VectorStoreScope(tenant_id=qdrant_config.tenant_id)
        embedding = IntergraxEmbeddingBootstrapAdapter(
            resolved_embedding_configuration,
            execution_configuration=resolved_execution_configuration,
        )
        return cls(
            _vector_store=vector_store,
            _scope=scope,
            _embedding=embedding,
            _embedding_configuration=resolved_embedding_configuration,
            _catalog_scope_id=catalog_scope_id,
            _owns_embedding=True,
        )

    @classmethod
    def from_dependencies(
        cls,
        *,
        vector_store: VectorStore,
        scope: VectorStoreScope,
        embedding: EmbeddingExecutionPort,
        embedding_configuration: VpiEmbeddingConfiguration,
        catalog_scope_id: str | None = None,
    ) -> QdrantVectorCandidateSearchAdapter:
        return cls(
            _vector_store=vector_store,
            _scope=scope,
            _embedding=embedding,
            _embedding_configuration=embedding_configuration,
            _catalog_scope_id=catalog_scope_id,
            _owns_embedding=False,
        )

    def search(self, query: VectorSearchQuery) -> VectorSearchResult:
        try:
            query_vectors = self._embedding.embed_batch((query.query_text,))
        except (VpiBootstrapProviderError, VpiEmbeddingDimensionMismatchError) as exc:
            return VectorSearchResult(candidates=(), failure=_map_embedding_failure(exc))

        if len(query_vectors) != 1:
            return VectorSearchResult(
                candidates=(),
                failure=_provider_data_failure("query embedding must return exactly one vector"),
            )

        query_vector = query_vectors[0]
        if len(query_vector) != self._embedding_configuration.expected_dimension:
            return VectorSearchResult(
                candidates=(),
                failure=CatalogSearchFailure(
                    kind=CatalogSearchFailureKind.INVALID_QUERY,
                    message="query embedding dimension mismatch",
                ),
            )

        metadata_filter = None
        if self._catalog_scope_id is not None:
            metadata_filter = MetadataFilter(
                conditions={SOURCE_CATALOG_PAYLOAD_KEY: self._catalog_scope_id}
            )

        try:
            hits = self._vector_store.query(
                np.asarray(query_vector, dtype=np.float32),
                scope=self._scope,
                top_k=query.limit,
                metadata_filter=metadata_filter,
            )
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            VectorStoreContractError,
            IntegrationDependencyError,
        ) as exc:
            mapped = _map_vector_store_failure(exc)
            if mapped is not None:
                return VectorSearchResult(candidates=(), failure=mapped)
            raise

        decoded_hits: list[tuple[VectorHitIdentity, float]] = []
        for hit in hits:
            mapped = self._decode_provider_hit(hit)
            if isinstance(mapped, CatalogSearchFailure):
                return VectorSearchResult(candidates=(), failure=mapped)
            decoded_hits.append(mapped)

        if not decoded_hits:
            return VectorSearchResult(candidates=())

        ranked = _rank_hits_deterministic(_deduplicate_hits(tuple(decoded_hits)))
        bounded = ranked[: query.limit]
        candidates = tuple(
            ProductCandidate(
                offer_id=identity.offer_id,
                channel=RetrievalChannel.VECTOR,
                rank=index,
                source_ref=identity.to_source_record_ref(),
                channel_score=VectorChannelScore(cosine_similarity=score),
            )
            for index, (identity, score) in enumerate(bounded)
        )
        return VectorSearchResult(candidates=candidates)

    def _decode_provider_hit(
        self,
        hit: VectorStoreHit,
    ) -> tuple[VectorHitIdentity, float] | CatalogSearchFailure:
        try:
            identity = decode_vector_hit_identity_from_metadata(hit.document.metadata)
            score = _validate_cosine_similarity(float(hit.similarity_score))
        except VectorHitIdentityDecodeError as exc:
            return _provider_data_failure(str(exc))
        except VectorHitScoreValidationError as exc:
            return _provider_data_failure(str(exc))
        return identity, score

    def close(self) -> None:
        if self._owns_embedding:
            self._embedding.close()
