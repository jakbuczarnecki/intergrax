# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Optional

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.profiles.rag_profile import RagProfile, rag_profile_from_env
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import (
    create_default_reranker_engine,
    create_default_reranker_registry_only,
)
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.rerankers.re_ranker_manager import ReRankerManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager


def resolve_retrieval_service(
    *,
    vectorstore_manager: BaseVectorstoreManager,
    embedding_manager: BaseEmbeddingManager,
    retriever_manager: BaseRetrieverManager | None = None,
    reranker_manager: BaseRerankerManager | None = None,
    profile: Optional[RagProfile] = None,
    retrieval_service: Optional[RetrievalService] = None,
) -> Optional[RetrievalService]:
    if retrieval_service is not None:
        return retrieval_service
    if vectorstore_manager is None or embedding_manager is None:
        return None

    profile = profile or rag_profile_from_env()
    if retriever_manager is None:
        retriever_manager = create_default_retriever_manager(
            vector_store=vectorstore_manager,
            embedding_manager=embedding_manager,
            profile=profile,
        )
    if reranker_manager is None and profile.enable_rerank:
        registry = create_default_reranker_registry_only(
            embedding_manager=embedding_manager,
        )
        reranker_manager = ReRankerManager(
            engine=create_default_reranker_engine(
                embedding_manager=embedding_manager,
                registry=registry,
            )
        )

    return RetrievalService(
        retriever_manager=retriever_manager,
        reranker_manager=reranker_manager,
        profile=profile,
    )
