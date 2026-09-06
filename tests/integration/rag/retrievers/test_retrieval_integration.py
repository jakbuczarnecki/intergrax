# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import sys

import pytest
from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider

from intergrax.rag.retrievers.providers.vector_similarity_retriever import (
    VectorSimilarityRetriever,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.retrievers.engine.retriever_engine import RetrieverEngine
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.retriever_manager import RetrieverManager
from intergrax.rag.retrievers.pipeline.retriever_pipeline import RetrieverPipeline
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="ChromaDB Rust backend crashes on Windows during upsert",
    ),
]


def create_embedding_manager() -> EmbeddingManager:

    provider = HFEmbeddingProvider()
    engine = EmbeddingEngine(provider=provider)
    pipeline = EmbeddingPipeline(engine=engine)
    return EmbeddingManager(pipeline=pipeline)


def create_retriever_manager(vector_store: BaseVectorstoreManager, embedding_manager: BaseEmbeddingManager):

    retriever = VectorSimilarityRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
    )

    registry = RetrieverRegistry()
    registry.register(retriever)

    engine = RetrieverEngine(registry)

    pipeline = RetrieverPipeline(
        engine=engine,
        embedding_manager=embedding_manager,
    )

    manager = RetrieverManager(
        pipeline=pipeline
    )

    return manager


def test_retrieval_pipeline():

    embedding_manager = create_embedding_manager()

    documents = [
        Document(page_content="Artificial intelligence transforms software."),
        Document(page_content="Machine learning enables pattern discovery."),
        Document(page_content="Vector databases store embeddings."),
    ]

    result = embedding_manager.embed_documents(documents)

    vector_store = VectorstoreManager(
        store=create_qdrant_vector_store(
            collection_name="retrieval_integration_it_qdrant",
            tenant_id="tenant_a",
        )
    )


    vector_store.add_documents(
        result.documents,
        result.embeddings,
    )

    retriever_manager = create_retriever_manager(
        vector_store,
        embedding_manager,
    )

    query = RetrieverQuery(
        query_text="artificial intelligence",
        query_embedding=None,
        top_k=2,
        metadata_filter=None,
        include_embeddings=False,
    )

    results = retriever_manager.retrieve_query(query)

    assert len(results) == 2

    contents = [r.content for r in results]

    assert any(
        "Artificial intelligence" in c
        for c in contents
    )