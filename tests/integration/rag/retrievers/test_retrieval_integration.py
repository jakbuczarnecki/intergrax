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
from intergrax.rag.embedding.registry.embedding_provider_registry import (
    EmbeddingProviderRegistry,
)
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
from intergrax.rag.vectorstore.providers.qdrant_vector_store import QdrantConfig, QdrantVectorStore


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="ChromaDB Rust backend crashes on Windows during upsert",
    ),
]


def create_embedding_manager() -> EmbeddingManager:

    registry = EmbeddingProviderRegistry()

    provider = HFEmbeddingProvider()

    registry.register(provider)

    engine = EmbeddingEngine(registry)

    pipeline = EmbeddingPipeline(
        engine=engine,
        provider_id=provider.provider_name(),
    )

    manager = EmbeddingManager(pipeline=pipeline)

    return manager


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

    embedded_docs = embedding_manager.embed_documents(documents)

    cfg = QdrantConfig(
            collection_name="retrieval_integration_it_qdrant",
            tenant_id="tenant_a",
        )
    vector_store = QdrantVectorStore(cfg)

    embeddings = [
        doc.metadata[EmbeddingMetadataKey.VECTOR]
        for doc in embedded_docs
    ]

    vector_store.add_documents(
        embedded_docs,
        embeddings,
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