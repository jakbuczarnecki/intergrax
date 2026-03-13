# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

import sys

from langchain_core.documents import Document
import pytest

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_pipeline
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.providers.vector_similarity_retriever import VectorSimilarityRetriever
from intergrax.rag.vectorstore.providers.chroma_vector_store import ChromaConfig, ChromaVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="ChromaDB Rust backend crashes on Windows during upsert",
    ),
]


def test_retriever_pipeline_basic() -> None:

    chroma_cfg = ChromaConfig(
        tenant_id="intergrax",
        collection_name="test_retrieval_pipeline_basic",
        persist_directory=None,
        settings=None,
    )

    vector_provider = ChromaVectorStore(cfg=chroma_cfg)
    vector_store = VectorstoreManager(store=vector_provider)

    embedding_registry = EmbeddingProviderRegistry()
    embedding_provider = HFEmbeddingProvider()
    embedding_registry.register(embedding_provider)

    engine = EmbeddingEngine(embedding_registry)

    embedding_pipeline = EmbeddingPipeline(
        engine=engine,
        provider_id=embedding_provider.provider_name(),
    )

    embedding_manager = EmbeddingManager(pipeline=embedding_pipeline)

    docs = [
        Document(page_content="Transformers are neural network architectures"),
        Document(page_content="Vector databases store embeddings"),
        Document(page_content="Machine learning models learn from data"),
    ]
    texts = [doc.page_content for doc in docs]
    embeddings = embedding_manager.embed_texts(texts)

    vector_store.add_documents(
        documents=docs,
        embeddings=embeddings,
    )

    retriever_pipeline = create_default_retriever_pipeline(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
    )

    query = RetrieverQuery(
        query_text="neural network",
        top_k=3,
    )

    results = retriever_pipeline.retrieve_query(
        query,
        retriever_id=VectorSimilarityRetriever.name(),
    )

    assert len(results) > 0