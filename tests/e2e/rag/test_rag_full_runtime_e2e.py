# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document

from intergrax.rag.document_loaders.bootstrap.default_loader import create_default_documents_loader
from intergrax.rag.document_splitters.bootstrap.default_chunking_engine import (
    create_default_document_splitter,
)
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import create_default_reranker_manager
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import create_default_vectorstore_manager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager

pytestmark = pytest.mark.e2e


class RagE2EPipeline:
    """
    End-to-end RAG pipeline using canonical ``RetrievalService`` (Phase Q+-L.6).

    Does not import ``intergrax.rag.answers``.
    """

    def __init__(self) -> None:
        self.dataset_path = Path("tests/fixtures/documents")
        self.documents: list[Document] | None = None
        self.chunked_documents: list[Document] | None = None
        self.embedding_manager: BaseEmbeddingManager | None = None
        self.vectorstore: BaseVectorstoreManager | None = None
        self.retriever: BaseRetrieverManager | None = None
        self.reranker: BaseRerankerManager | None = None
        self.retrieval_service: RetrievalService | None = None
        self.query = "What information exists in the dataset?"
        self.result = None

    def load_dataset(self) -> None:
        loader = create_default_documents_loader()
        documents: List[Document] = []
        for dataset_dir in (
            self.dataset_path / "pdf",
            self.dataset_path / "docx",
            self.dataset_path / "xlsx",
        ):
            if not dataset_dir.exists():
                continue
            docs = loader.load_documents(str(dataset_dir))
            if docs:
                documents.extend(docs)

        text_documents = [d for d in documents if d.page_content and d.page_content.strip()]
        assert text_documents
        self.documents = text_documents

    def build_index(self) -> None:
        self.embedding_manager = create_default_embedding_manager()
        self.vectorstore = create_default_vectorstore_manager()
        assert self.embedding_manager is not None
        assert self.vectorstore is not None

    def chunk_documents(self) -> None:
        chunker = create_default_document_splitter()
        assert self.documents is not None
        self.chunked_documents = chunker.split_documents(self.documents)
        assert self.chunked_documents

    def index_documents(self) -> None:
        assert self.chunked_documents is not None
        assert self.embedding_manager is not None
        assert self.vectorstore is not None
        embed_result = self.embedding_manager.embed_documents(self.chunked_documents)
        assert len(embed_result.documents) == len(self.chunked_documents)
        self.vectorstore.add_documents(
            documents=embed_result.documents,
            embeddings=embed_result.embeddings,
        )
        assert self.vectorstore.count() > 0

    def build_retriever(self) -> None:
        assert self.vectorstore is not None
        self.retriever = create_default_retriever_manager(
            vector_store=self.vectorstore,
            embedding_manager=self.embedding_manager,
        )

    def build_reranker(self) -> None:
        self.reranker = create_default_reranker_manager()

    def build_retrieval_service(self) -> None:
        assert self.retriever is not None
        profile = RagProfile(
            retriever_id="hybrid",
            enable_rerank=False,
            route_mode="off",
        )
        self.retrieval_service = RetrievalService(
            retriever_manager=self.retriever,
            reranker_manager=self.reranker,
            profile=profile,
        )

    def run_query(self) -> None:
        assert self.retrieval_service is not None
        self.result = self.retrieval_service.retrieve(
            RetrievalRequest(query=self.query, final_top_k=5),
        )

    def validate_result(self) -> None:
        assert self.result is not None
        assert self.result.used is True
        assert self.result.chunks
        assert self.result.chunks[0].text.strip()


def test_rag_full_runtime_e2e() -> None:
    pipeline = RagE2EPipeline()
    pipeline.load_dataset()
    pipeline.build_index()
    pipeline.chunk_documents()
    pipeline.index_documents()
    pipeline.build_retriever()
    pipeline.build_reranker()
    pipeline.build_retrieval_service()
    pipeline.run_query()
    pipeline.validate_result()
