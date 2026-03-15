# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from pathlib import Path
from typing import List
import pytest
from langchain_core.documents import Document

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.rag.answers.bootstrap.answer_bootstrap import create_default_answer_engine, create_default_answerer_manager
from intergrax.rag.answers.contracts.answer_request import AnswerRequest
from intergrax.rag.answers.contracts.base_answer_manager import BaseAnswerManager
from intergrax.rag.document_loaders.bootstrap.default_loader import create_default_documents_loader
from intergrax.rag.document_splitters.bootstrap.default_chunking_engine import create_default_document_splitter
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey
from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import create_default_reranker_manager
from intergrax.rag.rerankers.contracts.base_reranker_manager import BaseRerankerManager
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import create_default_vectorstore_manager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager

pytestmark = pytest.mark.e2e


class RagE2EPipeline:
    """
    Orchestrates the full RAG E2E pipeline.

    This class represents the production flow executed
    by the runtime but allows step-by-step construction
    during test development.
    """

    def __init__(self):

        self.dataset_path = Path("tests/fixtures/documents")

        # DATA
        self.documents: list[Document] | None = None
        self.chunked_documents: list[Document] | None = None
        self.retrieved_documents: list[Document] | None = None
        self.context_documents: list[Document] | None = None

        # COMPONENTS
        self.embedding_manager: BaseEmbeddingManager | None = None
        self.vectorstore: BaseVectorstoreManager | None = None
        self.retriever: BaseRetrieverManager | None = None
        self.reranker: BaseRerankerManager | None = None
        self.answer_engine: BaseAnswerManager | None = None

        # QUERY
        self.query = "What information exists in the dataset?"

        # RESULT
        self.result = None

    # --------------------------------------------------
    # STEP 1
    # Dataset loading
    # --------------------------------------------------

    def load_dataset(self):
        """
        Loads documents from dataset fixtures using the same loader
        as used in ingestion integration tests.
        """

        loader = create_default_documents_loader()

        documents : List[Document] = []

        dataset_dirs = [
            self.dataset_path / "pdf",
            self.dataset_path / "docx",
            # self.dataset_path / "html",
            self.dataset_path / "xlsx",
            # self.dataset_path / "txt",
        ]

        for dataset_dir in dataset_dirs:

            if not dataset_dir.exists():
                continue

            docs = loader.load_documents(str(dataset_dir))

            if docs:
                documents.extend(docs)

        assert documents is not None
        assert len(documents) > 0

        text_documents = [
            d for d in documents
            if d.page_content and d.page_content.strip()
        ]

        assert len(text_documents) > 0

        self.documents = text_documents

    # --------------------------------------------------
    # STEP 2
    # Index construction
    # --------------------------------------------------

    def build_index(self):
        """
        Prepares indexing infrastructure.
        """

        # embedding manager
        self.embedding_manager = create_default_embedding_manager()

        assert self.embedding_manager is not None

        # vector store
        self.vectorstore = create_default_vectorstore_manager()

        assert self.vectorstore is not None


    # --------------------------------------------------
    # STEP 3
    # Chunk documents
    # --------------------------------------------------
    def chunk_documents(self):
        """
        Splits documents into chunks.
        """

        chunker = create_default_document_splitter()

        self.chunked_documents = chunker.split_documents(self.documents)

        assert self.chunked_documents is not None
        assert len(self.chunked_documents) > 0


    # --------------------------------------------------
    # STEP 4
    # Create index in vectorstore - add documents
    # --------------------------------------------------
    def index_documents(self):
        """
        Indexes chunked documents into vectorstore.
        """

        assert self.chunked_documents is not None
        assert self.embedding_manager is not None
        assert self.vectorstore is not None

        # generate embeddings
        result = self.embedding_manager.embed_documents(
            self.chunked_documents
        )

        assert result is not None
        assert len(result.documents) == len(self.chunked_documents)
        
        self.vectorstore.add_documents(
            documents=result.documents,
            embeddings=result.embeddings,
        )
        
        assert self.vectorstore.count() > 0


    # --------------------------------------------------
    # STEP 4
    # Retriever construction
    # --------------------------------------------------

    def build_retriever(self):
        """
        Builds retriever using vector store.
        """

        assert self.vectorstore is not None

        self.retriever = create_default_retriever_manager(
            vector_store=self.vectorstore,
            embedding_manager=self.embedding_manager,
        )

        assert self.retriever is not None

    # --------------------------------------------------
    # STEP 5
    # Reranker construction
    # --------------------------------------------------

    def build_reranker(self):
        """
        Optional reranker used after retrieval.
        """
        self.reranker = create_default_reranker_manager()

        assert self.reranker is not None

    # --------------------------------------------------
    # STEP 6
    # Answer pipeline construction
    # --------------------------------------------------

    def build_answer_engine(self):
        """
        Builds the answer generation pipeline.
        """

        assert self.retriever is not None
        assert self.reranker is not None

        engine = create_default_answer_engine(
            retriever_manager=self.retriever,
            reranker_engine=self.reranker,
        )

        self.answer_engine = create_default_answerer_manager(
            engine=engine,
        )

        assert self.answer_engine is not None


    # --------------------------------------------------
    # STEP 7
    # Query execution
    # --------------------------------------------------

    def run_query(self):
        """
        Executes user query through answer engine.
        """

        assert self.answer_engine is not None

        request = AnswerRequest(
            query=self.query,
            llm=LLMAdapterRegistry.create(LLMProvider.OLLAMA)
        )

        self.result = self.answer_engine.answer(
            request=request,
        )

        assert self.result is not None

    # --------------------------------------------------
    # STEP 8
    # Result validation
    # --------------------------------------------------

    def validate_result(self):
        """
        Validates final RAG result.
        """

        assert self.result is not None
        assert self.result.answer
        assert self.result.context_documents is not None


# ------------------------------------------------------
# ENTRYPOINT TEST
# ------------------------------------------------------

def test_rag_full_runtime_e2e():

    pipeline = RagE2EPipeline()

    # dataset
    pipeline.load_dataset()

    # vectorstore
    pipeline.build_index()

    # chunking
    pipeline.chunk_documents()

    # add documents to vectorstore
    pipeline.index_documents()

    # retrieval
    pipeline.build_retriever()

    # reranking
    pipeline.build_reranker()

    # answer pipeline
    pipeline.build_answer_engine()

    # query execution
    pipeline.run_query()

    # assertions
    pipeline.validate_result()