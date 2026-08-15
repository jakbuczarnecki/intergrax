# © Artur Czarnecki. All rights reserved.

"""Canonical RAG retrieval path (Phase S-H.4) — ``RetrievalService`` only; no ``rag.answers`` stack."""

from __future__ import annotations

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
from testing_support.builder import build_fake_embedding_manager, build_in_memory_vectorstore_manager


pytestmark = [pytest.mark.integration, pytest.mark.gate]


def test_retrieval_service_hybrid_path_without_answers_stack() -> None:
    vectorstore_manager = build_in_memory_vectorstore_manager(tenant_id="default")
    embedding_manager = build_fake_embedding_manager()

    documents = [
        KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {"document_id": document_id, "root_document_id": document_id},
                "scope": {"tenant_id": "default"},
                "content": content,
                "metadata": {},
                "provenance": {"source_kind": "test", "source_id": document_id},
            }
        )
        for document_id, content in (
            ("paris", "Paris is the capital of France."),
            ("berlin", "Berlin is the capital of Germany."),
        )
    ]
    result = embedding_manager.embed_documents(documents)
    vectorstore_manager.add_documents(documents=result.documents, embeddings=result.embeddings)

    retriever_manager = create_default_retriever_manager(
        vector_store=vectorstore_manager,
        embedding_manager=embedding_manager,
    )
    profile = RagProfile()
    service = RetrievalService(retriever_manager=retriever_manager, profile=profile)

    result = service.retrieve(RetrievalRequest(query="capital of France", top_k=2))
    assert result.used is True
    assert result.chunks
    assert any("Paris" in chunk.text for chunk in result.chunks)
