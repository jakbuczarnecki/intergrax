# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.network,
]


def _require_path(p: Path, *, label: str) -> Path:
    if not p.exists():
        pytest.skip(f"NETWORK test requires {label} at: {p}")
    return p


async def test_hybrid_multi_source_rag_retrieval_pipeline() -> None:
    """
    This is a NETWORK/ENV integration test by design:
    - requires local corpus
    - requires embeddings provider (Ollama) used by EmbeddingManager
    """

    from intergrax.globals.settings import GLOBAL_SETTINGS
    from intergrax.rag.documents_loader import DocumentsLoader
    from intergrax.rag.documents_splitter import DocumentsSplitter
    from intergrax.rag.embedding_manager import EmbeddingManager
    from intergrax.rag.vectorstore_manager import VectorstoreManager, VSConfig
    from intergrax.rag.rag_retriever import RagRetriever

    import intergrax.logging  # initializes logging

    # ---- Tenant / corpus configuration
    TENANT = "intergrax"
    CORPUS = "hybrid-multi-source"
    VERSION = "v1"

    # ---- Locate local corpus ----
    repo_root = Path.cwd()
    base_docs_dir = repo_root / "documents" / "hybrid-corpus"
    pdf_dir = _require_path(base_docs_dir / "pdf", label="PDF corpus directory")
    docx_dir = _require_path(base_docs_dir / "docx", label="DOCX corpus directory")

    # ---- Instantiate core components ----
    doc_loader = DocumentsLoader(docx_mode="paragraphs")
    splitter = DocumentsSplitter()

    embed_manager = EmbeddingManager(
        provider="ollama",
        model_name=GLOBAL_SETTINGS.default_ollama_embed_model,
        assume_ollama_dim=1536,
    )

    vs_config = VSConfig(
        provider="chroma",
        collection_name="hybrid_multi_source_rag",
        chroma_persist_directory=None,  # ephemeral (test-friendly)
    )
    vectorstore = VectorstoreManager(config=vs_config)

    # ---- Load local docs via loader.load_documents ----
    pdf_docs = doc_loader.load_documents(str(pdf_dir))
    docx_docs = doc_loader.load_documents(str(docx_dir))
    all_docs = list(pdf_docs) + list(docx_docs)

    if not all_docs:
        pytest.skip("Corpus directories exist but no documents were loaded.")

    # ---- Split into chunks (splitter.split_documents(documents=...)) ----
    chunks = splitter.split_documents(documents=all_docs)
    assert chunks, "No chunks produced from local corpus."

    # ---- Embed chunks (embed_manager.embed_documents(docs=...)) ----
    embeddings, documents = embed_manager.embed_documents(docs=chunks)
    assert len(embeddings) == len(documents), "Embeddings/documents length mismatch."
    assert embeddings, "No embeddings produced."

    # ---- Prepare ids + base metadata) ----
    base_metadata = {
        "tenant": TENANT,
        "corpus": CORPUS,
        "version": VERSION,
    }

    ids = []
    for i, d in enumerate(documents):
        md = getattr(d, "metadata", {}) or {}
        chunk_id = md.get("chunk_id")
        if chunk_id:
            ids.append(str(chunk_id))
            continue

        # Fallback: stable-ish id from source + index
        src = md.get("source") or md.get("source_path") or "unknown"
        ids.append(f"{src}#ch{i}")

    # ---- Store vectors (vectorstore.add_documents) ----
    vectorstore.add_documents(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        batch_size=128,
        base_metadata=base_metadata,
    )

    # ---- Validate corpus is present ----
    total = vectorstore.count()
    assert isinstance(total, int)
    assert total > 0, "Vectorstore count is 0 after ingest."

    # ---- Retrieval (RagRetriever.retrieve) ----
    rag_retriever = RagRetriever(vectorstore, embed_manager)

    question = "Summarize the key ideas present in the local corpus."

    hits = rag_retriever.retrieve(
        question=question,
        top_k=8,
        score_threshold=0.15,
        where={"tenant": TENANT, "corpus": CORPUS, "version": VERSION},
        max_per_parent=2,
        use_mmr=True,
        include_embeddings=False,
        prefetch_factor=5,
    )

    assert isinstance(hits, list)
    assert hits, "RAG retrieval returned zero hits."
