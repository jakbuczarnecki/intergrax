# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# === Twoje komponenty ===
from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm_adapters.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.rag.rag_answerer import (
    RagAnswerer,
    AnswererConfig,
    ChatMessage,
)

from intergrax.rag.rag_retriever import RagRetriever
from intergrax.rag.vectorstore_manager import VectorstoreManager, VSConfig
from intergrax.rag.documents_loader import DocumentsLoader
from intergrax.rag.documents_splitter import DocumentsSplitter
from intergrax.rag.re_ranker import ReRanker,ReRankerConfig
from intergrax.rag.embedding_manager import EmbeddingManager

# === ustawienia środowiskowe / katalogi ===
PERSIST_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = GLOBAL_SETTINGS.default_ollama_embed_model
DEFAULT_MODEL = GLOBAL_SETTINGS.default_ollama_model

# === Singletons (leniwe) ===
_vectorstore: Optional[VectorstoreManager] = None
_embedder: Optional[EmbeddingManager] = None
_retriever: Optional[RagRetriever] = None
_reranker: Optional[ReRanker] = None
_answerers: Dict[str, RagAnswerer] = {}


# ------------------------------
# Vectorstore / Embedder
# ------------------------------
def _get_vectorstore() -> VectorstoreManager:
    global _vectorstore
    if _vectorstore is None:
        cfg = VSConfig(
            provider="chroma",
            collection_name="intergrax_docs",
            metric="cosine",
            chroma_persist_directory=PERSIST_DIR,
        )
        _vectorstore = VectorstoreManager(config=cfg)
    return _vectorstore


def _get_embedder() -> EmbeddingManager:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingManager(provider="ollama", model_name=EMBED_MODEL)
    return _embedder


# ------------------------------
# Retriever / Reranker / LLM
# ------------------------------
def _get_retriever() -> RagRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RagRetriever(
            vector_store=_get_vectorstore(),
            embedding_manager=_get_embedder(),
        )
    return _retriever


def _get_reranker() -> ReRanker:
    global _reranker
    if _reranker is None:      
        _reranker = ReRanker(
            embedding_manager=_get_embedder(),
            config=ReRankerConfig(
                use_score_fusion=True,
                fusion_alpha=0.4,
                normalize="minmax",
                doc_batch_size=256
            ),
        )
    return _reranker


def _load_rag_pipeline_prompts() -> tuple[str, str]:
    registry = YamlPromptRegistry.create_default(load=True)

    localized = registry.resolve_localized("rag_pipeline")

    system = (localized.system or "").rstrip("\n")
    user = (localized.user_template or "").rstrip("\n")

    return system, user


def get_answerer(model_name: Optional[str] = None) -> RagAnswerer:
    global _answerers
    name = model_name or DEFAULT_MODEL
    if name in _answerers:
        return _answerers[name]

    llm = LLMAdapterRegistry.create(LLMProvider.OLLAMA)

    # create answerer configuration
    cfg = AnswererConfig(
        top_k=10,
        min_score=0.15,
        use_history=True,
        re_rank_k=5,
        max_context_chars=12000,
        history_turns=1,
    )
    
    cfg.ensure_prompts()

    system_p, user_p = _load_rag_pipeline_prompts()
    cfg.system_prompt = system_p
    cfg.user_prompt_template = user_p

    # cfg.user_prompt_template=prompts.context_prompt_template().replace("{query}", "{question}")


    # create rag answerer (retriever + LLM)
    answerer = RagAnswerer(
        retriever=_get_retriever(),
        llm=llm,
        reranker=_get_reranker(),  
        config=cfg,
    )

    _answerers[name] = answerer
    return answerer


# ======================
# Ingest: load -> split -> add_documents
# ======================
def load_and_split_documents(file_path: str):
    p = Path(file_path)

    loader = DocumentsLoader(
        # tu możesz doprecyzować tryby (np. docx_mode="paragraphs", pdf_enable_ocr=True, itd.)
    )

    def _base_metadata(doc, absolute_path: Path):
        return {
            "source_path": str(absolute_path),
            "source_name": absolute_path.name,
        }

    docs = loader.load_document(
        file_path=str(p),
        use_default_metadata=True,
        call_custom_metadata=_base_metadata,
    )

    splitter = DocumentsSplitter()
    splits = splitter.split_documents(docs)

    return splits


def index_document_to_vectorstore(file_path: str, file_id: int) -> bool:
    # 1) load + split
    splits = load_and_split_documents(file_path)

    # 2) dopisz file_id do KAŻDEGO chunku
    for d in splits:
        meta = getattr(d, "metadata", None)
        if isinstance(meta, dict):
            meta["file_id"] = file_id
        else:
            setattr(d, "metadata", {"file_id": file_id})

    # 3) embeddings
    embedder = _get_embedder()
    try:
        # preferowane API, jeśli embedder potrafi przyjąć listę Document
        embeddings, docs = embedder.embed_documents(splits)
    except TypeError:
        # fallback: gdy embedder wymaga listy tekstów
        embeddings = embedder.embed_texts([d.page_content or "" for d in splits])

    # 4) upsert do VS (Twoje API: add_documents)
    vs = _get_vectorstore()
    vs.add_documents(
        documents=splits,
        embeddings=embeddings,
        # opcjonalnie: ids=[...], batch_size=256, base_metadata={...}
    )
    return True


# ======================
# Delete by file_id (Chroma/Qdrant/Pinecone)
# ======================
def delete_by_file_id(file_id: int) -> bool:
    """
    Usuwa wszystkie wektory o metadata.file_id == file_id.
    - Chroma: używa where
    - Qdrant: używa delete z filtrem
    - Pinecone: używa delete(filter=...)
    """
    vs = _get_vectorstore()

    try:
        provider = getattr(vs, "provider", "chroma")
        if provider == "chroma":
            # publiczne API menedżera nie ma delete_where, więc korzystamy z uchwytu kolekcji
            # (zgodnie z tym jak już robiłeś to wcześniej)
            _col = getattr(vs, "_collection", None)
            if _col is None:
                raise RuntimeError("Chroma collection not initialized")
            _col.delete(where={"file_id": file_id})
            return True

        elif provider == "qdrant":
            _cli = getattr(vs, "_client", None)
            cname = getattr(vs, "collection_name", None)
            if _cli is None or cname is None:
                raise RuntimeError("Qdrant client or collection not initialized")

            # Lekki filtr kompatybilny ze strukturą Qdrant Filter
            qfilter = {"must": [{"key": "file_id", "match": {"value": file_id}}]}
            # Różne wersje klienta akceptują dict jako selector
            _cli.delete(
                collection_name=cname,
                points_selector={"filter": qfilter},
            )
            return True

        elif provider == "pinecone":
            _idx = getattr(vs, "_collection", None)
            if _idx is None:
                raise RuntimeError("Pinecone index not initialized")
            # Pinecone wspiera filter delete
            _idx.delete(filter={"file_id": file_id})
            return True

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        print(f"[rag_pipeline.delete_by_file_id] Error while deleting file_id={file_id}: {e}")
        return False


# ======================
# Historia → wstrzyknięcie do answerera
# ======================
def set_history(answerer: RagAnswerer, history_pairs: List[Tuple[str, str]]) -> None:
    """
    history_pairs: lista (user_text, assistant_text) w porządku chronologicznym.
    Nadpisuje wewnętrzną historię answerera.
    """
    answerer._history.clear()
    for u, a in history_pairs:
        answerer._history.append(ChatMessage(role="user", content=u))
        answerer._history.append(ChatMessage(role="assistant", content=a))
