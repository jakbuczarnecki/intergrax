# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# === Twoje komponenty ===
from intergrax.rag.rag_answerer import (
    IntergraxRagAnswerer,
    IntergraxAnswererConfig,
    ChatMessage,
)
from intergrax.llm_adapters import LLMAdapterRegistry
from intergrax.rag.rag_retriever import IntergraxRagRetriever
from intergrax.rag.vectorstore_manager import IntergraxVectorstoreManager, VSConfig
from intergrax.rag.documents_loader import IntergraxDocumentsLoader
from intergrax.rag.documents_splitter import IntergraxDocumentsSplitter
from intergrax.rag.re_ranker import IntergraxReRanker,ReRankerConfig
from intergrax.rag.embedding_manager import IntergraxEmbeddingManager  # ⬅️ embedder

# === ustawienia środowiskowe / katalogi ===
PERSIST_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "llama3.1:latest")

# === Singletons (leniwe) ===
_vectorstore: Optional[IntergraxVectorstoreManager] = None
_embedder: Optional[IntergraxEmbeddingManager] = None
_retriever: Optional[IntergraxRagRetriever] = None
_reranker: Optional[IntergraxReRanker] = None
_answerers: Dict[str, IntergraxRagAnswerer] = {}


# ------------------------------
# Vectorstore / Embedder
# ------------------------------
def _get_vectorstore() -> IntergraxVectorstoreManager:
    global _vectorstore
    if _vectorstore is None:
        cfg = VSConfig(
            provider="chroma",
            collection_name="intergrax_docs",
            metric="cosine",
            chroma_persist_directory=PERSIST_DIR,
        )
        _vectorstore = IntergraxVectorstoreManager(config=cfg, verbose=True)
    return _vectorstore


def _get_embedder() -> IntergraxEmbeddingManager:
    global _embedder
    if _embedder is None:
        _embedder = IntergraxEmbeddingManager(provider="ollama", model_name=EMBED_MODEL)
    return _embedder


# ------------------------------
# Retriever / Reranker / LLM
# ------------------------------
def _get_retriever() -> IntergraxRagRetriever:
    global _retriever
    if _retriever is None:
        _retriever = IntergraxRagRetriever(
            vector_store=_get_vectorstore(),
            embedding_manager=_get_embedder(), 
            verbose=True)
    return _retriever


def _get_reranker() -> IntergraxReRanker:
    global _reranker
    if _reranker is None:      
        _reranker = IntergraxReRanker(
            embedding_manager=_get_embedder(),
            config=ReRankerConfig(
                use_score_fusion=True,
                fusion_alpha=0.4,
                normalize="minmax",
                doc_batch_size=256
            ),
            verbose=False
        )
    return _reranker


def _build_llm_adapter(model_name: str):
    from  langchain_ollama import ChatOllama
    chat = ChatOllama(model=model_name)
    return LLMAdapterRegistry.create("ollama", chat=chat)


def _default_user_prompt() -> str:
    return """
            Rola i zasady pracy (STRICT RAG)

            Jesteś asystentem wiedzy. Twoim jedynym źródłem informacji są dokumenty podłączone do tej rozmowy przez narzędzie file_search (vector store). Nie wolno Ci korzystać z wiedzy ogólnej ani dopowiadać faktów, których nie ma w dokumentach.

            Cel

            Odpowiadaj na pytania użytkownika wyłącznie na podstawie treści znalezionych w dokumentach bazy wiedzy.

            Odpowiedzi mają być dokładne, precyzyjne i rozwinięte, z jasnymi odniesieniami do źródeł.

            Procedura (krok po kroku)

            Zrozum pytanie. Jeśli jest wieloczęściowe, rozbij je na podzadania i pokryj każde z nich.

            Wyszukaj kontekst. Użyj file_search, pobierz wystarczającą liczbę trafień (w razie potrzeby wykonaj kilka zapytań o różnym sformułowaniu).

            Zweryfikuj spójność. Porównaj znalezione fragmenty; jeśli źródła są sprzeczne, wskaż rozbieżności i podaj możliwe interpretacje, każdą z odnośnikiem.

            Odpowiedz. Opracuj zwięzłe wnioski + szersze objaśnienie (definicje, kontekst, konsekwencje) - wyłącznie na bazie przytoczonych fragmentów.

            Cytuj. Zawsze dołącz odniesienia do źródeł (tytuł/pliku + lokalizacja: strona/sekcja/rozdział, jeśli dostępne). Gdy cytujesz kluczowe zdania, oznacz je jako cytat i podaj źródło.

            Zasady cytowania

            Po każdym kluczowym twierdzeniu dodaj nawias z referencją, np.:
            (Źródło: 'nazwa_pliku', s. 'strona') lub (Źródło: 'nazwa_pliku', sekcja 'sekcja').

            Przy dłuższej odpowiedzi dodaj na końcu sekcję „Źródła” z listą pozycji.

            Cytaty dosłowne używaj oszczędnie i tylko gdy są niezbędne; nie przekraczaj krótkich fragmentów.

            Granice i niepewność

            Jeśli w dokumentach brakuje danych do pełnej odpowiedzi, powiedz to wprost:
            „Na podstawie dostępnych dokumentów nie mogę jednoznacznie odpowiedzieć na X.”
            Następnie:

            wskaż, jakich informacji brakuje (np. nazwa sekcji/rodzaj dokumentu),

            zaproponuj konkretne frazy do doszukania w bazie lub dodania nowych plików.

            Nie przywołuj wiedzy spoza dokumentów. Nie spekuluj. Jeśli musisz sformułować wniosek, oprzyj go na przytoczonych fragmentach i oznacz jako „Wniosek na podstawie źródeł”.

            Styl odpowiedzi

            Najpierw krótkie podsumowanie (2-4 zdania z sednem odpowiedzi).

            Potem szczegółowe wyjaśnienie (krok po kroku, listy punktowane, małe nagłówki).

            Precyzyjna terminologia, zero ogólników.

            Jeśli pytanie dotyczy procedury/algorytmu/listy wymagań - przygotuj listę kontrolną lub pseudo-procedurę.

            Jeśli pytanie dotyczy liczb/zakresów - podaj konkretne wartości z cytatami.

            Format wynikowy (gdy to możliwe)

            Podsumowanie

            Szczegóły i uzasadnienie (z odnośnikami w tekście)

            Źródła (lista: nazwa pliku + strona/sekcja)

            Zakazy (ważne)

            Nie używaj informacji, których nie znalazłeś w dokumentach.

            Nie odwołuj się do „wiedzy powszechnej”, internetu ani własnych domysłów.

            Nie ukrywaj niepewności - jeśli coś nie wynika z materiałów, powiedz to.

            (Opcjonalnie) Przykładowe odniesienia

            „… zgodnie z definicją procesu (Źródło: Specyfikacja_Proces_A.pdf, s. 12) …”

            „… wymagania niefunkcjonalne: dostępność 99.9% (Źródło: Wymagania_Systemowe.docx, sekcja 3.2) …
            
            >>Najważniejsza zasada<<
            Zawsze odpowiadaj użytkownikowi jak najdokładniej na podstawie dokumentów z bazy wiedzy.
            Rozwijaj odpowiedź maksymalnie dokładnie używając jak największej liczby słów.
            Użytkownik musi mieć odczucie, że rozmawia asystentem, który ma ogromną wiedzę w zakresie intergrax i potrafi ją w sposób dokładny przekazać.

            Kontekst:
            {context}

            {history_block}

            Pytanie użytkownika: {question}
            """


def _default_system_prompt() -> str:
    return """
            Rola i zasady pracy (STRICT RAG)

            Jesteś asystentem wiedzy. Twoim jedynym źródłem informacji są dokumenty podłączone do tej rozmowy przez narzędzie file_search (vector store). Nie wolno Ci korzystać z wiedzy ogólnej ani dopowiadać faktów, których nie ma w dokumentach.

            >>Najważniejsza zasada<<
            Zawsze odpowiadaj użytkownikowi jak najdokładniej na podstawie dokumentów z bazy wiedzy.
            Rozwijaj odpowiedź maksymalnie dokładnie używając jak największej liczby słów.
            Użytkownik musi mieć odczucie, że rozmawia asystentem, który ma ogromną wiedzę w zakresie intergrax i potrafi ją w sposób dokładny przekazać.

            """


def get_answerer(model_name: Optional[str] = None) -> IntergraxRagAnswerer:
    global _answerers
    name = model_name or DEFAULT_MODEL
    if name in _answerers:
        return _answerers[name]

    llm = _build_llm_adapter(name)

    # create answerer configuration
    cfg = IntergraxAnswererConfig(
        top_k=10,
        min_score=0.15,
        use_history=True,
        re_rank_k=5,             # włącz rerank (jeśli podałeś reranker)
        max_context_chars=12000,
        history_turns=1,         # dołącz 1 poprzednie Q/A do promptu    
    )
    cfg.system_prompt = _default_system_prompt()
    cfg.user_prompt_template = _default_user_prompt()

    # cfg.user_prompt_template=prompts.context_prompt_template().replace("{query}", "{question}")


    # create rag answerer (retriever + LLM)
    answerer = IntergraxRagAnswerer(
        retriever=_get_retriever(),
        llm=llm,
        reranker=_get_reranker(),  
        config=cfg,
        verbose=True,
    )

    _answerers[name] = answerer
    return answerer


# ======================
# Ingest: load -> split -> add_documents
# ======================
def load_and_split_documents(file_path: str):
    p = Path(file_path)

    loader = IntergraxDocumentsLoader(
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

    splitter = IntergraxDocumentsSplitter()
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
def set_history(answerer: IntergraxRagAnswerer, history_pairs: List[Tuple[str, str]]) -> None:
    """
    history_pairs: lista (user_text, assistant_text) w porządku chronologicznym.
    Nadpisuje wewnętrzną historię answerera.
    """
    answerer._history.clear()
    for u, a in history_pairs:
        answerer._history.append(ChatMessage(role="user", content=u))
        answerer._history.append(ChatMessage(role="assistant", content=a))
