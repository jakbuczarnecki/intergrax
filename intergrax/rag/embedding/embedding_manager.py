# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import logging

from intergrax.logging import IntergraxLogging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from intergrax.globals.settings import GLOBAL_SETTINGS

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

if TYPE_CHECKING:
    from langchain_ollama import OllamaEmbeddings

if TYPE_CHECKING:
    from langchain_openai import OpenAIEmbeddings

from langchain_core.documents import Document

logger = IntergraxLogging.get_logger(__name__, component="rag")

PROVIDERS = Literal["ollama", "hg", "openai"]


@dataclass(frozen=True)
class EmbeddingStats:
    provider: str
    model_name: str
    dim: int
    count: int


class EmbeddingManager:
    """
    Unified embedding manager for HuggingFace (SentenceTransformer), Ollama, or OpenAI embeddings.

    Features:
    - Provider switch: "hg", "ollama", "openai"
    - Reasonable defaults if model_name is None
    - Batch/single text embedding; optional L2 normalization
    - Embedding for LangChain Documents (returns np.ndarray + aligned docs)
    - Cosine similarity utilities and top-K retrieval
    - Robust logging, shape validation, light retry for transient errors
    """

    def __init__(
        self,
        provider: Optional[PROVIDERS] = None,
        model_name: Optional[str] = None,
        *,
        normalize: bool = True,
        # HF settings
        hf_device: Optional[str] = None,
        hf_batch_size: int = 32,
        hf_normalize_inside: bool = False,
        hf_max_length: Optional[int] = None,
        # Ollama settings
        ollama_probe_dim: bool = True,
        assume_ollama_dim: int = 1536,
        # OpenAI settings: use default env variables (OPENAI_API_KEY, OPENAI_BASE, etc.)
        # nothing to pass here — model and key are handled by `langchain_openai.OpenAIEmbeddings`
        retries: int = 1,
    ) -> None:        
        self.provider: PROVIDERS = provider or "ollama"
        self.model_name = model_name or self._default_model_for(self.provider)
        self.normalize = normalize

        self.hf_device = hf_device
        self.hf_batch_size = int(hf_batch_size)
        self.hf_normalize_inside = hf_normalize_inside
        self.hf_max_length = hf_max_length

        self.ollama_probe_dim = ollama_probe_dim
        self._assume_ollama_dim = int(assume_ollama_dim)

        self.retries = max(0, int(retries))

        self.model: Optional[Union[SentenceTransformer, OllamaEmbeddings, OpenAIEmbeddings]] = None # type: ignore
        self.embed_dim: Optional[int] = None

        self._load_model()

    # ----------------------
    # Model loading
    # ----------------------
    def _default_model_for(self, provider: PROVIDERS) -> str:
        if provider == "hg":
            return GLOBAL_SETTINGS.default_hf_embed_model
        elif provider == "ollama":
            return GLOBAL_SETTINGS.default_ollama_embed_model
        elif provider == "openai":
            # stable, inexpensive default model; larger one is "text-embedding-3-large"
            return GLOBAL_SETTINGS.default_openai_embed_model
        raise ValueError(f"Unknown provider: {provider}")

    def _load_model(self) -> None:
        try:
            if logger.isEnabledFor(logging.DEBUG):                    
                logger.info("[intergraxEmbeddingManager] Loading model '%s' (provider=%s)",
                            self.model_name, self.provider)

            if self.provider == "hg":
                self.model = SentenceTransformer(self.model_name, device=self.hf_device)
                if self.hf_max_length is not None:
                    try:
                        self.model.max_seq_length = int(self.hf_max_length)
                    except Exception:
                        pass
                self.embed_dim = int(self.model.get_sentence_embedding_dimension())

            elif self.provider == "ollama":
                self.model = OllamaEmbeddings(model=self.model_name)
                if self.ollama_probe_dim:
                    try:
                        test = self.model.embed_query("probe-dimension")
                        dim = len(test)
                        self.embed_dim = int(dim) if dim > 0 else self._assume_ollama_dim
                    except Exception as e:
                        logger.warning("[intergraxEmbeddingManager] Ollama dim probe failed: %s", e)
                        self.embed_dim = self._assume_ollama_dim
                else:
                    self.embed_dim = self._assume_ollama_dim

            elif self.provider == "openai":
                # Uses OPENAI_API_KEY from env; you can also pass api_key/base via OpenAIEmbeddings(...)
                self.model = OpenAIEmbeddings(model=self.model_name)
                # Probe dimension — quick single-vector request
                try:
                    test_vec = self.model.embed_query("probe-dimension")
                    self.embed_dim = int(len(test_vec)) if test_vec else None
                except Exception as e:
                    logger.exception("[intergraxEmbeddingManager] OpenAI dim probe failed: %s", e)
                    # if probing fails, assume common sizes (small=1536, large=3072)
                    self.embed_dim = 1536 if "small" in (self.model_name or "") else 3072

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            if logger.isEnabledFor(logging.DEBUG):                    
                logger.info("[intergraxEmbeddingManager] Loaded. Embedding dim = %s", self.embed_dim)

        except Exception as e:
            logger.exception("[intergraxEmbeddingManager] Error loading model '%s': %s", self.model_name, e)
            raise

    # ----------------------
    # Embedding helpers
    # ----------------------
    def _to_ndarray(self, vecs: Sequence[Sequence[float]]) -> NDArray[np.float32]:
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def _maybe_normalize(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self.normalize:
            return X
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1e-12, norms)
        return X / norms

    def _ensure_dim_known(self) -> None:
        if self.embed_dim is None:
            raise RuntimeError("Embedding dimension is unknown (model not loaded or probe failed).")

    # ----------------------
    # Public API
    # ----------------------
    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """
        Embed a list of texts. Returns a 2D numpy array [n_texts, dim].
        """
        self._ensure_dim_known()

        if not texts:
            return np.empty((0, self.embed_dim or 0), dtype=np.float32)
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        if logger.isEnabledFor(logging.DEBUG):                    
            logger.info("[intergraxEmbeddingManager] Embedding %d texts...", len(texts))

        attempts = self.retries + 1
        last_err: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                if self.provider == "hg":
                    vecs = self.model.encode(
                        list(texts),
                        batch_size=self.hf_batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=self.hf_normalize_inside,
                    )
                    X = vecs.astype(np.float32)

                elif self.provider in ("ollama", "openai"):
                    # Both providers in LC expose embed_documents for batches
                    vecs = self.model.embed_documents(list(texts))  # type: ignore[attr-defined]
                    X = self._to_ndarray(vecs)

                else:
                    raise ValueError(f"Unsupported provider at embed_texts: {self.provider}")

                X = self._maybe_normalize(X)
                return X

            except Exception as e:
                last_err = e
                logger.warning("[intergraxEmbeddingManager] Embed attempt %d/%d failed: %s",
                               attempt, attempts, e)

        logger.exception("[intergraxEmbeddingManager] All embedding attempts failed.")
        raise last_err or RuntimeError("Embedding failed for unknown reason.")

    def embed_one(self, text: str) -> NDArray[np.float32]:
        """
        Embed a single text. Returns a 1xD numpy array.
        Uses provider-specific 'query' method for Ollama/OpenAI.
        """
        self._ensure_dim_known()
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        if self.provider in ("ollama", "openai"):
            attempts = self.retries + 1
            last_err: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    vec = self.model.embed_query(text)  # type: ignore[attr-defined]
                    X = self._to_ndarray(vec)
                    return self._maybe_normalize(X)
                except Exception as e:
                    last_err = e
                    logger.warning("[intergraxEmbeddingManager] embed_query attempt %d/%d failed: %s",
                                   attempt, attempts, e)
            raise last_err or RuntimeError("embed_query failed.")
        else:
            return self.embed_texts([text])

    def embed_documents(self, docs: Sequence[Document]) -> Tuple[NDArray[np.float32], List[Document]]:
        """
        Embed LangChain Document objects based on their page_content.
        Returns: (embeddings, aligned_docs)
        """
        self._ensure_dim_known()
        if not docs:
            return np.empty((0, self.embed_dim or 0), dtype=np.float32), []
        texts = [d.page_content or "" for d in docs]
        X = self.embed_texts(texts)
        return X, list(docs)

    # ----------------------
    # Similarity utilities
    # ----------------------
    @staticmethod
    def cosine_sim_matrix(A: NDArray[np.float32], B: NDArray[np.float32]) -> NDArray[np.float32]:
        if A.size == 0 or B.size == 0:
            return np.empty((A.shape[0], B.shape[0]), dtype=np.float32)
        if A.shape[1] != B.shape[1]:
            raise ValueError(f"Dim mismatch: A is (*,{A.shape[1]}), B is (*,{B.shape[1]})")
        A_norms = np.linalg.norm(A, axis=1, keepdims=True)
        B_norms = np.linalg.norm(B, axis=1, keepdims=True)
        A_norms = np.where(A_norms == 0.0, 1e-12, A_norms)
        B_norms = np.where(B_norms == 0.0, 1e-12, B_norms)
        return (A @ B.T) / (A_norms * B_norms)

    @staticmethod
    def top_k_similar(
        query_vecs: NDArray[np.float32],
        corpus_vecs: NDArray[np.float32],
        k: int = 5,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        if corpus_vecs.size == 0 or k <= 0:
            return (
                np.empty((query_vecs.shape[0], 0), dtype=np.int64),
                np.empty((query_vecs.shape[0], 0), dtype=np.float32),
            )
        k = min(k, corpus_vecs.shape[0])
        sims = EmbeddingManager.cosine_sim_matrix(query_vecs, corpus_vecs)
        idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
        row_indices = np.arange(sims.shape[0])[:, None]
        top_scores = sims[row_indices, idx]
        order = np.argsort(-top_scores, axis=1)
        idx_sorted = idx[row_indices, order]
        scores_sorted = top_scores[row_indices, order]
        return idx_sorted, scores_sorted

    # ----------------------
    # Introspection
    # ----------------------
    def stats(self, count: int = 0) -> EmbeddingStats:
        return EmbeddingStats(
            provider=self.provider,
            model_name=self.model_name,
            dim=int(self.embed_dim or 0),
            count=int(count),
        )
