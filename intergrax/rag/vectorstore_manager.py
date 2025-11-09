# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from langchain_core.documents import Document

# --- ChromaDB ---
import chromadb
from chromadb.config import Settings as ChromaSettings

# --- Qdrant ---
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter as QFilter,
        FieldCondition,
        MatchValue,
        PointIdsList,
    )
except ImportError:
    QdrantClient = None  # type: ignore
    Distance = None      # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None   # type: ignore
    QFilter = None       # type: ignore
    FieldCondition = None # type: ignore
    MatchValue = None    # type: ignore
    PointIdsList = None  # type: ignore

# --- Pinecone ---
try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None  # type: ignore

VectorProvider = Literal["chroma", "qdrant", "pinecone"]
Metric = Literal["cosine", "dot", "euclidean"]


@dataclass
class VSConfig:
    """Generic vector-store configuration."""
    provider: VectorProvider
    collection_name: str = "documentation"
    metric: Metric = "cosine"

    # Chroma
    chroma_persist_directory: Optional[str] = "data/vector_store"
    chroma_settings: Optional[ChromaSettings] = None

    # Qdrant
    qdrant_url: Optional[str] = None  # e.g., "http://localhost:6333" or Cloud endpoint
    qdrant_api_key: Optional[str] = None

    # Pinecone
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None  # if None, defaults to collection_name
    pinecone_cloud: Optional[str] = None       # e.g., "aws" (optional)
    pinecone_region: Optional[str] = None      # e.g., "us-east-1" (optional)


class IntergraxVectorstoreManager:
    """
    Unified vector store manager supporting ChromaDB, Qdrant, and Pinecone.

    Features:
    - Initialize target store and (if needed) create collection/index (lazy for Qdrant/Pinecone)
    - Upsert documents + embeddings (with batching)
    - Query top-K by cosine/dot/euclidean similarity
    - Count vectors
    - Delete by ids

    Assumptions:
    - `embeddings` is either a numpy array of shape [N, D] or list[list[float]]
    - `documents` is a list of LangChain `Document` with text in `.page_content`
    """

    # ------------------------------
    # Construction
    # ------------------------------
    def __init__(self, config: VSConfig, *, verbose: bool = True) -> None:
        self.cfg = config
        self.verbose = verbose

        self.provider: VectorProvider = config.provider
        self.collection_name = config.collection_name
        self.metric = config.metric

        self._client = None
        self._collection = None  # Chroma collection or Pinecone Index handle; Qdrant uses client only
        self._dim: Optional[int] = None  # remember vector dimension after first upsert if needed

        self._initialize_store()

    # ------------------------------
    # Initialization
    # ------------------------------
    def _initialize_store(self) -> None:
        try:
            if self.provider == "chroma":
                self._init_chroma()
            elif self.provider == "qdrant":
                self._init_qdrant()
            elif self.provider == "pinecone":
                self._init_pinecone()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            if self.verbose:
                print(f"[intergraxVectorstoreManager] Initialized provider={self.provider}, collection={self.collection_name}")
                # count() may create indexes lazily (pinecone/qdrant) – that's fine.
                print(f"[intergraxVectorstoreManager] Existing count: {self.count()}")

        except Exception as e:
            print(f"[intergraxVectorstoreManager] Error initializing vector store: {e}")
            raise

    def _init_chroma(self) -> None:
        # Create Chroma client (persistent or in-memory)
        settings = self.cfg.chroma_settings or ChromaSettings()

        persist_dir = self.cfg.chroma_persist_directory
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir, settings=settings)
        else:
            # In newer versions you can use EphemeralClient, but Client is a safe choice.
            self._client = chromadb.Client(settings=settings)

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Document embeddings for intergrax system"},
        )

    def _init_qdrant(self) -> None:
        if QdrantClient is None:
            raise ImportError("qdrant-client is not installed. `pip install qdrant-client`")

        if self.cfg.qdrant_url:
            self._client = QdrantClient(url=self.cfg.qdrant_url, api_key=self.cfg.qdrant_api_key)
        else:
            # Local default
            self._client = QdrantClient(host="localhost", port=6333, api_key=self.cfg.qdrant_api_key)
        # Collection is created lazily when we know the dimension (first upsert or explicit ensure_collection()).

    def _init_pinecone(self) -> None:
        if Pinecone is None:
            raise ImportError("pinecone client is not installed. `pip install pinecone-client`")
        if not self.cfg.pinecone_api_key:
            raise ValueError("Pinecone requires `pinecone_api_key` in VSConfig.")

        pc = Pinecone(api_key=self.cfg.pinecone_api_key)
        self._client = pc

        index_name = self.cfg.pinecone_index_name or self.collection_name
        self.collection_name = index_name  # unify naming

        # If the index exists, keep a handle; otherwise create it lazily.
        try:
            self._collection = pc.Index(index_name)
        except Exception:
            self._collection = None  # create lazily when we know dim

    # ------------------------------
    # Helpers
    # ------------------------------
    @staticmethod
    def _to_list_of_lists(emb: Union[NDArray[np.float32], Sequence[Sequence[float]]]) -> List[List[float]]:
        if isinstance(emb, np.ndarray):
            if emb.ndim == 1:
                emb = np.expand_dims(emb, axis=0)
            return emb.astype(np.float32).tolist()
        return [list(map(float, v)) for v in emb]

    @staticmethod
    def _doc_texts(docs: Sequence[Document]) -> List[str]:
        return [d.page_content or "" for d in docs]

    @staticmethod
    def _doc_payloads(docs: Sequence[Document], base: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        base = base or {}
        out: List[Dict[str, Any]] = []
        for d in docs:
            md = dict(base)
            md.update(dict(d.metadata))
            out.append(md)
        return out

    @staticmethod
    def _make_ids(n: int, prefix: str = "doc") -> List[str]:
        return [f"{prefix}_{uuid.uuid4().hex[:8]}_{i}" for i in range(n)]

    def _ensure_dim_consistency(self, batch: Sequence[Sequence[float]]):
        if not batch:
            return
        if self._dim is None:
            self._dim = len(batch[0])
        else:
            bad = [i for i, v in enumerate(batch) if len(v) != self._dim]
            if bad:
                raise ValueError(
                    f"Inconsistent embedding dimension in batch at positions {bad[:5]} (expected {self._dim})."
                )

    def _pinecone_metric(self) -> str:
        # Pinecone uses “dotproduct” instead of “dot”
        mapping = {"cosine": "cosine", "euclidean": "euclidean", "dot": "dotproduct"}
        return mapping.get(self.metric, "cosine")

    def _ensure_qdrant_collection(self) -> None:
        assert self._client is not None, "Qdrant client is not initialized"
        assert self._dim is not None, "Embedding dim unknown; cannot create Qdrant collection."

        # Map metric
        metric_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        dist = metric_map.get(self.metric, Distance.COSINE)

        try:
            # If the collection does not exist, get_collection will raise
            self._client.get_collection(self.collection_name)
        except Exception:
            # Create only if it does not exist (no destructive recreate)
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self._dim, distance=dist),
            )

    def _ensure_pinecone_index(self) -> None:
        assert self._client is not None, "Pinecone client is not initialized"
        assert self._dim is not None, "Embedding dim unknown; cannot create Pinecone index."

        pc: Pinecone = self._client  # type: ignore
        index_name = self.collection_name
        try:
            # If exists — just get a handle
            self._collection = pc.Index(index_name)
        except Exception:
            # Create index
            pc.create_index(
                name=index_name,
                dimension=self._dim,
                metric=self._pinecone_metric(),
                # cloud=self.cfg.pinecone_cloud,
                # region=self.cfg.pinecone_region,
            )
            self._collection = pc.Index(index_name)

    def _qdrant_filter(self, where: Optional[Dict[str, Any]]) -> Optional[QFilter]:  # type: ignore
        """Lightweight helper: simple dict -> Filter(must=[FieldCondition(...)])."""
        if not where or QFilter is None:
            return None
        must: List[Dict[str, Any]] = []
        for k, v in where.items():
            # simple equality
            must.append({"key": k, "match": {"value": v}})
        return QFilter(**{"must": must})

    # ------------------------------
    # Public: optional explicit ensure
    # ------------------------------
    def ensure_collection(self, dim: int) -> None:
        """
        (Optional) Explicitly create collection/index if it does not exist yet.
        Lets you enforce parameters ahead of upserts.
        """
        self._dim = self._dim or int(dim)
        if self.provider == "qdrant":
            self._ensure_qdrant_collection()
        elif self.provider == "pinecone":
            self._ensure_pinecone_index()
        elif self.provider == "chroma":
            # Chroma is created at init
            pass
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ------------------------------
    # Upsert
    # ------------------------------
    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Union[NDArray[np.float32], Sequence[Sequence[float]]],
        *,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 256,
        store_text_in_metadata_for: Optional[Sequence[VectorProvider]] = ("qdrant", "pinecone"),
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Upsert documents + embeddings into the configured vector store.

        Notes:
        - For Chroma we store `documents` field (raw text) and `metadatas`.
        - For Qdrant/Pinecone we put text into metadata under key `"text"` (configurable via `store_text_in_metadata_for`).
        - Qdrant/Pinecone collections/indexes are created lazily on the first call when `dim` is known.
        """
        if len(documents) == 0:
            if self.verbose:
                print("[intergraxVectorstoreManager] No documents to add.")
            return

        X = self._to_list_of_lists(embeddings)
        if len(X) != len(documents):
            raise ValueError("Number of documents must match number of embeddings")

        n = len(documents)
        ids = list(ids) if ids else self._make_ids(n)
        if len(ids) != n:
            raise ValueError("Length of `ids` must match number of documents")

        # Remember/check dimension for lazy creation
        first_dim = len(X[0]) if X and X[0] else None
        if first_dim is None:
            raise ValueError("Embeddings appear empty/corrupt; cannot infer dimension.")
        self._dim = self._dim or first_dim

        if self.verbose:
            print(f"[intergraxVectorstoreManager] Upserting {n} items (dim={self._dim}) to provider={self.provider}...")

        # Perform batched upsert
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ids_batch = ids[start:end]
            embeddings_batch = X[start:end]
            self._ensure_dim_consistency(embeddings_batch)

            docs_batch = documents[start:end]
            metas_batch = self._doc_payloads(docs_batch, base=base_metadata)

            # Provider-specific upsert
            if self.provider == "chroma":
                self._upsert_chroma(ids_batch, embeddings_batch, metas_batch, self._doc_texts(docs_batch))
            elif self.provider == "qdrant":
                if store_text_in_metadata_for and "qdrant" in store_text_in_metadata_for:
                    for i, d in enumerate(docs_batch):
                        metas_batch[i] = dict(metas_batch[i], text=d.page_content or "")
                self._upsert_qdrant(ids_batch, embeddings_batch, metas_batch)
            elif self.provider == "pinecone":
                if store_text_in_metadata_for and "pinecone" in store_text_in_metadata_for:
                    for i, d in enumerate(docs_batch):
                        metas_batch[i] = dict(metas_batch[i], text=d.page_content or "")
                self._upsert_pinecone(ids_batch, embeddings_batch, metas_batch)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        if self.verbose:
            print(f"[intergraxVectorstoreManager] Upsert complete. New count: {self.count()}")

    # --- Provider-specific upsert implementations ---
    def _upsert_chroma(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        documents: Sequence[str],
    ) -> None:
        # Some Chroma versions may not have upsert; fall back to add.
        try:
            self._collection.upsert(
                ids=list(ids),
                embeddings=list(embeddings),
                metadatas=list(metadatas),
                documents=list(documents),
            )
        except AttributeError:
            self._collection.add(
                ids=list(ids),
                embeddings=list(embeddings),
                metadatas=list(metadatas),
                documents=list(documents),
            )

    def _upsert_qdrant(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        self._ensure_qdrant_collection()
        points = [
            PointStruct(
                id=ids[i],
                vector=list(map(float, embeddings[i])),
                payload=metadatas[i],
            )
            for i in range(len(ids))
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def _upsert_pinecone(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        self._ensure_pinecone_index()
        vectors = [
            {"id": ids[i], "values": list(map(float, embeddings[i])), "metadata": metadatas[i]}
            for i in range(len(ids))
        ]
        # Different client versions use different parameter names
        try:
            self._collection.upsert(vectors=vectors)
        except TypeError:
            self._collection.upsert(items=vectors)

    # ------------------------------
    # Query
    # ------------------------------
    def query(
        self,
        query_embeddings: Union[NDArray[np.float32], Sequence[Sequence[float]]],
        top_k: int = 5,
        *,
        where: Optional[Dict[str, Any]] = None,  # filter/payload filter
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the vector store by embeddings.

        Returns provider-normalized-ish result dict:
          {
            "ids": List[List[str]],
            "scores": List[List[float]],     # note: for Chroma cosine -> 1 - distance (approximate ascending “score”)
            "metadatas": List[List[dict]],
            "documents": List[List[str]] | None,
          }
        """
        Q = self._to_list_of_lists(query_embeddings)

        if self.provider == "chroma":
            include = ["metadatas", "documents", "distances"]
            if include_embeddings:
                include.append("embeddings")

            res = self._collection.query(
                query_embeddings=Q,
                n_results=top_k,
                where=where or None,   # important: None instead of {}
                include=include,
            )

            distances = res.get("distances", [])
            if self.metric == "cosine":
                # convert to “ascending score” for consumer convenience
                scores = [[1.0 - float(d) for d in row] for row in distances]
            else:
                # for other metrics leave as-is (or add your own normalization)
                scores = distances

            return {
                "ids": res.get("ids", []),
                "scores": scores,
                "metadatas": res.get("metadatas", []),
                "documents": res.get("documents", []),
            }

        elif self.provider == "qdrant":
            self._ensure_qdrant_collection()
            out = {"ids": [], "scores": [], "metadatas": [], "documents": []}
            for q in Q:
                qr = self._client.search(
                    collection_name=self.collection_name,
                    query_vector=list(map(float, q)),
                    limit=top_k,
                    query_filter=self._qdrant_filter(where),
                )
                out["ids"].append([str(h.id) for h in qr])
                out["scores"].append([float(h.score) for h in qr])  # ascending = better (Qdrant returns similarity)
                out["metadatas"].append([dict(h.payload or {}) for h in qr])
                out["documents"].append([str((h.payload or {}).get("text", "")) for h in qr])
            return out

        elif self.provider == "pinecone":
            self._ensure_pinecone_index()
            out = {"ids": [], "scores": [], "metadatas": [], "documents": []}
            for q in Q:
                qr = self._collection.query(
                    vector=list(map(float, q)),
                    top_k=top_k,
                    include_values=False,
                    include_metadata=True,
                    filter=where or None,
                )
                matches = qr.get("matches", []) if isinstance(qr, dict) else qr.matches
                ids_row, scores_row, mds_row, docs_row = [], [], [], []
                for m in matches:
                    if isinstance(m, dict):
                        ids_row.append(m.get("id"))
                        scores_row.append(float(m.get("score", 0.0)))
                        md = m.get("metadata", {}) or {}
                    else:
                        ids_row.append(m.id)
                        scores_row.append(float(m.score))
                        md = m.metadata or {}
                    mds_row.append(md)
                    docs_row.append(str(md.get("text", "")))
                out["ids"].append(ids_row)
                out["scores"].append(scores_row)  # Pinecone: higher score = better
                out["metadatas"].append(mds_row)
                out["documents"].append(docs_row)
            return out

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ------------------------------
    # Count
    # ------------------------------
    def count(self) -> int:
        if self.provider == "chroma":
            return int(self._collection.count())
        elif self.provider == "qdrant":
            self._ensure_qdrant_collection()
            c = self._client.count(self.collection_name, exact=True)
            # Newer versions return an object with a `.count` field
            try:
                return int(c.count)  # type: ignore[attr-defined]
            except Exception:
                # Fallback: if the client returns a dict-like or different shape
                return int(getattr(c, "count", 0))
        elif self.provider == "pinecone":
            self._ensure_pinecone_index()
            try:
                stats = self._collection.describe_index_stats()
                return int(stats.get("total_vector_count", 0))
            except Exception:
                # Some versions expose describe_index_stats on the client
                stats = self._client.describe_index_stats(index_name=self.collection_name)
                return int(stats.get("total_vector_count", 0))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ------------------------------
    # Delete by ids
    # ------------------------------
    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        if self.provider == "chroma":
            self._collection.delete(ids=list(ids))
        elif self.provider == "qdrant":
            self._ensure_qdrant_collection()
            try:
                if PointIdsList is not None:
                    self._client.delete(
                        collection_name=self.collection_name,
                        points_selector=PointIdsList(points=list(ids)),
                    )
                else:
                    # Older client versions accepted a dict selector
                    self._client.delete(self.collection_name, points_selector={"points": list(ids)})
            except TypeError:
                # Backward compatibility
                self._client.delete(self.collection_name, points_selector={"points": list(ids)})
        elif self.provider == "pinecone":
            self._ensure_pinecone_index()
            self._collection.delete(ids=list(ids))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
