# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import hashlib
import logging
from hashlib import sha1
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from intergrax.logging import IntergraxLogging

logger = IntergraxLogging.get_logger(__name__, component="rag")

# signature: (chunk_doc, chunk_index, chunk_total) -> dict | None
ChunkMetadataFn = Callable[[Document, int, int], Optional[Dict[str, Any]]]


def _hash_text(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


class DocumentsSplitter:
    def __init__(
        self,
        *,
        default_chunk_size: int = 1000,
        default_chunk_overlap: int = 100,
        default_separators: Sequence[str] = ("\n\n", "\n", " ", ""),
        min_chunk_chars: int = 0,                 # 0 = off
        length_function: Optional[Callable[[str], int]] = None,
        drop_empty: bool = True,
        merge_small_tail: bool = True,
        tail_min_chars: int = 120,
        max_chunks_per_doc: Optional[int] = None,
    ):
        """
        High-quality text splitter for RAG pipelines (stable chunk ids + rich metadata).
        Implements 'semantic atom' policy: if a doc is already a small semantic unit
        (paragraph/row/page/image), do not split it further.
        """
        self.default_chunk_size = int(default_chunk_size)
        self.default_chunk_overlap = int(default_chunk_overlap)
        self.default_separators = tuple(default_separators)
        self.min_chunk_chars = int(min_chunk_chars)
        self.length_function = length_function or len
        self.drop_empty = drop_empty
        self.merge_small_tail = merge_small_tail
        self.tail_min_chars = int(tail_min_chars)
        self.max_chunks_per_doc = max_chunks_per_doc

        if self.default_chunk_size <= self.default_chunk_overlap:
            raise ValueError("default_chunk_size must be > default_chunk_overlap.")

    # -----------------------------
    # internals
    # -----------------------------
    @staticmethod
    def _infer_page_index(meta: Dict[str, Any]) -> Optional[int]:
        """Try to infer page index from common loader keys."""
        for key in ("page_index", "page", "page_number", "pdf_page"):
            if key in meta:
                try:
                    return int(meta[key])
                except Exception:
                    return None
        return None

    @staticmethod
    def _ensure_source_fields(doc: Document) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Returns (parent_id, source_name, source_path) derived from known metadata keys.
        """
        m = doc.metadata or {}
        parent_id = m.get("parent_id") or m.get("id") or m.get("source_path") or m.get("source") or m.get("file_name")
        source_name = m.get("source_name") or m.get("file_name") or m.get("source") or "unknown"
        source_path = m.get("source_path") or m.get("source") or m.get("file_path") or ""
        return parent_id, source_name, source_path

    @staticmethod
    def _is_semantic_atom(meta: Dict[str, Any]) -> bool:
        """
        Decide if document should be treated as an indivisible semantic atom.
        """
        # DOCX paragraph-mode
        if meta.get("doc_type") == "docx" and meta.get("para_ix") is not None:
            return True
        # Excel/CSV row
        if meta.get("excel_mode") == "rows" and meta.get("row_ix") is not None:
            return True
        # PDF page (with or without OCR)
        if meta.get("doc_type") == "pdf" and (
            meta.get("page_index") is not None or meta.get("page") is not None
        ):
            return True
        # Images
        fmt = meta.get("format")
        if fmt in {"JPEG", "PNG", "TIFF", "BMP", "WEBP"}:
            return True
        return False

    @staticmethod
    def _build_chunk_id(meta: Dict[str, Any], idx: int, content: str) -> str:
        """
        Build a stable, human-readable chunk_id using available anchors (para_ix/row_ix/page_index).
        Fallback: index + hash of content.
        """
        parent = meta.get("parent_id") or meta.get("source_path") or meta.get("source_name", "doc")
        h8 = _hash_text(content, n=8)

        # Prefer semantic anchors if present
        if meta.get("doc_type") == "docx" and meta.get("para_ix") is not None:
            sec = meta.get("section_ix")
            if sec is not None:
                return f"{parent}#sec{int(sec)}:p{int(meta['para_ix']):06d}-{h8}"
            return f"{parent}#p{int(meta['para_ix']):06d}-{h8}"

        if meta.get("excel_mode") == "rows" and meta.get("row_ix") is not None:
            return f"{parent}#row{int(meta['row_ix']):06d}-{h8}"

        pg = meta.get("page_index")
        if meta.get("doc_type") == "pdf" and pg is not None:
            return f"{parent}#pg{int(pg):05d}-{h8}"

        # Default: chunk index + hash
        return f"{parent}#ch{idx:04d}-{h8}"

    def _finalize_chunks(
        self,
        chunks: List[Document],
        *,
        parent_id: Optional[str],
        source_name: Optional[str],
        source_path: Optional[str],
        call_custom_metadata: Optional[ChunkMetadataFn],
    ) -> List[Document]:
        """
        Adds:
          - chunk_index, chunk_total
          - parent_id, source_name, source_path
          - page_index (if present upstream)
          - stable chunk_id (uses para_ix/row_ix/page_index when possible)
        Optionally merges tiny tail; applies max cap; merges custom metadata safely.
        """

        # Merge tiny last chunk with previous one (per document)
        if self.merge_small_tail and len(chunks) >= 2:
            last = chunks[-1]
            if len(last.page_content.strip()) < self.tail_min_chars:
                prev = chunks[-2]
                merged = deepcopy(prev)
                merged.page_content = (prev.page_content.rstrip() + "\n" + last.page_content.lstrip()).strip()
                # keep prev metadata; reindex below
                chunks = chunks[:-2] + [merged]

        # Optional hard cap
        if self.max_chunks_per_doc is not None and len(chunks) > self.max_chunks_per_doc:
            logger.debug(
                "[intergraxDocumentsSplitter] Cap reached: %d > %d (source=%s)",
                len(chunks), self.max_chunks_per_doc, source_name or parent_id or "unknown",
            )
            chunks = chunks[: self.max_chunks_per_doc]

        chunk_total = len(chunks)
        finalized: List[Document] = []

        # Fallback parent when none available
        base_parent = parent_id or source_name or "doc"
        if base_parent == "doc":
            seed = (chunks[0].page_content[:256] if chunks else "empty")
            base_parent = f"doc-{_hash_text(seed, n=10)}"

        for idx, ch in enumerate(chunks):
            c = deepcopy(ch)
            meta = dict(c.metadata or {})

            # Core source fields (do not overwrite existing)
            meta.setdefault("parent_id", base_parent)
            meta.setdefault("source_name", source_name or "unknown")
            if source_path:
                meta.setdefault("source_path", source_path)

            # Indexing
            meta["chunk_index"] = idx
            meta["chunk_total"] = chunk_total

            # Page index if available
            if "page_index" not in meta:
                pg = self._infer_page_index(meta)
                if pg is not None:
                    meta["page_index"] = pg

            # Stable chunk id: prefer existing; otherwise build new (with anchors if present)
            if "chunk_id" not in meta:
                meta["chunk_id"] = self._build_chunk_id(meta, idx, c.page_content)

            # Custom metadata (safe merge; do not override core keys)
            if callable(call_custom_metadata):
                try:
                    extra = call_custom_metadata(c, idx, chunk_total) or {}
                    if isinstance(extra, dict):
                        for k, v in extra.items():
                            if v is None:
                                continue
                            if k in {"chunk_id", "chunk_index", "chunk_total", "parent_id"}:
                                continue  # protect core ids
                            meta.setdefault(k, v)
                except Exception as e:                    
                    logger.exception("[intergraxDocumentsSplitter] Metadata callback error: %s", e)

            c.metadata = meta
            finalized.append(c)

        return finalized

    # -----------------------------
    # public
    # -----------------------------
    def split_documents(
        self,
        documents: List[Document],
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[Sequence[str]] = None,
        call_custom_metadata: Optional[ChunkMetadataFn] = None,
    ) -> List[Document]:
        """
        Split documents for RAG with stable chunk ids and rich metadata.

        Each chunk gets:
          - chunk_id (stable), chunk_index, chunk_total
          - parent_id, source_name, source_path
          - page_index (if present upstream)
          - plus any extras from call_custom_metadata (without overriding core ids)

        Policy:
          - If document is a semantic atom (paragraph/row/page/image), do NOT split it.
          - Otherwise, use RecursiveCharacterTextSplitter.
        """
        if not documents:
            logger.debug("[intergraxDocumentsSplitter] Empty document list.")
            return []

        eff_chunk_size = int(chunk_size or self.default_chunk_size)
        eff_overlap = int(chunk_overlap or self.default_chunk_overlap)
        if eff_chunk_size <= eff_overlap:
            raise ValueError("chunk_size must be > chunk_overlap.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=eff_chunk_size,
            chunk_overlap=eff_overlap,
            length_function=self.length_function,
            separators=list(separators or self.default_separators),
        )

        all_chunks: List[Document] = []
        total_inputs = len(documents)

        for i, doc in enumerate(documents):
            if not isinstance(doc, Document):
                continue

            meta = doc.metadata or {}

            # === 0) Semantic atom short-circuit ===
            if self._is_semantic_atom(meta):
                content = (doc.page_content or "").strip()
                if self.drop_empty and not content:
                    # do not add empty atoms
                    continue
                atom_doc = deepcopy(doc)
                atom_doc.page_content = content
                parent_id, source_name, source_path = self._ensure_source_fields(atom_doc)
                finalized = self._finalize_chunks(
                    [atom_doc],  # single chunk
                    parent_id=parent_id,
                    source_name=source_name,
                    source_path=source_path,
                    call_custom_metadata=call_custom_metadata,
                )
                all_chunks.extend(finalized)

                if logger.isEnabledFor(logging.DEBUG):
                    src = source_name or f"doc_{i}"
                    logger.debug(
                        "[intergraxDocumentsSplitter] %d/%d -> '%s': semantic-atom (1 chunk)",
                        i + 1,
                        total_inputs,
                        src,
                        extra={"data": {"index": i + 1, "total": total_inputs, "source": src}},
                    )

                continue

            # === 1) Standard split path ===
            doc_chunks = splitter.split_documents([doc])

            # === 2) Normalize / filter ===
            normalized: List[Document] = []
            for c in doc_chunks:
                content = (c.page_content or "").strip()
                if self.drop_empty and not content:
                    continue
                if self.min_chunk_chars > 0 and len(content) < self.min_chunk_chars:
                    continue
                if content != c.page_content:
                    c = deepcopy(c)
                    c.page_content = content
                normalized.append(c)

            # === 3) Source fields for this document ===
            parent_id, source_name, source_path = self._ensure_source_fields(doc)

            # === 4) Finalize chunk metadata (ids, indices, page, extras) ===
            finalized = self._finalize_chunks(
                normalized,
                parent_id=parent_id,
                source_name=source_name,
                source_path=source_path,
                call_custom_metadata=call_custom_metadata,
            )
            all_chunks.extend(finalized)

            if logger.isEnabledFor(logging.DEBUG):
                src = source_name or f"doc_{i}"
                logger.debug(
                    "[intergraxDocumentsSplitter] %d/%d -> '%s': %d chunks",
                    i + 1,
                    total_inputs,
                    src,
                    len(finalized),
                    extra={
                        "data": {
                            "index": i + 1,
                            "total": total_inputs,
                            "source": src,
                            "chunks": len(finalized),
                        }
                    },
                )


        if logger.isEnabledFor(logging.DEBUG):
            total_docs = len(documents)
            total_chunks = len(all_chunks)

            logger.debug(
                "[intergraxDocumentsSplitter] Split %d documents into %d chunks",
                total_docs,
                total_chunks,
                extra={"data": {"documents": total_docs, "chunks": total_chunks}},
            )

            if all_chunks:
                ex = all_chunks[0]
                preview = ex.page_content[:300] if ex.page_content else ""
                logger.debug(
                    "[intergraxDocumentsSplitter] Example chunk: %s ...",
                    preview.replace("\n", " "),
                )
                logger.debug(
                    "[intergraxDocumentsSplitter] Example metadata: %s",
                    ex.metadata,
                    extra={"data": {"metadata": ex.metadata}},
                )


        return all_chunks

    def make_id(self, d: Document) -> str:
        cid = d.metadata.get("chunk_id")
        if not cid:
            parent = d.metadata.get("parent_id") or d.metadata.get("source_path") or d.metadata.get("source_name", "doc")
            h8 = sha1(d.page_content.encode("utf-8", "ignore")).hexdigest()[:8]
            idx = int(d.metadata.get("chunk_index", 0))
            cid = f"{parent}#ch{idx:04d}-{h8}"
        return cid
