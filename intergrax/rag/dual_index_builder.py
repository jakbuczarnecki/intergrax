# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import logging
import math
from typing import List, Dict, Callable, Optional
from langchain_core.documents import Document
from .vectorstore_manager import VectorstoreManager
from .embedding_manager import EmbeddingManager

logger = logging.getLogger("intergrax.dual_index_builder")


def build_dual_index(
    *,
    docs: List[Document],
    embed_manager: EmbeddingManager,                 # intergraxEmbeddingManager
    vs_chunks: VectorstoreManager,                   # main collection (CHUNKS)
    vs_toc: Optional[VectorstoreManager] = None,     # lightweight collection (TOC)
    batch_size: int = 512,
    make_toc_from_docx_headings: bool = True,
    toc_min_level: int = 1,
    toc_max_level: int = 3,
    prefilter: Optional[Callable[[Document], bool]] = None,
    skip_if_populated: bool = True,    
    verbose: bool = False,
):
    """
    Builds two vector indexes: primary (CHUNKS) and auxiliary (TOC).
    - CHUNKS: all chunks/documents after splitting.
    - TOC: only DOCX headings within levels [toc_min_level, toc_max_level].
    """
    log = logger.getChild("build")
    if verbose:
        log.setLevel(logging.INFO)

    def _safe_count(vs: VectorstoreManager) -> int:
        try:
            return int(vs.count() or 0)
        except Exception:
            return 0

    if skip_if_populated:
        chunks_count = _safe_count(vs_chunks)
        toc_count = _safe_count(vs_toc) if vs_toc is not None else 0

        if chunks_count > 0 or toc_count > 0:
            log.warning(
                "[DualIndex] skip_if_populated=True → skipping ingest "
                "(CHUNKS=%d, TOC=%d).",
                chunks_count, toc_count
            )
            return

    # sanity for level range
    if toc_min_level > toc_max_level:
        toc_min_level, toc_max_level = toc_max_level, toc_min_level

    total = len(docs)
    log.info("[DualIndex] Start (input=%d)", total)

    chunk_docs: List[Document] = []
    toc_docs: List[Document] = []

    for d in docs:
        if prefilter and not prefilter(d):
            continue

        text = (d.page_content or "").strip()
        if not text:
            continue

        md: Dict = dict(d.metadata or {})
        # simplified source name (e.g., file name)
        if "source_name" not in md:
            sp = md.get("source_path") or md.get("path") or md.get("file_name") or md.get("filename") or "unknown"
            md["source_name"] = str(sp).split("/")[-1].split("\\")[-1]

        # 1) CHUNKS: every doc/chunk goes to the primary collection
        chunk_docs.append(Document(page_content=text, metadata=md))

        # 2) TOC: from DOCX headings (if enabled and we have a TOC collection)
        if (
            vs_toc is not None
            and make_toc_from_docx_headings
            and md.get("doc_type") == "docx"
            and md.get("is_heading") is True
        ):
            # determine level
            level = md.get("heading_level")
            if level is None:
                hp = md.get("heading_path") or ""
                level = 1 + hp.count(" / ") if hp else 1
            try:
                ilevel = int(level)
            except Exception:
                ilevel = toc_min_level  # fallback

            if toc_min_level <= ilevel <= toc_max_level:
                # shorten heading content to a reasonable length (e.g., 512 chars)
                toc_docs.append(Document(page_content=text[:512], metadata=md))

    log.info("[DualIndex] Prepared: chunks=%d, toc=%d", len(chunk_docs), len(toc_docs))

    # --- CHUNKS ---
    if chunk_docs:
        # compute embeddings (returns (embeddings, aligned_docs))
        X_chunks, aligned_chunks = embed_manager.embed_documents(chunk_docs)

        n = len(aligned_chunks)
        total_batches = math.ceil(n / batch_size)
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            vs_chunks.add_documents(
                documents=aligned_chunks[i:j],
                embeddings=X_chunks[i:j],
                batch_size=batch_size,
            )
            if verbose:
                log.info(
                    "[DualIndex] CHUNKS batch %d/%d inserted (%d items)",
                    (i // batch_size) + 1, total_batches, j - i
                )
        try:
            log.info("[DualIndex] CHUNKS done (count now ~%d)", vs_chunks.count())
        except Exception:
            log.info("[DualIndex] CHUNKS done")

    # --- TOC ---
    if vs_toc is not None:
        if toc_docs:
            X_toc, aligned_toc = embed_manager.embed_documents(toc_docs)

            n = len(aligned_toc)
            total_batches = math.ceil(n / batch_size)
            for i in range(0, n, batch_size):
                j = min(i + batch_size, n)
                vs_toc.add_documents(
                    documents=aligned_toc[i:j],
                    embeddings=X_toc[i:j],
                    batch_size=batch_size,
                )
                if verbose:
                    log.info(
                        "[DualIndex] TOC batch %d/%d inserted (%d items)",
                        (i // batch_size) + 1, total_batches, j - i
                    )
            try:
                log.info("[DualIndex] TOC done (count now ~%d)", vs_toc.count())
            except Exception:
                log.info("[DualIndex] TOC done")
        else:
            log.warning("[DualIndex] TOC enabled, but no DOCX headings matched the criteria.")
    
    log.info("[DualIndex] Done.")
