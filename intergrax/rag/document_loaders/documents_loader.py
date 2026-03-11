# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

from tqdm import tqdm
from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.base_document_loader import BaseDocumentsLoader
from intergrax.rag.document_loaders.pipeline.metadata_pipeline import MetadataPipeline
from intergrax.rag.document_loaders.pipeline.normalizer_pipeline import NormalizerPipeline
from intergrax.rag.document_loaders.registry.document_handler_registry import DocumentHandlerRegistry

logger = logging.getLogger(__name__)


class DocumentsLoader(BaseDocumentsLoader):
    """
    Entry point for document ingestion.

    Responsibilities
    ----------------
    - file discovery
    - handler resolution
    - document loading
    - metadata enrichment
    """

    def __init__(
        self,
        *,
        registry: DocumentHandlerRegistry,
        normalizer_pipeline: NormalizerPipeline,
        metadata_pipeline: MetadataPipeline,
        allowed_exts: Optional[Sequence[str]] = None,
        file_patterns: Optional[Sequence[str]] = None,
        follow_symlinks: bool = False,
        max_files: Optional[int] = None,
    ) -> None:

        self._registry = registry
        self._normalizer_pipeline = normalizer_pipeline
        self._metadata_pipeline = metadata_pipeline

        self._allowed_exts = {e.lower() for e in (allowed_exts or [])}
        self._file_patterns = list(file_patterns or ["**/*"])

        self._follow_symlinks = follow_symlinks
        self._max_files = max_files

    # ---------------------------------------------------------
    # Load single file
    # ---------------------------------------------------------

    def load_document(self, source: str) -> List[Document]:
        """
        Load a single source (path/http/s3/etc.) using handler registry + metadata pipeline.

        NOTE:
        - DocumentsLoader does NOT validate source correctness.
        - Handler is responsible for interpreting and validating the source.
        """

        docs: List[Document] = []

        try:
            handler = self._registry.resolve(source)

            loaded = handler.load(source)
            if not loaded:
                return docs
            
            normalized = self._normalizer_pipeline.normalize(
                loaded,
                source,
            )

            enriched = self._metadata_pipeline.enrich(normalized, source)
            docs.extend(enriched)
            return docs

        except Exception as e:
            logger.exception(
                "[intergraxDocumentsLoader] Error while loading source %s: %s",
                source,
                e,
            )
            return docs

    # ---------------------------------------------------------
    # Load directory
    # ---------------------------------------------------------

    def load_documents(
        self,
        directory_path: str,
    ) -> List[Document]:

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[intergraxDocumentsLoader] Loading documents from %s",
                directory_path,
            )

        docs: List[Document] = []

        root = Path(directory_path)

        if not root.exists():
            logger.error(
                "[intergraxDocumentsLoader] Directory not found: %s",
                root,
            )
            return docs

        candidate_files: List[Path] = []

        for pattern in self._file_patterns:

            for f in root.glob(pattern):

                try:

                    if not self._follow_symlinks and f.is_symlink():
                        continue

                    if not f.is_file():
                        continue

                    if self._allowed_exts:
                        ext = f.suffix.lower()
                        if ext not in self._allowed_exts:
                            continue

                    candidate_files.append(f)

                except OSError:
                    continue

        if self._max_files is not None and len(candidate_files) > self._max_files:

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[intergraxDocumentsLoader] Too many files (%d). Truncating to %d.",
                    len(candidate_files),
                    self._max_files,
                )

            candidate_files = candidate_files[: self._max_files]

        with tqdm(
            desc=f"Loading files from {directory_path}",
            unit="file",
            leave=False,
            total=len(candidate_files),
            disable=not logger.isEnabledFor(logging.DEBUG),
        ) as pbar:

            for file in candidate_files:

                try:

                    file_docs = self.load_document(str(file))

                    if file_docs:
                        docs.extend(file_docs)

                except Exception as e:

                    logger.exception(
                        "[intergraxDocumentsLoader] Error while loading file %s: %s",
                        file,
                        e,
                    )

                finally:

                    pbar.update(1)

        logger.debug(
            "[intergraxDocumentsLoader] Done. Loaded documents: %d",
            len(docs),
        )

        return docs