# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import List, Optional, Sequence

from tqdm import tqdm

from pydantic import ValidationError

from intergrax.compat.langchain import from_langchain_document, to_langchain_document
from intergrax.knowledge.contracts import KnowledgeDocument, KnowledgeDocumentScope
from intergrax.knowledge.contracts.validation import JsonValue, knowledge_metadata_to_plain
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    copy_parser_runtime_state,
)
from intergrax.rag.document_loaders.contracts.base_document_loader import (
    BaseDocumentsLoader,
    MetadataCallback,
)
from intergrax.rag.document_loaders.pipeline.metadata_pipeline import MetadataPipeline
from intergrax.rag.document_loaders.pipeline.normalizer_pipeline import NormalizerPipeline
from intergrax.rag.document_loaders.registry.document_handler_registry import DocumentHandlerRegistry

logger = logging.getLogger(__name__)


def _merge_custom_metadata(
    document: KnowledgeDocument,
    extra: Mapping[str, JsonValue],
) -> KnowledgeDocument:
    merged = {
        **knowledge_metadata_to_plain(document.metadata),
        **dict(extra),
    }
    payload = document.model_dump(mode="python")
    payload["metadata"] = merged
    validated = KnowledgeDocument.model_validate(payload)
    return copy_parser_runtime_state(document, validated)


def _roundtrip_knowledge_document(
    original: KnowledgeDocument,
    langchain_doc: object,
) -> KnowledgeDocument:
    if getattr(langchain_doc, "id", None) is None:
        converted = from_langchain_document(
            langchain_doc,
            document_id=original.identity.document_id,
        )
    else:
        converted = from_langchain_document(langchain_doc)

    if converted.identity.document_id != original.identity.document_id:
        raise ValueError(
            "document_id mismatch after normalization: "
            f"expected {original.identity.document_id!r}, "
            f"got {converted.identity.document_id!r}"
        )

    payload = converted.model_dump(mode="python")
    payload["identity"] = original.identity.model_dump()
    payload["scope"] = original.scope.model_dump()
    payload["provenance"] = original.provenance.model_dump()
    validated = KnowledgeDocument.model_validate(payload)
    return copy_parser_runtime_state(original, validated)


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

    def load_document(
        self,
        source: str,
        *,
        tenant_id: str,
        namespace: str | None = None,
        use_default_metadata: bool = True,
        call_custom_metadata: Optional[MetadataCallback] = None,
    ) -> List[KnowledgeDocument]:
        """
        Load a single source (path/http/s3/etc.) using handler registry + metadata pipeline.

        NOTE:
        - DocumentsLoader does NOT validate source correctness.
        - Handler is responsible for interpreting and validating the source.
        """

        scope = KnowledgeDocumentScope(tenant_id=tenant_id, namespace=namespace)

        docs: List[KnowledgeDocument] = []

        try:
            handler = self._registry.resolve(source)

            loaded = handler.load(source, scope=scope)
            if not loaded:
                return docs

            # Temporary LCI-2B bridge around legacy normalization/metadata pipelines.
            # Removed in LCI-2C.
            langchain_docs = [to_langchain_document(doc) for doc in loaded]
            normalized = self._normalizer_pipeline.normalize(
                langchain_docs,
                source,
            )

            if use_default_metadata:
                enriched_seq = self._metadata_pipeline.enrich(normalized, source)
                enriched_langchain = list(enriched_seq)
            else:
                enriched_langchain = list(normalized)

            enriched = [
                _roundtrip_knowledge_document(original, doc)
                for original, doc in zip(loaded, enriched_langchain, strict=True)
            ]

            if call_custom_metadata is not None:
                merged_docs: List[KnowledgeDocument] = []
                for doc in enriched:
                    extra = call_custom_metadata(doc, source)
                    if extra:
                        merged_docs.append(_merge_custom_metadata(doc, extra))
                    else:
                        merged_docs.append(doc)
                enriched = merged_docs

            docs.extend(enriched)
            return docs

        except ValidationError:
            raise
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
        *,
        tenant_id: str,
        namespace: str | None = None,
    ) -> List[KnowledgeDocument]:

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[intergraxDocumentsLoader] Loading documents from %s",
                directory_path,
            )

        docs: List[KnowledgeDocument] = []

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

                    file_docs = self.load_document(
                        str(file),
                        tenant_id=tenant_id,
                        namespace=namespace,
                    )

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
