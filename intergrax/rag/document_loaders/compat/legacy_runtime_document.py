# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any

from pydantic import PrivateAttr

from intergrax.compat.langchain import to_langchain_document
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey


class _KnowledgeDocumentWithParserRuntime(KnowledgeDocument):
    _parser_native_handle: Any | None = PrivateAttr(default=None)


def _as_runtime_document(document: KnowledgeDocument) -> _KnowledgeDocumentWithParserRuntime:
    if isinstance(document, _KnowledgeDocumentWithParserRuntime):
        return document
    return _KnowledgeDocumentWithParserRuntime.model_validate(
        document.model_dump(mode="python")
    )


def _get_parser_native_handle(document: KnowledgeDocument) -> object | None:
    if isinstance(document, _KnowledgeDocumentWithParserRuntime):
        return document._parser_native_handle
    return None


def attach_parser_native_handle(document: KnowledgeDocument, handle: object) -> KnowledgeDocument:
    runtime_doc = _as_runtime_document(document)
    runtime_doc._parser_native_handle = handle
    return runtime_doc


def copy_parser_runtime_state(source: KnowledgeDocument, target: KnowledgeDocument) -> KnowledgeDocument:
    handle = _get_parser_native_handle(source)
    if handle is None:
        return target
    return attach_parser_native_handle(target, handle)


def to_legacy_rag_document(document: KnowledgeDocument) -> object:
    langchain_doc = to_langchain_document(document)
    metadata = dict(langchain_doc.metadata or {})
    metadata[DocumentMetadataKey.DOCUMENT_ID.value] = document.identity.document_id
    handle = _get_parser_native_handle(document)
    if handle is not None:
        metadata[DocumentMetadataKey.DOCLING_DOCUMENT_META.value] = handle
    document_cls = type(langchain_doc)
    return document_cls(
        id=langchain_doc.id,
        page_content=langchain_doc.page_content,
        metadata=metadata,
    )
