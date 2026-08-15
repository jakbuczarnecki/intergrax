# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import PrivateAttr

from intergrax.compat.langchain import to_langchain_document
from intergrax.compat.langchain.documents import make_langchain_document
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.utils import attribute_access


class _KnowledgeDocumentWithParserRuntime(KnowledgeDocument):
    _parser_native_handle: Any | None = PrivateAttr(default=None)


def _as_runtime_document(document: KnowledgeDocument) -> _KnowledgeDocumentWithParserRuntime:
    if isinstance(document, _KnowledgeDocumentWithParserRuntime):
        return document
    return _KnowledgeDocumentWithParserRuntime.model_validate(
        document.model_dump(mode="python")
    )


def get_parser_native_handle(document: KnowledgeDocument) -> object | None:
    if isinstance(document, _KnowledgeDocumentWithParserRuntime):
        return document._parser_native_handle
    return None


def attach_parser_native_handle(document: KnowledgeDocument, handle: object) -> KnowledgeDocument:
    runtime_doc = _as_runtime_document(document)
    runtime_doc._parser_native_handle = handle
    return runtime_doc


def copy_parser_runtime_state(source: KnowledgeDocument, target: KnowledgeDocument) -> KnowledgeDocument:
    handle = get_parser_native_handle(source)
    if handle is None:
        return target
    return attach_parser_native_handle(target, handle)


def to_legacy_rag_document(document: KnowledgeDocument) -> object:
    langchain_doc = to_langchain_document(document)
    metadata = dict(langchain_doc.metadata or {})
    metadata[DocumentMetadataKey.DOCUMENT_ID.value] = document.identity.document_id
    document_cls = type(langchain_doc)
    return document_cls(
        id=langchain_doc.id,
        page_content=langchain_doc.page_content,
        metadata=metadata,
    )


def from_legacy_rag_hit(hit: object) -> KnowledgeDocument:
    """Reconstruct a native document from a legacy provider hit."""
    try:
        metadata = attribute_access.optional(hit, "metadata")
        content = attribute_access.optional(hit, "content")
        document_id = attribute_access.optional(hit, "id", None)
    except AttributeError as exc:
        raise ValueError("legacy vector-store hit is malformed") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("legacy vector-store hit metadata must be a mapping")
    if not isinstance(content, str):
        raise ValueError("legacy vector-store hit content must be a string")
    if document_id is not None and not isinstance(document_id, str):
        raise ValueError("legacy vector-store hit id must be a string")
    metadata_copy = dict(metadata)
    metadata_document_id = metadata_copy.get("document_id")
    if metadata_document_id is not None and not isinstance(metadata_document_id, str):
        raise ValueError("legacy vector-store hit document_id must be a string")
    resolved_document_id = metadata_document_id or document_id
    if resolved_document_id is None:
        raise ValueError("legacy vector-store hit has no document identity")
    metadata_copy.pop("document_id", None)
    metadata_copy.pop("text", None)

    from intergrax.compat.langchain.documents import from_langchain_document

    legacy_document = make_langchain_document(
        document_id=resolved_document_id,
        content=content,
        metadata=metadata_copy,
    )
    return from_langchain_document(legacy_document)
