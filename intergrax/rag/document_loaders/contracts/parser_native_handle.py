# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Parser-native handle attachment for KnowledgeDocument (contract-level seam)."""

from __future__ import annotations

from typing import Any

from pydantic import PrivateAttr

from intergrax.knowledge.contracts import KnowledgeDocument


class _KnowledgeDocumentWithParserRuntime(KnowledgeDocument):
    _parser_native_handle: Any | None = PrivateAttr(default=None)


def attach_parser_native_handle(document: KnowledgeDocument, handle: object) -> KnowledgeDocument:
    if isinstance(document, _KnowledgeDocumentWithParserRuntime):
        runtime_doc = document
    else:
        runtime_doc = _KnowledgeDocumentWithParserRuntime.model_validate(
            document.model_dump(mode="python")
        )
    runtime_doc._parser_native_handle = handle
    return runtime_doc


__all__ = ["attach_parser_native_handle"]
