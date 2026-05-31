# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Document parser integration contract (Phase M.7)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


class ParsedDocumentFragment(BaseModel):
    """Normalized output from a document parser backend before RAG mapping."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    native_handle: Any | None = None


@runtime_checkable
class DocumentParser(Protocol):
    """
    Backend-agnostic document parsing facade.

  Implementations live in ``integrations/providers/document_parser/<slug>/``.
  RAG ``document_loaders`` map fragments to LangChain ``Document`` objects.
    """

    def parser_id(self) -> str:
        """Stable id, e.g. ``docling``, ``pymupdf``."""

    def is_available(self) -> bool:
        """True when dependencies and configuration allow parsing."""

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        """Parse a local file path into one or more text fragments."""
