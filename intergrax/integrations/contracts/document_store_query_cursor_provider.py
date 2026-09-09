# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral document-store query cursor capability shared across runtime domains."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.document_store import DocumentQueryCursorCodec


@runtime_checkable
class DocumentStoreQueryCursorProvider(Protocol):
    @property
    def query_cursor_codec(self) -> DocumentQueryCursorCodec:
        """Authenticated codec for document-store query continuation cursors."""


__all__ = ["DocumentStoreQueryCursorProvider"]
