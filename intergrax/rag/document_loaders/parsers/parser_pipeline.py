# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.base_document_parser import (
    BaseDocumentParser,
)


class ParserPipeline:
    """
    Deterministic pipeline of document parsers.

    Parsers are executed sequentially until one successfully
    produces documents.

    Order defines priority.
    """

    def __init__(self, parsers: List[BaseDocumentParser]) -> None:
        if not parsers:
            raise ValueError("ParserPipeline requires at least one parser.")

        self._parsers = parsers

    def parse(self, source: str) -> Sequence[Document]:
        last_error: Exception | None = None

        for parser in self._parsers:
            if not parser.is_available():
                continue

            try:
                docs = parser.load(source)

                if docs:
                    return docs

            except Exception as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "No available document parser could process the source."
        )