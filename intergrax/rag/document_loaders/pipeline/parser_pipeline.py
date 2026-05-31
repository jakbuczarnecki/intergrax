# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import logging
import time
from typing import Any, List, Sequence

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser

logger = logging.getLogger(__name__)

TRACE_METADATA_KEY = "integration_parser_trace"


class ParserPipeline:
    """
    Deterministic pipeline of document parsers.

    Parsers are executed sequentially until one successfully produces documents.
    Emits structured trace metadata on returned documents for ingestion observability.
    """

    def __init__(self, parsers: List[BaseDocumentParser]) -> None:
        if not parsers:
            raise ValueError("ParserPipeline requires at least one parser.")
        self._parsers = parsers

    def parse(self, source: str) -> Sequence[Document]:
        last_error: Exception | None = None
        attempts: list[dict[str, Any]] = []

        for parser in self._parsers:
            parser_id = parser.parser_id()
            if not parser.is_available():
                attempts.append({"parser_id": parser_id, "status": "skipped_unavailable"})
                continue

            started = time.perf_counter()
            try:
                docs = parser.load(source)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

                if docs:
                    attempts.append(
                        {
                            "parser_id": parser_id,
                            "status": "success",
                            "latency_ms": elapsed_ms,
                            "num_documents": len(docs),
                        }
                    )
                    return self._attach_trace(source, docs, attempts)

                attempts.append(
                    {"parser_id": parser_id, "status": "empty", "latency_ms": elapsed_ms}
                )

            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
                last_error = exc
                attempts.append(
                    {
                        "parser_id": parser_id,
                        "status": "error",
                        "latency_ms": elapsed_ms,
                        "error": str(exc),
                    }
                )
                logger.debug(
                    "parser_pipeline fallback parser_id=%s error=%s",
                    parser_id,
                    exc,
                )
                continue

        if last_error is not None:
            raise last_error

        raise RuntimeError("No available document parser could process the source.")

    @staticmethod
    def _attach_trace(
        source: str,
        docs: Sequence[Document],
        attempts: list[dict[str, Any]],
    ) -> Sequence[Document]:
        winning = attempts[-1] if attempts else {}
        trace = {
            "attempts": attempts,
            "parser_id": winning.get("parser_id"),
            "latency_ms": winning.get("latency_ms"),
        }
        from intergrax.rag.document_loaders.observability.parser_trace_exporter import export_parser_trace

        export_parser_trace(source=source, trace=trace)
        enriched: list[Document] = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            metadata[TRACE_METADATA_KEY] = trace
            metadata.setdefault("integration_parser_id", trace.get("parser_id"))
            enriched.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )
        return enriched
