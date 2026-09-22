# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import logging
import time
from typing import Any, List, Sequence

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser

logger = logging.getLogger(__name__)

TRACE_METADATA_KEY = "integration_parser_trace"


class ParserPipeline:
    """
    Deterministic pipeline of document parsers.

    Parsers are executed sequentially until one successfully produces fragments.
    Emits structured trace metadata on returned fragments for ingestion observability.
    """

    def __init__(self, parsers: List[BaseDocumentParser]) -> None:
        if not parsers:
            raise ValueError("ParserPipeline requires at least one parser.")
        self._parsers = parsers

    def parse(self, source: str) -> Sequence[ParsedDocumentFragment]:
        last_error: Exception | None = None
        attempts: list[dict[str, Any]] = []

        for parser in self._parsers:
            parser_id = parser.parser_id()
            if not parser.is_available():
                attempts.append({"parser_id": parser_id, "status": "skipped_unavailable"})
                continue

            started = time.perf_counter()
            try:
                fragments = parser.load(source)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

                if fragments:
                    attempts.append(
                        {
                            "parser_id": parser_id,
                            "status": "success",
                            "latency_ms": elapsed_ms,
                            "num_documents": len(fragments),
                        }
                    )
                    return self._attach_trace(source, fragments, attempts)

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
        fragments: Sequence[ParsedDocumentFragment],
        attempts: list[dict[str, Any]],
    ) -> Sequence[ParsedDocumentFragment]:
        winning = attempts[-1] if attempts else {}
        from intergrax.rag.document_loaders.observability.parser_trace_contract import (
            DocumentParserTrace,
            ParserAttemptStatus,
            ParserTraceAttempt,
        )
        from intergrax.rag.document_loaders.observability.parser_trace_exporter import export_parser_trace

        typed_attempts: list[ParserTraceAttempt] = []
        for attempt in attempts:
            status_raw = attempt.get("status")
            if not isinstance(status_raw, str):
                continue
            try:
                status = ParserAttemptStatus(status_raw)
            except ValueError:
                continue
            parser_id_raw = attempt.get("parser_id")
            if not isinstance(parser_id_raw, str):
                continue
            latency = attempt.get("latency_ms")
            latency_ms = float(latency) if isinstance(latency, (int, float)) and not isinstance(latency, bool) else None
            num_docs = attempt.get("num_documents")
            num_documents = num_docs if isinstance(num_docs, int) and not isinstance(num_docs, bool) else None
            err = attempt.get("error")
            typed_attempts.append(
                ParserTraceAttempt(
                    parser_id=parser_id_raw,
                    status=status,
                    latency_ms=latency_ms,
                    num_documents=num_documents,
                    error=str(err) if isinstance(err, str) else None,
                )
            )
        winning_parser_id = winning.get("parser_id")
        winning_latency = winning.get("latency_ms")
        trace = DocumentParserTrace(
            parser_id=winning_parser_id if isinstance(winning_parser_id, str) else None,
            attempts=tuple(typed_attempts),
            latency_ms=(
                float(winning_latency)
                if isinstance(winning_latency, (int, float)) and not isinstance(winning_latency, bool)
                else None
            ),
        )
        export_parser_trace(source=source, trace=trace)
        trace_metadata = trace.to_logging_extra_value()
        enriched: list[ParsedDocumentFragment] = []
        for fragment in fragments:
            metadata = dict(fragment.metadata or {})
            metadata[TRACE_METADATA_KEY] = trace_metadata
            metadata.setdefault("integration_parser_id", trace.parser_id)
            enriched.append(
                ParsedDocumentFragment(
                    text=fragment.text,
                    metadata=metadata,
                    native_handle=fragment.native_handle,
                )
            )
        return enriched
