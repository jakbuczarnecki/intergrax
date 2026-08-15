# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
Optional contextual chunk enrichment at ingest (Anthropic-style situate).

Uses an injected ``LLMAdapter`` when configured — never hardcodes a provider.
"""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


class ContextualChunkEnricher:
    """Prepend short document-level context to each chunk before embedding."""

    def __init__(
        self,
        llm: Optional[LLMAdapter] = None,
        *,
        max_context_tokens: int = 120,
    ) -> None:
        self._llm = llm
        self._max_context_tokens = max_context_tokens

    def enrich(
        self,
        documents: Sequence[KnowledgeDocument],
        chunks: Sequence[KnowledgeDocument],
    ) -> list[KnowledgeDocument]:
        if self._llm is None or not chunks:
            return list(chunks)

        source_text = "\n\n".join(
            document.content.strip()
            for document in documents
            if document.content.strip()
        )[:12000]
        if not source_text:
            return list(chunks)

        enriched: list[KnowledgeDocument] = []
        for chunk in chunks:
            body = chunk.content
            if not body.strip():
                enriched.append(chunk)
                continue
            context_line = self._situate(source_text, body)
            if not context_line:
                enriched.append(chunk)
                continue

            payload = chunk.model_dump(mode="json")
            payload["content"] = f"{context_line}\n\n{body}"
            meta = dict(payload["metadata"])
            meta["contextual_enrich"] = True
            payload["metadata"] = meta
            enriched.append(KnowledgeDocument.model_validate(payload))
        return enriched

    def _situate(self, document_excerpt: str, chunk_text: str) -> str:
        prompt = (
            "Give a short situating context (1-2 sentences) for the chunk below "
            "relative to the document excerpt. Output only the context, no preamble.\n\n"
            f"<document>\n{document_excerpt[:8000]}\n</document>\n\n"
            f"<chunk>\n{chunk_text[:2000]}\n</chunk>"
        )
        try:
            response = self._llm.generate_messages(
                [ChatMessage(role="user", content=prompt)],
                run_id="rag-contextual-enrich",
            )
            text = (response.content or "").strip()
            if len(text) > self._max_context_tokens * 4:
                text = text[: self._max_context_tokens * 4]
            return text
        except Exception:
            return ""
