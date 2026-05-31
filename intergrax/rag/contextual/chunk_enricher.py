# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
Optional contextual chunk enrichment at ingest (Anthropic-style situate).

Uses an injected ``LLMAdapter`` when configured — never hardcodes a provider.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from langchain_core.documents import Document

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
        documents: Sequence[Document],
        chunks: Sequence[Document],
    ) -> List[Document]:
        if self._llm is None or not chunks:
            return list(chunks)

        source_text = "\n\n".join(
            (d.page_content or "").strip() for d in documents if (d.page_content or "").strip()
        )[:12000]
        if not source_text:
            return list(chunks)

        enriched: List[Document] = []
        for chunk in chunks:
            body = (chunk.page_content or "").strip()
            if not body:
                enriched.append(chunk)
                continue
            context_line = self._situate(source_text, body)
            if context_line:
                combined = f"{context_line}\n\n{body}"
            else:
                combined = body
            meta = dict(chunk.metadata or {})
            meta["contextual_enrich"] = True
            enriched.append(Document(page_content=combined, metadata=meta))
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
