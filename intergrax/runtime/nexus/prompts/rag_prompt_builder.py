# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
if TYPE_CHECKING:
    from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_builder import (
    RetrievedChunk,
    BuiltContext,
)


@dataclass
class RagPromptBundle:
    """
    Prompt elements related to RAG:

    - context_messages: system-level messages injecting retrieved document context.
    """
    context_messages: List[ChatMessage]


class RagPromptBuilder(Protocol):
    """
    Strategy interface for building the RAG-related part of the prompt.

    You can provide a custom implementation and pass it to
    RuntimeEngine to fully control:

    - the exact system prompt text,
    - how retrieved chunks are formatted and injected as messages.
    """

    def build_rag_prompt(self, built: BuiltContext) -> RagPromptBundle:
        ...


class DefaultRagPromptBuilder(RagPromptBuilder):
    """
    Default prompt builder for nexus Mode.

    Responsibilities:
    - Inject retrieved chunks into system-level context messages.
    - Global system instructions are owned by the runtime.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config

    def build_rag_prompt(self, built: BuiltContext) -> RagPromptBundle:
        context_messages: List[ChatMessage] = []

        if built.retrieved_chunks:
            rag_context_text = self._format_rag_context(built.retrieved_chunks)
            context_messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "The following excerpts were retrieved from the user's "
                        "documents. Use them as factual context when answering "
                        "the user's question.\n\n"
                        f"{rag_context_text}"
                    ),
                )
            )

        return RagPromptBundle(
            context_messages=context_messages,
        )


    def _format_rag_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Build a compact, model-friendly text block from retrieved chunks.

        Design goals:
        - Provide enough semantic context.
        - Avoid internal markers ([CTX ...], scores, ids) that the model
          could copy into the final answer.
        - Keep format simple and natural.
        """
        if not chunks:
            return ""

        lines: List[str] = []

        for ch in chunks:
            source_name = (
                ch.metadata.get("source_name")
                or ch.metadata.get("attachment_id")
                or "document"
            )
            lines.append(f"Source: {source_name}")
            lines.append("Excerpt:")
            lines.append(ch.text)
            lines.append("")  # blank line separator

        # Optional: add truncation based on config (e.g. max chars)
        # For now we keep full text and rely on upstream chunking.
        return "\n".join(lines)