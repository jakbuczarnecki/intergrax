# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument

from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey


class ContextBuilder:
    """
    Builds LLM context from retrieved documents.

    Converts a list of retrieved documents into a formatted
    context string passed to the prompt builder.

    Context size is controlled using token limits instead
    of character limits.
    """

    def __init__(
        self,
        *,
        tokenizer,
        max_tokens: int = 4000,
    ) -> None:

        self._tokenizer = tokenizer
        self._max_tokens = int(max_tokens)

    def build(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> str:

        context_parts: list[str] = []

        total_tokens = 0

        for doc in documents:

            text = doc.content.strip()

            if not text:
                continue

            metadata = doc.metadata

            source = metadata.get(
                DocumentMetadataKey.SOURCE_NAME,
                doc.provenance.source_id,
            )

            block = f"[Source: {source}]\n{text}"

            token_count = self._tokenizer.count_tokens(block)

            if total_tokens + token_count > self._max_tokens:
                break

            context_parts.append(block)

            total_tokens += token_count

        return "\n\n".join(context_parts)