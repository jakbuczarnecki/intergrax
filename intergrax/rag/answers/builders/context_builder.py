# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import List, Optional

from langchain_core.documents import Document

from intergrax.rag.answers.contracts.base_context_builder import BaseContextBuilder
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.tokenizers.contracts.base_tokenizer_manager import BaseTokenizerManager


class DefaultContextBuilder(BaseContextBuilder):
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
        tokenizer_manager: BaseTokenizerManager,        
    ) -> None:

        self._tokenizer_manager = tokenizer_manager        

    def build(
        self,
        documents: List[Document],
        tokenizer_id: Optional[str] = None,
        max_tokens: int = 4000,
    ) -> str:

        context_parts: List[str] = []

        total_tokens = 0

        for doc in documents:

            text = (doc.page_content or "").strip()

            if not text:
                continue

            metadata = doc.metadata or {}

            source = metadata.get(
                DocumentMetadataKey.SOURCE_NAME,
                "unknown",
            )

            block = f"[Source: {source}]\n{text}"

            token_count = self._tokenizer_manager.count_tokens(block, tokenizer_id=tokenizer_id)

            if total_tokens + token_count > max_tokens:
                break

            context_parts.append(block)

            total_tokens += token_count

        return "\n\n".join(context_parts)