# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.retrievers.registry.retriever_registry import DEFAULT_RETRIEVER_ID
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter


@dataclass(slots=True)
class AnswerRequest:
    """
    Request object used by AnswerEngine.
    """

    query: str

    top_k: int = 5

    metadata_filter: MetadataFilter | None = None

    retriever_id: str = DEFAULT_RETRIEVER_ID

    include_embeddings: bool = False

    llm: Optional[LLMAdapter] = None