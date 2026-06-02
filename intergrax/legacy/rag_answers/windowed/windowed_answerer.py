# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from intergrax.legacy.rag_answers.answer_manager import AnswerManager
from intergrax.legacy.rag_answers.contracts.answer_request import AnswerRequest
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager


class WindowedAnswerer:
    """
    Windowed RAG answering using map → reduce strategy.

    Uses existing RAG pipeline instead of re-implementing retrieval
    and prompting logic.
    """

    def __init__(
        self,
        *,
        answer_manager: AnswerManager,
        retriever_manager: BaseRetrieverManager,
    ) -> None:

        self._answer_manager = answer_manager
        self._retriever_manager = retriever_manager

    def ask_windowed(
        self,
        *,
        request: AnswerRequest,
        top_k_total: int = 60,
        window_size: int = 12,
    ):

        # -------------------------------------------------
        # 1. Broad retrieval
        # -------------------------------------------------

        candidates = self._retriever_manager.retrieve(
            query_text=request.query,
            retriever_id=request.retriever_id,
            query_embedding=None,
            top_k=top_k_total,
            metadata_filter=request.metadata_filter,
            include_embeddings=request.include_embeddings,
        )

        if not candidates:
            return {
                "answer": "No sufficiently relevant context was found to answer.",
                "sources": [],
                "summary": None,
                "stats": {"windows": 0},
            }

        # -------------------------------------------------
        # 2. Convert candidates → Documents
        # -------------------------------------------------

        docs: List[Document] = [
            Document(
                page_content=c.content,
                metadata=c.metadata,
            )
            for c in candidates
        ]

        # -------------------------------------------------
        # 3. Window splitting
        # -------------------------------------------------

        windows: List[List[Document]] = [
            docs[i:i + window_size]
            for i in range(0, len(docs), window_size)
        ]

        partial_answers: List[str] = []

        # -------------------------------------------------
        # 4. MAP phase
        # -------------------------------------------------

        for window in windows:

            window_request = AnswerRequest(
                query=request.query,
                llm=request.llm,
                top_k=len(window),
                metadata_filter=None,
                retriever_id=request.retriever_id,
            )

            result = self._answer_manager.answer(request=window_request)

            partial_answers.append(result.answer)

        # -------------------------------------------------
        # 5. REDUCE phase
        # -------------------------------------------------

        synthesis_context = "\n\n".join(partial_answers)

        reduce_request = AnswerRequest(
            query=request.query,
            llm=request.llm,
            top_k=1,
        )

        reduce_result = self._answer_manager.answer(request=reduce_request)

        return {
            "answer": reduce_result.answer,
            "sources": [],
            "summary": None,
            "stats": {"windows": len(windows)},
        }