# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.answers.engine.answer_engine import AnswerEngine
from intergrax.rag.answers.contracts.answer_request import AnswerRequest
from intergrax.rag.answers.contracts.answer_result import AnswerResult


class AnswerManager:
    """
    Public entry point for the RAG answering subsystem.

    The manager delegates answering to the configured AnswerEngine.
    """

    def __init__(self, engine: AnswerEngine) -> None:
        self._engine = engine

    def answer(
        self,
        *,
        request: AnswerRequest,
    ) -> AnswerResult:
        """
        Generate an answer for a given request.
        """

        return self._engine.answer(request=request)