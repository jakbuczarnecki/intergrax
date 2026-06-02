# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod

from .answer_request import AnswerRequest
from .answer_result import AnswerResult


class AnswerEngine(ABC):
    """
    Execution contract for RAG + LLM answering.
    """

    @abstractmethod
    def answer(
        self,
        *,
        request: AnswerRequest,
    ) -> AnswerResult:
        """
        Execute full RAG answering pipeline.
        """
        raise NotImplementedError