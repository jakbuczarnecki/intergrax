# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.rag.answers.contracts.answer_request import AnswerRequest
from intergrax.rag.answers.contracts.answer_result import AnswerResult


class BaseAnswerManager(ABC):

    @abstractmethod
    def answer(
        self,
        *,
        request: AnswerRequest,
    ) -> AnswerResult:

        raise NotImplementedError