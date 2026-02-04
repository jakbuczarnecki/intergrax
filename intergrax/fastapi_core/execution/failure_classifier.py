# FILE: intergrax/fastapi_core/execution/failure_classifier.py

from typing import Type
from intergrax.fastapi_core.execution.failures import FailureCategory


class FailureClassifier:
    """
    Maps exceptions to runtime failure categories.
    """

    def classify(self, exc: BaseException) -> FailureCategory:
        from concurrent.futures import CancelledError

        if isinstance(exc, CancelledError):
            return FailureCategory.CANCELED

        if isinstance(exc, TimeoutError):
            return FailureCategory.TIMEOUT

        # retryable examples (network, transient)
        if isinstance(exc, ConnectionError):
            return FailureCategory.RETRYABLE

        return FailureCategory.TERMINAL
