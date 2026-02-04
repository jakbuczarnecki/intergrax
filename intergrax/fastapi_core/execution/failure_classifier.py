# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.


from intergrax.fastapi_core.execution.failures import FailureCategory
from typing import Dict, Type
from concurrent.futures import CancelledError


class FailureClassifier:
    """
    Registry-based exception classifier.
    """

    def __init__(
        self,
        mapping: Dict[Type[BaseException], FailureCategory] | None = None,
    ) -> None:
        self._mapping = mapping or {
            CancelledError: FailureCategory.CANCELED,
            TimeoutError: FailureCategory.TIMEOUT,
            ConnectionError: FailureCategory.RETRYABLE,
        }

    def classify(self, exc: BaseException) -> FailureCategory:
        for exc_type, category in self._mapping.items():
            if isinstance(exc, exc_type):
                return category
        return FailureCategory.TERMINAL
