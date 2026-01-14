# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode


class ErrorClassifier:
    @staticmethod
    def classify(exc: Exception) -> RuntimeErrorCode:
        if isinstance(exc, ValueError):
            return RuntimeErrorCode.VALIDATION_ERROR
        if isinstance(exc, TimeoutError):
            return RuntimeErrorCode.TIMEOUT
        return RuntimeErrorCode.INTERNAL_ERROR
