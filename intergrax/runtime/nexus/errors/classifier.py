# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from intergrax.runtime.nexus.budget.budget_enforcer import BudgetExceededError
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode


class ErrorClassifier:
    @staticmethod
    def classify(exc: Exception) -> RuntimeErrorCode:
        if isinstance(exc, BudgetExceededError):
            return RuntimeErrorCode.POLICY_ERROR
        if isinstance(exc, PermissionError):
            return RuntimeErrorCode.PERMISSION_ERROR
        if isinstance(exc, TimeoutError):
            return RuntimeErrorCode.TIMEOUT
        if isinstance(exc, ConnectionError):
            return RuntimeErrorCode.DEPENDENCY_ERROR
        if isinstance(exc, ValueError):
            return RuntimeErrorCode.VALIDATION_ERROR
        message = str(exc).lower()
        if "policy" in message or "denied" in message:
            return RuntimeErrorCode.POLICY_ERROR
        if "quality" in message or "validation failed" in message:
            return RuntimeErrorCode.QUALITY_ERROR
        if "user" in message:
            return RuntimeErrorCode.USER_ERROR
        return RuntimeErrorCode.INTERNAL_ERROR
