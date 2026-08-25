# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from intergrax.runtime.architecture.data_classification_enforcement import (
    DataClassificationPolicyError,
)
from intergrax.runtime.nexus.budget.budget_enforcer import BudgetExceededError
from intergrax.runtime.nexus.budget.production_budget_policy import (
    ProductionBudgetPolicyError,
)
from intergrax.runtime.background_execution.required_audit_evidence import (
    RequiredAuditEvidencePersistenceError,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode


class ErrorClassifier:
    @staticmethod
    def classify(exc: Exception) -> RuntimeErrorCode:
        if isinstance(exc, RequiredAuditEvidencePersistenceError):
            return RuntimeErrorCode.DEPENDENCY_ERROR
        if isinstance(exc, BudgetExceededError):
            return RuntimeErrorCode.POLICY_ERROR
        if isinstance(exc, (PermissionError, DataClassificationPolicyError)):
            return RuntimeErrorCode.PERMISSION_ERROR
        if isinstance(exc, ProductionBudgetPolicyError):
            return RuntimeErrorCode.POLICY_ERROR
        if isinstance(exc, TimeoutError):
            return RuntimeErrorCode.TIMEOUT
        if isinstance(exc, (ConnectionError, OSError)):
            return RuntimeErrorCode.DEPENDENCY_ERROR
        if isinstance(exc, RuntimeError):
            return RuntimeErrorCode.RUNTIME_ERROR
        if isinstance(exc, ValueError):
            return RuntimeErrorCode.VALIDATION_ERROR
        message = str(exc).lower()
        if "policy" in message or "denied" in message:
            return RuntimeErrorCode.POLICY_ERROR
        if "quality" in message or "validation failed" in message:
            return RuntimeErrorCode.QUALITY_ERROR
        if "user" in message:
            return RuntimeErrorCode.USER_ERROR
        if "llm" in message or "model" in message:
            return RuntimeErrorCode.LLM_ERROR
        if "tool" in message:
            return RuntimeErrorCode.TOOL_ERROR
        return RuntimeErrorCode.INTERNAL_ERROR
