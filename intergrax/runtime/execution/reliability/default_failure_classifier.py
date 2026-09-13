# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default provider-neutral execution failure classifier (EE-B1.1)."""

from __future__ import annotations

from intergrax.contracts.execution_reliability.failure_classification_contract import (
    ExecutionFailureContext,
    ExecutionFailureDecision,
    ExecutionFailureSemanticCategory,
)
from intergrax.contracts.execution_retry import ExecutionFailureKind
from intergrax.contracts.resilience_policy import FailureClass
from intergrax.runtime.execution.retry.classification import classify_execution_failure

__all__ = ["DefaultExecutionFailureClassifier", "default_execution_failure_classifier"]


def _semantic_category(context: ExecutionFailureContext) -> ExecutionFailureSemanticCategory:
    if context.policy_blocked:
        return ExecutionFailureSemanticCategory.POLICY_BLOCKED
    if context.resource_exhausted:
        return ExecutionFailureSemanticCategory.RESOURCE_EXHAUSTED
    if context.dependency_unavailable:
        return ExecutionFailureSemanticCategory.DEPENDENCY_FAILURE
    if context.timeout:
        return ExecutionFailureSemanticCategory.TRANSIENT
    if context.failure_class is FailureClass.POLICY_ERROR:
        return ExecutionFailureSemanticCategory.POLICY_BLOCKED
    if context.failure_class is FailureClass.USER_ERROR:
        return ExecutionFailureSemanticCategory.PERMANENT
    if context.failure_class is FailureClass.DEPENDENCY_ERROR:
        return ExecutionFailureSemanticCategory.DEPENDENCY_FAILURE
    if context.failure_class in {
        FailureClass.RUNTIME_ERROR,
        FailureClass.QUALITY_ERROR,
    }:
        return ExecutionFailureSemanticCategory.TRANSIENT
    return ExecutionFailureSemanticCategory.UNKNOWN


def _retry_kind_for_category(
    category: ExecutionFailureSemanticCategory,
    *,
    context: ExecutionFailureContext,
) -> ExecutionFailureKind:
    if category is ExecutionFailureSemanticCategory.TRANSIENT:
        return ExecutionFailureKind.RETRYABLE_TRANSIENT
    if category is ExecutionFailureSemanticCategory.POLICY_BLOCKED:
        return ExecutionFailureKind.GOVERNANCE_DENIED
    if category is ExecutionFailureSemanticCategory.RESOURCE_EXHAUSTED:
        return ExecutionFailureKind.BUDGET_EXHAUSTED
    if category is ExecutionFailureSemanticCategory.PERMANENT:
        return ExecutionFailureKind.NON_RETRYABLE_PERMANENT
    if category is ExecutionFailureSemanticCategory.DEPENDENCY_FAILURE:
        return ExecutionFailureKind.RETRYABLE_TRANSIENT
    if context.has_unknown_side_effect:
        return ExecutionFailureKind.UNKNOWN_UNSAFE
    return ExecutionFailureKind.UNKNOWN


class DefaultExecutionFailureClassifier:
    """Maps normalized failure context to semantic category + retry projection."""

    def classify(self, failure_context: ExecutionFailureContext) -> ExecutionFailureDecision:
        category = _semantic_category(failure_context)
        kind = _retry_kind_for_category(category, context=failure_context)
        projection = classify_execution_failure(
            kind=kind,
            reason=failure_context.reason,
            failure_class=failure_context.failure_class,
            has_unknown_side_effect=failure_context.has_unknown_side_effect,
        )
        return ExecutionFailureDecision(
            category=category,
            reason=failure_context.reason,
            retry_projection=projection,
        )


_DEFAULT = DefaultExecutionFailureClassifier()


def default_execution_failure_classifier() -> DefaultExecutionFailureClassifier:
    return _DEFAULT
