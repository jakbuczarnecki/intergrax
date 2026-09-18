# © Artur Czarnecki. All rights reserved.

"""Typed ContextView admission scope vs source-proven candidate scope (MP-5G-C1).

View admission may be narrower than a source-native candidate on dimensions the source
did not prove (for example workspace/user Memory inside a work-item-scoped view).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.context_view import ContextViewCategory, ContextViewScope

__all__ = [
    "ContextViewScopeCompatibilityPolicy",
    "DefaultContextViewScopeCompatibilityPolicy",
    "collaborative_work_source_candidate_scope_compatible",
    "knowledge_source_candidate_scope_compatible",
    "memory_source_candidate_scope_compatible",
    "source_candidate_scope_compatible",
    "ucl_source_candidate_scope_compatible",
]


def _operation_scope_dimensions_compatible(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    request_op = request_scope.operation_scope
    candidate_op = candidate_scope.operation_scope
    if request_op is None:
        return True
    if candidate_op is None:
        return False
    if candidate_op.operation_id != request_op.operation_id:
        return False
    if request_op.resource_scope is not None:
        if candidate_op.resource_scope != request_op.resource_scope:
            return False
    return True


def _exact_collaborative_dimensions_compatible(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    if request_scope.tenant_id != candidate_scope.tenant_id:
        return False
    if request_scope.workspace_id != candidate_scope.workspace_id:
        return False
    if request_scope.work_item_id is not None:
        if candidate_scope.work_item_id != request_scope.work_item_id:
            return False
    if not _operation_scope_dimensions_compatible(request_scope, candidate_scope):
        return False
    return True


def _tenant_workspace_aligned(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    if request_scope.tenant_id != candidate_scope.tenant_id:
        return False
    if request_scope.workspace_id != candidate_scope.workspace_id:
        return False
    return True


def _work_item_claim_compatible(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
    *,
    allow_omitted_unproven: bool,
) -> bool:
    candidate_work_item = candidate_scope.work_item_id
    if candidate_work_item is None:
        return allow_omitted_unproven or request_scope.work_item_id is None
    if request_scope.work_item_id is None:
        return False
    return candidate_work_item == request_scope.work_item_id


def _optional_operation_claim_compatible(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    candidate_op = candidate_scope.operation_scope
    if candidate_op is None:
        return True
    return _operation_scope_dimensions_compatible(request_scope, candidate_scope)


def memory_source_candidate_scope_compatible(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    """Memory may omit work-item and operation dimensions when not source-proven."""
    if not _tenant_workspace_aligned(request_scope, candidate_scope):
        return False
    if not _work_item_claim_compatible(
        request_scope,
        candidate_scope,
        allow_omitted_unproven=True,
    ):
        return False
    return _optional_operation_claim_compatible(request_scope, candidate_scope)


def knowledge_source_candidate_scope_compatible(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    """Knowledge may omit work-item when document scope is authoritative."""
    if not _tenant_workspace_aligned(request_scope, candidate_scope):
        return False
    if not _work_item_claim_compatible(
        request_scope,
        candidate_scope,
        allow_omitted_unproven=True,
    ):
        return False
    return _optional_operation_claim_compatible(request_scope, candidate_scope)


def ucl_source_candidate_scope_compatible(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    """UCL may omit work-item when context scope is authoritative."""
    if not _tenant_workspace_aligned(request_scope, candidate_scope):
        return False
    if not _work_item_claim_compatible(
        request_scope,
        candidate_scope,
        allow_omitted_unproven=True,
    ):
        return False
    return _optional_operation_claim_compatible(request_scope, candidate_scope)


def collaborative_work_source_candidate_scope_compatible(
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    return _exact_collaborative_dimensions_compatible(request_scope, candidate_scope)


def source_candidate_scope_compatible(
    *,
    category: ContextViewCategory,
    request_scope: ContextViewScope,
    candidate_scope: ContextViewScope,
) -> bool:
    if category is ContextViewCategory.MEMORY:
        return memory_source_candidate_scope_compatible(request_scope, candidate_scope)
    if category is ContextViewCategory.KNOWLEDGE:
        return knowledge_source_candidate_scope_compatible(request_scope, candidate_scope)
    if category is ContextViewCategory.UCL_CONTEXT_LIFECYCLE:
        return ucl_source_candidate_scope_compatible(request_scope, candidate_scope)
    if category is ContextViewCategory.COLLABORATIVE_WORK:
        return collaborative_work_source_candidate_scope_compatible(
            request_scope,
            candidate_scope,
        )
    return _exact_collaborative_dimensions_compatible(request_scope, candidate_scope)


@runtime_checkable
class ContextViewScopeCompatibilityPolicy(Protocol):
    """Replaceable admission policy for source candidate scope vs view request scope."""

    def candidate_scope_compatible(
        self,
        *,
        category: ContextViewCategory,
        request_scope: ContextViewScope,
        candidate_scope: ContextViewScope,
    ) -> bool:
        """Return True when candidate may be admitted without scope fabrication."""


class DefaultContextViewScopeCompatibilityPolicy:
    """Platform default: exact dimensions except category-specific broader Memory rules."""

    def candidate_scope_compatible(
        self,
        *,
        category: ContextViewCategory,
        request_scope: ContextViewScope,
        candidate_scope: ContextViewScope,
    ) -> bool:
        return source_candidate_scope_compatible(
            category=category,
            request_scope=request_scope,
            candidate_scope=candidate_scope,
        )
