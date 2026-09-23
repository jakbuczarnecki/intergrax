# © Artur Czarnecki. All rights reserved.

"""Invocation scope id ↔ authority scope compatibility (UCA-6C-R6-R5-H2)."""

from __future__ import annotations

from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)

# Canonical MSE operation_id for orchestration tool invocation enforcement
# (see ``TOOL_INVOCATION_INNER_ACTION_PREFIX`` in runtime inner governance).
CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID = (
    "orchestration.tool_invocation_authorization"
)


class UnknownInvocationScopeError(ValueError):
    """Invocation scope id is not a recognized authority scope identifier."""


def is_meaningful_side_effect_invocation_scope_id(invocation_scope_id: str) -> bool:
    normalized = invocation_scope_id.strip()
    if not normalized:
        return False
    if normalized.startswith("agr_") or normalized.startswith("dhr_"):
        return False
    if normalized == CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID:
        return True
    return normalized.startswith(
        f"{CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID}:",
    )


def recognize_authority_scope_from_invocation(
    invocation_scope_id: str,
) -> SuspendedOperationAuthorityScope:
    normalized = invocation_scope_id.strip()
    if not normalized:
        raise UnknownInvocationScopeError("invocation scope must be non-empty")
    if normalized.startswith("agr_"):
        return SuspendedOperationAuthorityScope.AGENT_RUNTIME_GOVERNANCE
    if normalized.startswith("dhr_"):
        return SuspendedOperationAuthorityScope.DECLARATIVE_GOVERNANCE
    if is_meaningful_side_effect_invocation_scope_id(normalized):
        return SuspendedOperationAuthorityScope.MEANINGFUL_SIDE_EFFECT
    raise UnknownInvocationScopeError(
        f"unrecognized invocation scope id: {invocation_scope_id!r}",
    )


def infer_authority_scope_from_invocation(
    invocation_scope_id: str,
) -> SuspendedOperationAuthorityScope:
    """Fail-closed scope inference — unknown scopes never map to MSE implicitly."""
    return recognize_authority_scope_from_invocation(invocation_scope_id)


def invocation_scope_matches_authority_scope(
    invocation_scope_id: str,
    authority_scope: SuspendedOperationAuthorityScope,
) -> bool:
    try:
        recognized = recognize_authority_scope_from_invocation(invocation_scope_id)
    except UnknownInvocationScopeError:
        return False
    return recognized is authority_scope


__all__ = [
    "CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID",
    "UnknownInvocationScopeError",
    "infer_authority_scope_from_invocation",
    "invocation_scope_matches_authority_scope",
    "is_meaningful_side_effect_invocation_scope_id",
    "recognize_authority_scope_from_invocation",
]
