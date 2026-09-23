# © Artur Czarnecki. All rights reserved.

"""Invocation scope id ↔ authority scope compatibility (UCA-6C-R6-R5)."""

from __future__ import annotations

from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)


def infer_authority_scope_from_invocation(
    invocation_scope_id: str,
) -> SuspendedOperationAuthorityScope:
    normalized = invocation_scope_id.strip()
    if normalized.startswith("agr_"):
        return SuspendedOperationAuthorityScope.AGENT_RUNTIME_GOVERNANCE
    if normalized.startswith("dhr_"):
        return SuspendedOperationAuthorityScope.DECLARATIVE_GOVERNANCE
    return SuspendedOperationAuthorityScope.MEANINGFUL_SIDE_EFFECT


def invocation_scope_matches_authority_scope(
    invocation_scope_id: str,
    authority_scope: SuspendedOperationAuthorityScope,
) -> bool:
    return infer_authority_scope_from_invocation(invocation_scope_id) is authority_scope


__all__ = [
    "infer_authority_scope_from_invocation",
    "invocation_scope_matches_authority_scope",
]
