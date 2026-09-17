# © Artur Czarnecki. All rights reserved.

"""Shared Memory scope authority and governance service resolution.

Used by default control-plane and reference-read surfaces without coupling
their implementations.
"""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlPlaneScope,
    MemoryControlScopeRef,
)
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)

__all__ = [
    "assert_memory_scope_authorized",
    "governance_service_or_default",
]


def governance_service_or_default(
    service: MemorySecurityGovernanceService | None,
) -> MemorySecurityGovernanceService:
    return (
        service
        if service is not None
        else build_default_memory_security_governance_service()
    )


def assert_memory_scope_authorized(
    identity: RequestIdentity,
    scope: MemoryControlScopeRef,
) -> None:
    if scope.tenant_id != identity.tenant_id:
        raise MemoryControlAccessDenied(
            "scope tenant_id conflicts with canonical identity"
        )
    if scope.kind is MemoryControlPlaneScope.USER:
        canonical_user = (identity.user_id or "").strip()
        scope_user = (scope.user_id or "").strip()
        if not scope_user or scope_user != canonical_user:
            raise MemoryControlAccessDenied(
                "user memory scope conflicts with canonical user_id"
            )
    if scope.kind is MemoryControlPlaneScope.SESSION:
        if not (scope.session_id or "").strip():
            raise MemoryControlAccessDenied("session scope requires session_id")
