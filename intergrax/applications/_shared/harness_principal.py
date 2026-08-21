# © Artur Czarnecki. All rights reserved.

"""Verified harness principal → RequestIdentity bridge (IDT-FIX-A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException, status

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.request_identity_spine import (
    api_key_service_request_identity,
    identity_user_to_request_identity,
    request_identity_to_actor_identity,
)
from intergrax.integrations.contracts.identity_provider import IdentityUser

__all__ = [
    "HarnessAuthenticatedPrincipal",
    "api_key_service_harness_principal",
    "assert_untrusted_metadata_identity_compatible",
    "harness_principal_to_request_identity",
    "identity_user_to_harness_principal",
    "reject_identity_assertion_conflicts",
    "request_identity_to_actor_identity",
]


@dataclass(frozen=True, slots=True)
class HarnessAuthenticatedPrincipal:
    """Canonical verified principal for authenticated Tier-3 harness surfaces."""

    tenant_id: str
    user_id: str
    principal_type: PrincipalType
    auth_subject: str
    auth_mode: Literal["identity_provider", "api_key"]


def identity_user_to_harness_principal(
    user: IdentityUser,
    *,
    tenant_required: bool = False,
) -> HarnessAuthenticatedPrincipal:
    try:
        identity = identity_user_to_request_identity(
            user,
            tenant_required=tenant_required,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    return HarnessAuthenticatedPrincipal(
        tenant_id=identity.tenant_id,
        user_id=identity.user_id or user.user_id,
        principal_type=identity.principal_type,
        auth_subject=identity.auth_subject or identity.user_id or user.user_id,
        auth_mode="identity_provider",
    )


def api_key_service_harness_principal(
    *,
    tenant_id: str | None,
    service_id: str,
    tenant_required: bool = False,
) -> HarnessAuthenticatedPrincipal:
    try:
        identity = api_key_service_request_identity(
            tenant_id=tenant_id,
            service_id=service_id,
            tenant_required=tenant_required,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    return HarnessAuthenticatedPrincipal(
        tenant_id=identity.tenant_id,
        user_id=identity.user_id or service_id,
        principal_type=identity.principal_type,
        auth_subject=identity.auth_subject or service_id,
        auth_mode="api_key",
    )


def harness_principal_to_request_identity(
    principal: HarnessAuthenticatedPrincipal,
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        principal_type=principal.principal_type,
        auth_subject=principal.auth_subject,
    )


def reject_identity_assertion_conflicts(
    *,
    canonical: RequestIdentity,
    asserted_tenant_id: str | None,
    asserted_user_id: str | None,
) -> None:
    """Reject body/context identity assertions that conflict with verified principal."""
    if asserted_tenant_id is not None and asserted_tenant_id != canonical.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tenant_id in request body conflicts with authenticated principal.",
        )
    if asserted_user_id is not None and asserted_user_id != (canonical.user_id or ""):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id in request body conflicts with authenticated principal.",
        )


from intergrax.contracts.request_identity_spine import (  # noqa: E402
    assert_untrusted_metadata_identity_compatible,
)
