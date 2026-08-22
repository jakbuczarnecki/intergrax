# © Artur Czarnecki. All rights reserved.

"""Canonical RequestIdentity spine helpers (IDT-FIX-A)."""

from __future__ import annotations

from typing import Any, Mapping

from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.integrations.contracts.identity_provider import IdentityUser


def identity_user_to_request_identity(
    user: IdentityUser,
    *,
    tenant_required: bool = False,
    default_tenant_id: str = "default",
) -> RequestIdentity:
    """Map provider-normalized IdentityUser to canonical RequestIdentity."""
    tenant_id = (user.tenant_id or "").strip()
    if not tenant_id:
        if tenant_required:
            raise ValueError("Verified principal lacks tenant_id")
        tenant_id = default_tenant_id
    user_id = user.user_id.strip()
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def api_key_service_request_identity(
    *,
    tenant_id: str | None,
    service_id: str,
    tenant_required: bool = False,
    default_tenant_id: str = "default",
) -> RequestIdentity:
    """Host-owned API-key service principal — never caller-selected tenant/user."""
    resolved_tenant = (tenant_id or "").strip()
    if not resolved_tenant:
        if tenant_required:
            raise ValueError("API key service principal tenant is not configured")
        resolved_tenant = default_tenant_id
    resolved_service = service_id.strip() or "harness-api-key"
    return RequestIdentity(
        tenant_id=resolved_tenant,
        user_id=resolved_service,
        principal_type=PrincipalType.SERVICE,
        auth_subject=resolved_service,
    )


def assert_untrusted_metadata_identity_compatible(
    canonical: RequestIdentity,
    metadata: Mapping[str, Any],
) -> None:
    """Fail closed when legacy metadata tries to override canonical RequestIdentity."""
    if "tenant_id" in metadata:
        meta_tenant = metadata.get("tenant_id")
        if meta_tenant is not None and str(meta_tenant) != canonical.tenant_id:
            raise ValueError(
                "metadata tenant_id conflicts with canonical RequestIdentity"
            )
    if "user_id" in metadata:
        meta_user = metadata.get("user_id")
        if meta_user is not None and str(meta_user) != (canonical.user_id or ""):
            raise ValueError(
                "metadata user_id conflicts with canonical RequestIdentity"
            )
    if "principal_type" in metadata:
        meta_principal = metadata.get("principal_type")
        if meta_principal is not None:
            try:
                asserted = PrincipalType(str(meta_principal))
            except ValueError as exc:
                raise ValueError(
                    "metadata principal_type conflicts with canonical RequestIdentity"
                ) from exc
            if asserted != canonical.principal_type:
                raise ValueError(
                    "metadata principal_type conflicts with canonical RequestIdentity"
                )
    if "auth_subject" in metadata:
        meta_subject = metadata.get("auth_subject")
        if meta_subject is not None and str(meta_subject) != (
            canonical.auth_subject or ""
        ):
            raise ValueError(
                "metadata auth_subject conflicts with canonical RequestIdentity"
            )


def request_identity_to_actor_identity(identity: RequestIdentity) -> ActorIdentity:
    """Derive runtime actor projection from canonical RequestIdentity (identity-only)."""
    if identity.principal_type is PrincipalType.USER:
        kind = ActorKind.USER
        actor_id = identity.user_id or identity.auth_subject or "anonymous"
    elif identity.principal_type is PrincipalType.SERVICE:
        kind = ActorKind.SERVICE
        actor_id = identity.auth_subject or identity.user_id or "anonymous"
    else:
        kind = ActorKind.SERVICE
        actor_id = identity.auth_subject or identity.user_id or "anonymous"
    return ActorIdentity(
        kind=kind,
        actor_id=actor_id,
        tenant_id=identity.tenant_id,
        delegated_from=None,
        permission_scopes=(),
    )
