# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Scoped credential grant/scope validation (AW-7C P0-2)."""

from __future__ import annotations

from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialScopeMismatchError,
    CredentialUseGrant,
    CredentialUseGrantExpiredError,
    CredentialUseScope,
)
from intergrax.runtime.sandbox.network_egress import NetworkEgressAllowlist
from intergrax.utils.time_provider import TimeProvider


def requested_target_scope_within_grant(
    requested: NetworkEgressAllowlist,
    grant_scope: NetworkEgressAllowlist,
) -> bool:
    """True when every requested host is explicitly covered by the grant scope."""
    if not requested.hosts or not grant_scope.hosts:
        return False
    grant_keys = {host.canonical_form() for host in grant_scope.hosts}
    requested_keys = {host.canonical_form() for host in requested.hosts}
    return requested_keys.issubset(grant_keys)


def validate_scoped_use_scope(scope: CredentialUseScope) -> None:
    """Fail closed on incomplete scoped resolution context."""
    if not scope.tenant_id.strip():
        raise CredentialScopeMismatchError("scoped credential resolution requires tenant context")
    if not scope.provider_id.strip():
        raise CredentialScopeMismatchError("scoped credential resolution requires provider identity")
    if not scope.integration_id.strip():
        raise CredentialScopeMismatchError("scoped credential resolution requires integration identity")
    if not scope.operation.strip():
        raise CredentialScopeMismatchError("scoped credential resolution requires operation")
    if not scope.execution_id.strip():
        raise CredentialScopeMismatchError("scoped credential resolution requires execution identity")
    if not scope.target_scope.hosts:
        raise CredentialScopeMismatchError("scoped credential resolution requires non-empty target scope")


def validate_credential_use_grant(grant: CredentialUseGrant) -> None:
    """Fail closed on incomplete grant authority."""
    if not grant.grant_id.strip():
        raise CredentialScopeMismatchError("credential use grant requires grant identity")
    if not grant.tenant_id.strip():
        raise CredentialScopeMismatchError("credential use grant requires tenant context")
    if not grant.provider_id.strip():
        raise CredentialScopeMismatchError("credential use grant requires provider identity")
    if not grant.integration_id.strip():
        raise CredentialScopeMismatchError("credential use grant requires integration identity")
    if not grant.operation.strip():
        raise CredentialScopeMismatchError("credential use grant requires operation")
    if not grant.execution_id.strip():
        raise CredentialScopeMismatchError("credential use grant requires execution identity")
    if not grant.target_scope.hosts:
        raise CredentialScopeMismatchError("credential use grant requires non-empty target scope")
    if grant.expires_at.tzinfo is None:
        raise CredentialScopeMismatchError("credential use grant expiry must be timezone-aware")


def assert_grant_not_expired(
    grant: CredentialUseGrant,
    *,
    time_provider: type[TimeProvider],
) -> None:
    now = time_provider.utc_now()
    if grant.expires_at <= now:
        raise CredentialUseGrantExpiredError("credential use grant has expired")


def assert_tenant_consistency(
    *,
    ref: CredentialRef,
    grant: CredentialUseGrant,
    scope: CredentialUseScope,
) -> None:
    context_tenant = scope.tenant_id.strip()
    grant_tenant = grant.tenant_id.strip()
    if grant_tenant != context_tenant:
        raise CredentialScopeMismatchError(
            "credential use grant tenant does not match resolution scope",
        )
    if ref.tenant_id is not None and ref.tenant_id.strip() != context_tenant:
        raise CredentialScopeMismatchError(
            "credential reference tenant does not match resolution scope",
        )


def assert_grant_matches_scope(grant: CredentialUseGrant, scope: CredentialUseScope) -> None:
    if grant.execution_id.strip() != scope.execution_id.strip():
        raise CredentialScopeMismatchError(
            "credential use grant execution does not match resolution scope",
        )
    if grant.operation.strip() != scope.operation.strip():
        raise CredentialScopeMismatchError(
            "credential use grant operation does not match resolution scope",
        )
    if grant.provider_id.strip() != scope.provider_id.strip():
        raise CredentialScopeMismatchError(
            "credential use grant provider does not match resolution scope",
        )
    if grant.integration_id.strip() != scope.integration_id.strip():
        raise CredentialScopeMismatchError(
            "credential use grant integration does not match resolution scope",
        )
    if grant.credential_ref.provider_id.strip() != scope.provider_id.strip():
        raise CredentialScopeMismatchError(
            "credential reference provider does not match resolution scope",
        )
    if not requested_target_scope_within_grant(scope.target_scope, grant.target_scope):
        raise CredentialScopeMismatchError(
            "requested target scope is not covered by credential use grant",
        )


def assert_grant_provider_matches_ref(grant: CredentialUseGrant) -> None:
    if grant.credential_ref.provider_id.strip() != grant.provider_id.strip():
        raise CredentialScopeMismatchError(
            "credential use grant provider does not match credential reference",
        )
