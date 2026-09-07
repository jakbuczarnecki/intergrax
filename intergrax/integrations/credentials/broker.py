# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Purpose-scoped credential broker over canonical resolution (AW-7C P0-2)."""

from __future__ import annotations

from intergrax.integrations.contracts.credential import (
    CredentialResolutionContext,
    CredentialResolver,
    CredentialScopeAdmissionDecision,
    CredentialScopeAdmissionDeniedError,
    CredentialScopeAdmissionPort,
    CredentialUseEvidence,
    CredentialUseGrant,
    CredentialUseScope,
    ScopedCredentialResolutionResult,
)
from intergrax.integrations.credentials.scope_validation import (
    assert_grant_matches_scope,
    assert_grant_not_expired,
    assert_grant_provider_matches_ref,
    assert_tenant_consistency,
    validate_credential_use_grant,
    validate_scoped_use_scope,
)
from intergrax.utils.time_provider import SystemTimeProvider, TimeProvider


class ScopedCredentialBroker:
    """Bounded credential resolution with scope validation and admission policy."""

    def __init__(
        self,
        *,
        resolver: CredentialResolver,
        admission: CredentialScopeAdmissionPort,
        time_provider: type[TimeProvider] = SystemTimeProvider,
    ) -> None:
        self._resolver = resolver
        self._admission = admission
        self._time_provider = time_provider

    def resolve_scoped(
        self,
        grant: CredentialUseGrant,
        scope: CredentialUseScope,
    ) -> ScopedCredentialResolutionResult:
        validate_scoped_use_scope(scope)
        validate_credential_use_grant(grant)
        assert_grant_provider_matches_ref(grant)
        assert_tenant_consistency(ref=grant.credential_ref, grant=grant, scope=scope)
        assert_grant_matches_scope(grant, scope)
        assert_grant_not_expired(grant, time_provider=self._time_provider)

        decision = self._admission.admit(grant, scope)
        if decision is CredentialScopeAdmissionDecision.UNAVAILABLE:
            raise CredentialScopeAdmissionDeniedError(
                "credential scope admission is unavailable",
            )
        if decision is not CredentialScopeAdmissionDecision.ALLOW:
            raise CredentialScopeAdmissionDeniedError(
                "credential scope admission denied",
            )

        resolution_context = CredentialResolutionContext(
            tenant_id=scope.tenant_id,
            execution_id=scope.execution_id,
            operation=scope.operation,
            integration_id=scope.integration_id,
            target_scope=scope.target_scope,
        )
        resolved = self._resolver.resolve(
            grant.credential_ref,
            context=resolution_context,
        )
        evidence = CredentialUseEvidence(
            credential_ref=grant.credential_ref,
            provider_id=scope.provider_id,
            tenant_id=scope.tenant_id,
            execution_id=scope.execution_id,
            operation=scope.operation,
            integration_id=scope.integration_id,
            target_scope_fingerprint=scope.target_scope.fingerprint(),
            credential_fingerprint=grant.credential_ref.identity_fingerprint(),
            grant_id=grant.grant_id,
            resolved_version=resolved.resolved_version,
        )
        return ScopedCredentialResolutionResult(
            resolved_credential=resolved,
            use_evidence=evidence,
        )
