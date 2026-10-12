# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R2 adversarial probes (fixture/unit; no broad rescans)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialResolutionContext,
    CredentialScopeAdmissionDecision,
    CredentialScopeMismatchError,
    CredentialUseGrant,
    CredentialUseScope,
    ResolvedCredential,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.credentials.scope_validation import validate_scoped_use_scope
from intergrax.integrations.qualification.reference_scoped_adaptive_integration_execution import (
    ReferenceScopedAdaptedIntegrationEffectExecutionContext,
    ReferenceScopedAdaptedIntegrationEffectExecutor,
    ReferenceScopedAdaptiveIntegrationSandboxSession,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.providers.native_provider_boundary import validate_scope
from intergrax.runtime.sandbox.network_egress import (
    NetworkEgressAllowlist,
    parse_network_egress_host,
)


def _hosts() -> NetworkEgressAllowlist:
    return NetworkEgressAllowlist(hosts=(parse_network_egress_host("https://api.vendor.com:443"),))


def _minimal_grant(tenant: str, ref: CredentialRef) -> CredentialUseGrant:
    expires = datetime(2099, 1, 1, tzinfo=UTC)
    return CredentialUseGrant(
        grant_id="grant-r2",
        credential_ref=ref,
        tenant_id=tenant,
        provider_id="vendor_api",
        integration_id="vendor_api:issue",
        operation="read",
        execution_id="exec-r2",
        target_scope=_hosts(),
        expires_at=expires,
    )


def _minimal_scope(tenant: str) -> CredentialUseScope:
    return CredentialUseScope(
        tenant_id=tenant,
        provider_id="vendor_api",
        integration_id="vendor_api:issue",
        operation="read",
        execution_id="exec-r2",
        target_scope=_hosts(),
    )


class _DenyAdmission:
    def admit(self, grant: object, scope: object) -> CredentialScopeAdmissionDecision:
        return CredentialScopeAdmissionDecision.ALLOW


class _NoResolve:
    def resolve(
        self,
        ref: CredentialRef,
        *,
        context: CredentialResolutionContext,
    ) -> ResolvedCredential:
        raise AssertionError("resolver must not run on denied broker path")


def assert_cross_tenant_credential_ref_rejects() -> None:
    ref = CredentialRef.from_secret_path(
        provider_id="vendor_api",
        secret_path="secrets/tenant-a/key",
        tenant_id="tenant-a",
    )
    broker = ScopedCredentialBroker(resolver=_NoResolve(), admission=_DenyAdmission())
    with pytest.raises(CredentialScopeMismatchError, match="tenant"):
        broker.resolve_scoped(
            _minimal_grant("tenant-b", ref),
            _minimal_scope("tenant-b"),
        )


def assert_missing_tenant_on_credential_scope_rejects() -> None:
    scope = _minimal_scope("tenant-a")
    broken = CredentialUseScope(
        tenant_id="   ",
        provider_id=scope.provider_id,
        integration_id=scope.integration_id,
        operation=scope.operation,
        execution_id=scope.execution_id,
        target_scope=scope.target_scope,
    )
    with pytest.raises(CredentialScopeMismatchError, match="tenant"):
        validate_scoped_use_scope(broken)


def assert_vector_scope_cross_tenant_rejects() -> None:
    pinned = "tenant-a"
    with pytest.raises(Exception, match="differs from provider tenant"):
        validate_scope(VectorStoreScope(tenant_id="tenant-b"), tenant_id=pinned)


def assert_reference_executor_rejects_evidence_tenant_mismatch() -> None:
    from intergrax.contracts.execution_identity import ExecutionId
    from intergrax.integrations.contracts.base import IntegrationCategory
    from intergrax.integrations.contracts.credential import (
        CredentialUseEvidence,
        ResolvedCredential,
        ScopedCredentialResolutionResult,
    )
    from intergrax.integrations.contracts.scoped_integration_adaptation import (
        ScopedAdaptedIntegrationEffectRequest,
        scoped_integration_adaptation_operation_id,
    )
    from intergrax.integrations.qualification.reference_scoped_integration_adaptation import (
        ReferenceScopedIntegrationAdaptationSpecification,
    )

    ref = CredentialRef.from_secret_path(
        provider_id="vendor_api",
        secret_path="secrets/x",
        tenant_id="tenant-a",
    )
    evidence = CredentialUseEvidence(
        credential_ref=ref,
        provider_id="vendor_api",
        tenant_id="tenant-b",
        execution_id="exec-r2",
        operation="read",
        integration_id="vendor_api:issue",
        target_scope_fingerprint=_hosts().fingerprint(),
        credential_fingerprint=ref.identity_fingerprint(),
        grant_id="g",
        resolved_version=None,
    )
    resolution = ScopedCredentialResolutionResult(
        resolved_credential=ResolvedCredential(ref=ref, value="secret", resolved_version=None),
        use_evidence=evidence,
    )
    spec = ReferenceScopedIntegrationAdaptationSpecification(
        adaptation_intent="r2-probe",
        normalized_resource_scope="vendor_api:issue",
        operation_subset=("read",),
    )
    effect_request = ScopedAdaptedIntegrationEffectRequest(
        artifact_id="art-r2",
        artifact_fingerprint="fp-r2",
        tenant_id="tenant-a",
        integration_category=IntegrationCategory.COLLABORATION_SUITE,
        provider_id="vendor_api",
        resource_scope="vendor_api:issue",
        requested_operation=scoped_integration_adaptation_operation_id("read"),
        network_allowlist=_hosts(),
        target_scope=_hosts(),
        specification=spec,
        execution_id="exec-r2",
    )
    ingress = ReferenceScopedAdaptedIntegrationEffectExecutionContext(
        effect_request=effect_request,
        bound_execution_id=ExecutionId("exec-r2"),
        sandbox_security_source=ReferenceScopedAdaptiveIntegrationSandboxSession(
            capabilities=SandboxSecurityCapabilities(
                isolation_tier="local",
                provider_id="tenant-x-r2-probe",
                network_egress_allowlist_enforced=True,
                enforced_network_hosts=_hosts(),
            ),
        ),
        credential_resolution=resolution,
    )
    executor = ReferenceScopedAdaptedIntegrationEffectExecutor()
    with pytest.raises(ValueError, match="credential evidence tenant mismatch"):
        executor.execute(ingress)
